"""Exercise the live mapping engine and optionally verify backend persistence."""

import argparse
import csv
import io
import sys
from pathlib import Path

import httpx
from pydantic import ValidationError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.contracts import Account, AIInput, AIOutput, MappingResult

ENGINE_PATH = "/api/v1/mappings/map-accounts"
HAPPY_TARGET = "6010-1650"
SAMPLE = AIInput(
    entity_id="demo-property",
    target_schema="entity_1_ucoa",
    accounts=[
        Account(
            source_code="510100",
            source_name="Asset Management Fees",
            normal_balance=None,
            ending_balance=12500.0,
            exclude=False,
        ),
        Account(
            source_code="999900",
            source_name="Dormant ledger control",
            normal_balance=None,
            ending_balance=0.0,
            exclude=True,
        ),
        Account(
            source_code="210050",
            source_name="Intercompany Loan Receivable",
            normal_balance="Debit",
            ending_balance=42000.0,
            exclude=False,
        ),
    ],
)


def check_output(output: AIOutput):
    expected = [(row.source_code, row.source_name) for row in SAMPLE.accounts]
    assert [
        (row.source_code, row.source_name) for row in output.mappings
    ] == expected, "Mappings must preserve account identity and order."
    approved, omitted, flagged = output.mappings
    assert approved.routing_status == "AUTO_APPROVED", (
        "The Happy Path fixture needs review."
    )
    assert approved.predicted_target_code == HAPPY_TARGET, (
        "Unexpected Happy Path target."
    )
    assert omitted.routing_status == "EXCLUDED", "The excluded account was mapped."
    assert (
        omitted.predicted_target_code,
        omitted.predicted_target_name,
        omitted.target_hierarchy,
        omitted.confidence_score,
        omitted.top_k_candidates,
    ) == (None, None, None, 0.0, []), (
        "The excluded account has populated mapping fields."
    )
    assert flagged.routing_status == "NEEDS_REVIEW", (
        "The red-flag account was auto-approved."
    )
    assert flagged.predicted_target_code is not None and flagged.top_k_candidates, (
        "The red-flag account should have a target suggestion and candidates."
    )
    assert flagged.confidence_score <= 0.84, (
        "The red-flag confidence exceeds the review cap."
    )


def check_engine(client: httpx.Client, url: str):
    health = client.get(f"{url}/health").raise_for_status().json()
    assert health == {"status": "ok", "service": "mapping-engine"}, (
        "Unexpected engine health response."
    )
    response = client.post(
        f"{url}{ENGINE_PATH}", json=SAMPLE.model_dump()
    ).raise_for_status()
    assert response.headers.get("Content-Encoding", "identity").lower() == "identity", (
        "The backend requires an uncompressed mapping response."
    )
    check_output(AIOutput.model_validate_json(response.content))
    print("Engine verified Happy Path, strict exclusion, and red-flag review.")


def sample_csv():
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "SourceAccountCode",
            "SourceAccountName",
            "Normal Balance",
            "Ending Balance",
            "Exclude",
        ]
    )
    for account in SAMPLE.accounts:
        writer.writerow(
            [
                account.source_code,
                account.source_name,
                account.normal_balance,
                account.ending_balance,
                int(account.exclude),
            ]
        )
    return stream.getvalue().encode("utf-8")


def check_backend(client: httpx.Client, url: str):
    batch = (
        client.post(
            f"{url}/api/v1/uploads",
            files={"file": ("demo-coa.csv", sample_csv(), "text/csv")},
            data={"entity_id": SAMPLE.entity_id, "target_schema": SAMPLE.target_schema},
        )
        .raise_for_status()
        .json()
    )
    batch_id = batch["batch_id"]
    run = (
        client.post(f"{url}/api/v1/mappings/predict", json={"batch_id": batch_id})
        .raise_for_status()
        .json()
    )
    assert run["status"] == "completed", "Backend prediction run did not complete."
    assert run["contract_profile"] == "complete-v1", "Unexpected mapping contract."
    assert run["prediction_count"] == len(SAMPLE.accounts), (
        "Backend did not save every result."
    )

    page = (
        client.get(
            f"{url}/api/v1/mappings",
            params={"batch_id": batch_id, "run_id": run["run_id"]},
        )
        .raise_for_status()
        .json()
    )
    assert page["total"] == len(SAMPLE.accounts), "Unexpected saved prediction count."
    by_code = {row["source"]["source_code"]: row for row in page["items"]}
    assert set(by_code) == {account.source_code for account in SAMPLE.accounts}, (
        "Saved source codes differ."
    )
    results = []
    for account in SAMPLE.accounts:
        row = by_code[account.source_code]
        results.append(
            MappingResult.model_validate(
                {
                    key: row["source"][key]
                    if key in {"source_code", "source_name"}
                    else row[key]
                    for key in MappingResult.model_fields
                }
            )
        )
    check_output(AIOutput(mappings=results))
    print(
        f"Backend verified upload, live prediction, and persisted results for run {run['run_id']}."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--backend-url", help="Also upload and persist a demo batch on this backend"
    )
    args = parser.parse_args()
    try:
        with httpx.Client(
            timeout=125.0, follow_redirects=False, trust_env=False
        ) as client:
            check_engine(client, args.engine_url.rstrip("/"))
            if args.backend_url:
                check_backend(client, args.backend_url.rstrip("/"))
    except (
        AssertionError,
        httpx.HTTPError,
        KeyError,
        ValueError,
        ValidationError,
    ) as error:
        parser.exit(1, f"Smoke check failed: {error}\n")


if __name__ == "__main__":
    main()
