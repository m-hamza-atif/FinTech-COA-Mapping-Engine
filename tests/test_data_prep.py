import json
from zipfile import ZipFile

import pytest
from openpyxl import Workbook

from scripts.prepare_datasets import (
    ARTIFACTS,
    COA_FILE,
    ENTITY_2,
    MAPPING_FILE,
    PMC_DIR,
    TEST_PMCS,
    TRAIN_PMCS,
    prepare,
)


def save_workbook(path, sheets):
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, records in sheets.items():
        sheet = workbook.create_sheet(title)
        for row in records:
            sheet.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


@pytest.fixture
def datasets(tmp_path):
    raw = tmp_path / "Given-Resources"
    files = tmp_path / "fixtures" / ARTIFACTS
    extracted = tmp_path / "data" / "raw_extracted"
    output = tmp_path / "data" / "processed"

    def account(code, name, kind, closing, report="Bal", balance="Debit"):
        return [
            code,
            name,
            balance,
            kind,
            report,
            None,
            None,
            None,
            None,
            None,
            closing,
        ]

    coa = [
        account("1000-0000", "Assets", "Head", "1999-9999"),
        account("1010-0000", "Fixed Assets", "Head", "1049-9999"),
        account("1010-4000", "Capitalized Costs", "Head", "1010-4999"),
        account("1010-4001", "Company Setup Costs", "Head", "1010-5999"),
        account("1010-4010", "Acquisition Fees", "Reg", "1010-4999"),
        account("1010-4999", "Total Capitalized Costs", "Tot", "1999-9999"),
        account("1010-5000", "Setup Costs", "Reg", "1010-5999"),
        account("1010-5999", "Total Company Setup Costs", "Tot", "1999-9999"),
        account("1999-9999", "Total Assets", "Tot", None),
        account("4000-0000", "Rental Revenue", "Head", None, "Inc", "Credit"),
        account("4000-0010", "Rent", "Reg", "4999-9999", "Inc", "Credit"),
        account("4999-9999", "Total Rental Revenue", "Tot", None, "Inc", "Credit"),
    ]
    save_workbook(files / COA_FILE, {"Report1": [[], [], [], [], [], [], *coa]})
    pmc_header = [
        "SourceAccountCode",
        "SourceAccountName",
        "TargetAccountCode",
        "Exclude",
    ]
    for name in TRAIN_PMCS + TEST_PMCS:
        records = [[f"{name}-1", f"{name} Bldg", "10104010", 0]]
        if name in TRAIN_PMCS:
            records.append([f"{name}-2", "Shared CAM", "10104010", 0])
            records.append([f"{name}-3", "  SLFALS1   BLDG ", "10104010", 0])
        if name == "223":
            records.append(["explicit", "Excluded with a name", "10104010", 1])
        if name == "c2720":
            records.append(["blank", "   ", "10104010", 0])
            records.append(["unlabeled", "Missing mapping", None, 0])
        save_workbook(
            files / PMC_DIR / f"{name}.xlsx",
            {"GLAccount_Template": [pmc_header, *records]},
        )
    save_workbook(
        files / ENTITY_2,
        {
            "Mapping Data": [
                [
                    "Code",
                    "PMC GL Description",
                    None,
                    None,
                    None,
                    None,
                    "FM Description",
                ],
                ["A", "FX gains - R&M", None, None, None, "B", "FX losses - R&M"],
                ["B", "Below Mkt Leases", None, None, None, "C", "Mkt-events"],
                ["C", "Acctng - CAD", None, None, None, "D", "WIP - Int renovation"],
            ],
        },
    )
    headers = [
        "Source Company orignal GL#",
        "Source Company GL Description",
        "ERP System GL#",
        "Updated ERP System GL# proposed",
        "Reviewer comments",
    ]
    corrected = [
        [
            str(index),
            f"Intercompany account {index}",
            "1010-5000",
            "1010-4010" if index < 80 else "4000-0010",
            "Check loan and functional currency",
        ]
        for index in range(101)
    ]
    save_workbook(
        raw / MAPPING_FILE,
        {
            "Mapping Final": [
                headers,
                *corrected,
                ["unchanged", "Rent income", "4000-0010", "4000-0010", "Verified"],
                ["no-proposal", "Unmapped", "1010-5000", None, None],
            ],
            "Source COA": [
                [],
                [],
                [],
                [],
                [],
                [],
                *[[row[0], row[1], "Debit"] for row in corrected],
            ],
            "Target COA": [
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                *[[None, None, *row] for row in coa],
            ],
        },
    )
    with ZipFile(raw / f"{ARTIFACTS}.zip", "w") as archive:
        for workbook in files.rglob("*.xlsx"):
            archive.write(workbook, workbook.relative_to(files.parent))

    prepare(raw, extracted, output)
    return raw, extracted, output


def read_jsonl(output, filename):
    return [
        json.loads(line)
        for line in (output / filename).read_text(encoding="utf-8").splitlines()
    ]


def test_target_coa_contains_only_postable_accounts(datasets):
    _, _, output = datasets
    coa = {row["target_code"]: row for row in read_jsonl(output, "target_coa.jsonl")}
    assert set(coa) == {"1010-4010", "1010-5000", "4000-0010"}
    assert (
        coa["1010-4010"]["parent_path"] == "Assets > Fixed Assets > Capitalized Costs"
    )
    assert (
        coa["1010-5000"]["parent_path"] == "Assets > Fixed Assets > Company Setup Costs"
    )
    assert coa["4000-0010"]["indexed_text"] == "Rent — Rental Revenue"
    assert (
        coa["1010-4010"]["indexed_text"]
        == "Acquisition Fees — Capitalized Costs (Assets)"
    )
    assert coa["4000-0010"]["report_type"] == "P&L"


def test_held_out_examples_and_exceptions(datasets):
    _, _, output = datasets
    golden = read_jsonl(output, "golden_test_set.jsonl")
    training = read_jsonl(output, "few_shot_kb.jsonl")
    cases = {row["source_code"]: row for row in golden}
    assert len(golden) == 106
    assert sum(row["expected_target_code"] == "4000-0010" for row in golden) == 21
    assert {row["source_code"] for row in training}.isdisjoint(cases)
    keys = {
        (" ".join(row["source_name"].casefold().split()), row["target_code"])
        for row in training
    }
    assert len(keys) == len(training)
    assert ("slfals1 bldg", "1010-4010") not in keys
    assert {row["source_code"] for row in training} >= {"unchanged", "1000101-1"}
    for code in ("explicit", "blank"):
        assert cases[code]["exclude"] is True
        assert cases[code]["expected_routing_status"] == "EXCLUDED"
        assert cases[code]["expected_target_code"] is None
    assert cases["blank"]["source_name"] is None
    assert cases["unlabeled"]["expected_routing_status"] == "NEEDS_REVIEW"
    assert cases["unlabeled"]["expected_target_code"] is None


def test_rules_and_fallback_are_reproducible(datasets):
    raw, extracted, output = datasets
    abbreviations = json.loads((output / "abbrev_map.json").read_text(encoding="utf-8"))
    flags = json.loads((output / "red_flag_keywords.json").read_text(encoding="utf-8"))
    assert abbreviations == {
        "bldg": "building",
        "cam": "common area maintenance",
        "fx": "foreign exchange",
        "r&m": "repairs and maintenance",
        "mkt leases": "market leases",
        "mkt-events": "marketing events",
        "acctng": "accounting",
        "cad": "canadian dollar",
    }
    assert set(flags) == {"loan", "intercompany", "functional currency"}

    first_run = {file.name: file.read_bytes() for file in output.iterdir()}
    (raw / f"{ARTIFACTS}.zip").unlink()
    (extracted / ARTIFACTS / COA_FILE).unlink()
    prepare(raw, extracted, output)
    assert {file.name: file.read_bytes() for file in output.iterdir()} == first_run
