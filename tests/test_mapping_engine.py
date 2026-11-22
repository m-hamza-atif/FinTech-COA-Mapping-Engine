import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from google.genai import errors
from pydantic import ValidationError

from app import main
from app.schemas.contracts import AIInput, AIOutput
from app.services.indexer import Candidate, SearchIndex, Target
from app.services.llm_reranker import Decision, GeminiReranker, RerankInput
from app.services.normalizer import Normalizer
from app.services.retriever import LocalReranker, MappingEngine, happy_path

CANDIDATES = [
    Candidate(
        Target("1010", "Building costs", "Assets > Property", "Debit"), 0.94, 0.94, 0.91
    ),
    Candidate(Target("1020", "Building repairs", "Expenses", "Debit"), 0.72),
]


def account(code, name, exclude=False):
    return {
        "source_code": code,
        "source_name": name,
        "normal_balance": None,
        "ending_balance": None,
        "exclude": exclude,
    }


def payload(*accounts):
    return {
        "entity_id": "sample",
        "target_schema": "entity_1_ucoa",
        "accounts": list(accounts),
    }


class StubIndex:
    def embed(self, texts):
        self.texts = texts
        return [[1.0] for _ in texts]

    def search_batch(self, queries, vectors):
        return (CANDIDATES for _ in queries)

    def similar_examples(self, vectors):
        return [[] for _ in vectors]


def test_endpoint_exclusions_and_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "load_dotenv", lambda *_: None)
    index = StubIndex()
    reranker = SimpleNamespace(enabled=True, rerank=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(main, "SearchIndex", lambda *_: index)
    monkeypatch.setattr(main, "GeminiReranker", lambda *_, **__: reranker)
    monkeypatch.setenv("MAPPING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RERANKER_MODEL", "off")
    with TestClient(main.app, follow_redirects=False) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "service": "mapping-engine",
        }
        path = "/api/v1/mappings/map-accounts"
        request = payload(
            account("001", "  Bldg Costs  "),
            account("002", "Explicit exclusion", True),
            account("003", "   "),
            account("004", None),
        )
        response = client.post(path, json=request, headers={"Accept-Encoding": "gzip"})
        assert response.status_code == 200
        assert response.headers["Content-Encoding"] == "identity"
        assert client.post(path + "/", json=request).status_code == 404
        assert (
            client.post(
                path, json=payload(account("001", "A"), account("001", "B"))
            ).status_code
            == 422
        )

    rows = response.json()["mappings"]
    assert [row["source_code"] for row in rows] == ["001", "002", "003", "004"]
    assert [row["source_name"] for row in rows] == [
        row["source_name"] for row in request["accounts"]
    ]
    assert rows[0]["routing_status"] == "AUTO_APPROVED"
    assert rows[0]["predicted_target_code"] == "1010"
    assert index.texts == ["building costs"]
    reranker.rerank.assert_not_awaited()
    for row in rows[1:]:
        assert row["routing_status"] == "EXCLUDED"
        assert row["confidence_score"] == 0
        assert row["top_k_candidates"] == []
        assert all(
            row[key] is None
            for key in (
                "predicted_target_code",
                "predicted_target_name",
                "target_hierarchy",
            )
        )

    AIOutput.model_validate(response.json())
    for field, invalid in {
        "unexpected": 1,
        "source_code": 1,
        "confidence_score": float("nan"),
        "predicted_target_name": "",
    }.items():
        with pytest.raises(ValidationError):
            AIOutput.model_validate({"mappings": [{**rows[0], field: invalid}]})


def test_red_flag_invokes_structured_rerank_and_caps_confidence(tmp_path):
    generate = AsyncMock(
        return_value=SimpleNamespace(
            text=json.dumps(
                {
                    "decisions": [
                        {
                            "index": 0,
                            "chosen_code": "1020",
                            "llm_confidence": 1.0,
                            "reasoning": "The source describes repair costs.",
                        }
                    ],
                }
            )
        )
    )
    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    reranker = GeminiReranker(None, "test-model", client=client)
    engine = MappingEngine(Normalizer(tmp_path), StubIndex(), reranker)
    batch = AIInput.model_validate(payload(account("005", "Interco bldg")))
    row = asyncio.run(engine.map_accounts(batch.accounts))[0]
    assert row.routing_status == "NEEDS_REVIEW"
    assert row.confidence_score == 0.84
    assert row.predicted_target_code == "1020"
    sent = json.loads(generate.await_args.kwargs["contents"])
    assert sent[0]["normalized_name"] == "intercompany building"
    assert generate.await_args.kwargs["config"].response_mime_type == "application/json"


def test_rate_limit_retries_with_batched_fallback(tmp_path):
    async def run():
        async def rate_limit(**kwargs):
            raise errors.ClientError(429, {"error": {"message": "Rate limited"}})

        generate = AsyncMock(side_effect=rate_limit)
        client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        )
        reranker = GeminiReranker(None, "test-model", client=client)
        engine = MappingEngine(Normalizer(tmp_path), StubIndex(), reranker)
        batch = AIInput.model_validate(
            payload(*(account(str(i), "Loan") for i in range(13)))
        )
        rows = await engine.map_accounts(batch.accounts)
        assert len(rows) == 13
        assert {row.routing_status for row in rows} == {"NEEDS_REVIEW"}
        assert {row.predicted_target_code for row in rows} == {"1010"}
        sizes = [
            len(json.loads(call.kwargs["contents"]))
            for call in generate.await_args_list
        ]
        assert sizes == [5, 5]
        assert generate.await_args_list[1].kwargs["model"] == "gemini-3.5-flash-lite"
        assert reranker.response_issues["cooldown_waits"] == 1

    asyncio.run(run())


def test_high_confidence_gemini_cannot_approve_nonflagged_account(
    tmp_path, monkeypatch
):
    index = StubIndex()
    close_candidates = [CANDIDATES[0], Candidate(CANDIDATES[1].target, 0.93)]
    monkeypatch.setattr(
        index,
        "search_batch",
        lambda queries, vectors: (close_candidates for _ in queries),
    )
    decision = Decision(
        index=0, chosen_code="1010", llm_confidence=1.0, reasoning="Building costs."
    )
    reranker = SimpleNamespace(
        enabled=True, rerank=AsyncMock(return_value={0: decision})
    )
    engine = MappingEngine(Normalizer(tmp_path), index, reranker)
    batch = AIInput.model_validate(payload(account("005", "Building costs")))
    row = asyncio.run(engine.map_accounts(batch.accounts))[0]
    assert row.routing_status == "NEEDS_REVIEW"
    assert row.confidence_score == 0.84
    assert row.predicted_target_code == "1010"
    reranker.rerank.assert_awaited_once()


def test_missing_key_keeps_ambiguous_candidates(tmp_path):
    reranker = GeminiReranker(None, "test-model")
    engine = MappingEngine(Normalizer(tmp_path), StubIndex(), reranker)
    batch = AIInput.model_validate(payload(account("005", "Loans")))
    row = asyncio.run(engine.map_accounts(batch.accounts))[0]
    assert row.routing_status == "NEEDS_REVIEW"
    assert row.predicted_target_code == "1010"
    assert len(row.top_k_candidates) == 2


def test_partial_gemini_response_retries_only_unresolved_accounts():
    def response(index, code):
        return SimpleNamespace(
            text=json.dumps(
                {
                    "decisions": [
                        {
                            "index": index,
                            "chosen_code": code,
                            "llm_confidence": 0.9,
                            "reasoning": "Costs.",
                        }
                    ]
                }
            )
        )

    generate = AsyncMock(side_effect=[response(0, "10-10"), response(1, "1020")])
    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    reranker = GeminiReranker(None, client=client)
    item = RerankInput("Building", "building", "Debit", CANDIDATES, [])
    results = asyncio.run(reranker.rerank([item, item], timeout=1))
    assert {key: item.chosen_code for key, item in results.items()} == {
        0: "1010",
        1: "1020",
    }
    assert [
        len(json.loads(call.kwargs["contents"])) for call in generate.await_args_list
    ] == [2, 1]
    assert reranker.response_issues["missing_indices"] == 1


def test_unanimous_vote_approves_only_without_red_flags():
    candidate = Candidate(CANDIDATES[0].target, 0.72, unanimous_vote=True)
    assert happy_path([candidate], has_red_flag=False)
    assert not happy_path([candidate], has_red_flag=True)


def test_cross_encoder_shortlist_breaks_ties_by_target_code(tmp_path):
    from contextlib import nullcontext

    pool = [
        Candidate(Target(f"{number:04}", "Cash", "Assets", "Debit"), 0.7)
        for number in range(25, 0, -1)
    ]
    cross = SimpleNamespace(
        lock=nullcontext(),
        model=SimpleNamespace(predict=lambda pairs, **_: [0.5] * len(pairs)),
    )
    local = SimpleNamespace(rank=lambda rows: LocalReranker.rank(cross, rows))
    index = SimpleNamespace(embed=StubIndex().embed, search_batch=lambda *_: [pool])
    engine = MappingEngine(Normalizer(tmp_path), index, GeminiReranker(None), local)
    accounts = AIInput.model_validate(payload(account("51", "Cash"))).accounts
    observed = []
    rows = asyncio.run(
        engine.map_accounts(
            accounts, observe_pool=lambda _, candidates: observed.extend(candidates)
        )
    )
    assert len(observed) == 25
    assert rows[0].predicted_target_code == "0001"
    assert [candidate.code for candidate in rows[0].top_k_candidates] == [
        f"{number:04}" for number in range(1, 9)
    ]
    assert rows[0].routing_status == "NEEDS_REVIEW"


def test_gemini_can_select_twelfth_candidate_without_expanding_public_alternatives(
    tmp_path,
):
    pool = [
        Candidate(Target(f"{number:04}", "Cash", "Assets", "Debit"), 0.7)
        for number in range(1, 13)
    ]
    rerank = AsyncMock(
        return_value={
            0: Decision(
                index=0, chosen_code="0012", llm_confidence=0.9, reasoning="Evidence."
            )
        }
    )
    gemini = SimpleNamespace(enabled=True, rerank=rerank)
    index = StubIndex()
    index.search_batch = lambda *_: [pool]
    engine = MappingEngine(Normalizer(tmp_path), index, gemini)
    accounts = AIInput.model_validate(payload(account("51", "Cash"))).accounts
    (row,) = asyncio.run(engine.map_accounts(accounts))
    assert len(rerank.await_args.args[0][0].candidates) == 12
    assert row.predicted_target_code == "0012"
    assert len(row.top_k_candidates) == 8
    assert row.routing_status == "NEEDS_REVIEW"


def test_service_error_retries_on_fallback_model():
    success = SimpleNamespace(
        text=json.dumps(
            {
                "decisions": [
                    {
                        "index": 0,
                        "chosen_code": "1010",
                        "llm_confidence": 0.9,
                        "reasoning": "Costs.",
                    }
                ]
            }
        )
    )
    generate = AsyncMock(
        side_effect=[
            errors.ServerError(503, {"error": {"message": "Unavailable"}}),
            success,
        ]
    )
    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    reranker = GeminiReranker(None, client=client)
    item = RerankInput("Building", "building", "Debit", CANDIDATES, [])
    decisions = asyncio.run(reranker.rerank([item], timeout=1))
    assert decisions[0].model == "gemini-3.5-flash-lite"
    assert [call.kwargs["model"] for call in generate.await_args_list] == [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ]


def test_deadline_keeps_completed_chunk_results():
    async def generate(**kwargs):
        records = json.loads(kwargs["contents"])
        if records[0]["index"]:
            await asyncio.Event().wait()
        return SimpleNamespace(
            text=json.dumps(
                {
                    "decisions": [
                        {
                            "index": 0,
                            "chosen_code": "1010",
                            "llm_confidence": 0.9,
                            "reasoning": "Costs.",
                        }
                    ]
                }
            )
        )

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    reranker = GeminiReranker(None, "test-model", client=client)
    item = RerankInput("Building", "building", "Debit", CANDIDATES, [])
    assert set(asyncio.run(reranker.rerank([item] * 13, timeout=0.1))) == {0}


def test_persistent_index_and_balance_filter(tmp_path):
    class EmbeddingModel:
        def encode(self, documents, **kwargs):
            return np.array(
                [[1.0, 0.0] if "cash" in doc else [0.0, 1.0] for doc in documents]
            )

    targets = [
        {
            "target_code": "1",
            "target_name": "Cash",
            "parent_path": "Assets",
            "normal_balance": "Debit",
            "indexed_text": "cash",
        },
        {
            "target_code": "2",
            "target_name": "Rent",
            "parent_path": "Revenue",
            "normal_balance": "Credit",
            "indexed_text": "rent",
        },
    ]
    example = {"source_name": "Cash at Bank", "target_code": "1", "target_name": "Cash"}
    for filename, rows in [
        ("target_coa.jsonl", targets),
        ("few_shot_kb.jsonl", [example]),
    ]:
        (tmp_path / filename).write_text(
            "\n".join(map(json.dumps, rows)), encoding="utf-8"
        )

    normalizer = Normalizer(tmp_path)
    for _ in range(2):
        index = SearchIndex(
            tmp_path, tmp_path / "cache", normalizer, model=EmbeddingModel()
        )
        batch = AIInput.model_validate(payload(account("5", "Cash")))
        query = normalizer.account_query(batch.accounts[0])
        vectors = index.embed([query.text])
        candidates = next(index.search_batch([query], vectors))
        assert [candidate.target.code for candidate in candidates] == ["1"]
        assert candidates[0].historical_similarity == pytest.approx(1.0)
        assert candidates[0].active_count == 1
        assert index.active_priors.tolist() == pytest.approx([0.08, 0.0])
        assert index.similar_examples(vectors) == [[example]]
        unanimous = index.search(query, np.array([0.7, 0.7]), [(0, 0.85)] * 3)
        assert unanimous[0].vote_similarity == 0.85
        assert happy_path(unanimous, has_red_flag=False)
        rent = normalizer.account_query(
            AIInput.model_validate(payload(account("6", "Rent"))).accounts[0]
        )
        fallback = next(index.search_batch([rent], index.embed([rent.text])))
        assert fallback[0].target.code == "2"
