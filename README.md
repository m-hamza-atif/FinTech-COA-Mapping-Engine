# Mapping Engine Microservice

Developed by the **FAST-NUCES Islamabad AI/NLP team** for the **BCU International FinTech Competition 2026**. The service maps source ledger accounts to the target chart of accounts (COA). It runs on port `8001` alongside Fintech-Backend on port `8000`.

### Links

Fintech-Mapping-Engine Docker Image: https://hub.docker.com/r/mhamzaatif/fintech-mapping-engine

Fintech-Backend Docker Image: https://hub.docker.com/r/mhuzaima/fintech-backend

## How it works

| Routing path | Behavior |
|---|---|
| Skip / `EXCLUDED` | An account marked `exclude=true` or missing a description immediately returns null targets, zero confidence, and no alternatives. |
| Deterministic / `AUTO_APPROVED` | Strong hybrid matches meeting the 0.85 score and supporting evidence thresholds, or unanimous historical matches meeting their threshold, bypass Gemini when there are no red flags. The offline BGE Small benchmark averaged 0.74 seconds per 100 held-out accounts. Deterministic approvals achieved 84.62% precision in the final held-out evaluation. |
| Ambiguous or red-flagged / Gemini reranking | Low-margin matches and terms such as intercompany, shareholder, reimbursable, FX, and suspense go to paced Gemini batches when enabled using the example environment settings. The primary is `gemini-3.6-flash` with `gemini-3.5-flash-lite` as fallback. Gemini chooses from 12 candidates, gives a short reason, and has its confidence capped at 0.84. |
| Human review / `NEEDS_REVIEW` | The backend forwards the suggested mapping and ordered alternatives to the UI for accountant review. Show the first three alternatives from the up-to-eight entries in `top_k_candidates`. When Gemini is unavailable, the engine suggests the highest-ranked local match. |

## Architecture

Hierarchy-aware hybrid retrieval combines `BAAI/bge-small-en-v1.5` embeddings, BM25 keyword scores, and ChromaDB indexes. Target COA parent breadcrumbs help distinguish similar account names. The `few_shot_kb` history supplies examples and prioritizes targets observed in prior mappings. Raw-name retrieval scored lower in the benchmark. The existing COA hierarchy and historical pairs supplied the relationships needed for this workflow without building and maintaining a separate knowledge graph.

## Held-out benchmark

| Metric | Result |
|---|---:|
| Top-1 accuracy | 30.83% |
| Top-3 recall | 45.86% |
| Exclusion accuracy | 100% (340/340) |
| Auto-approval precision | 84.62% |
| Paired Gemini Top-1 uplift | +14.06 percentage points |

Top-1 and Top-3 use the 399 labeled held-out accounts. Exclusions were audited separately. The paired uplift compares local retrieval with Gemini on the same 185 usable decisions.

[Benchmark history and final paced evaluation](reports/benchmark_report.md) · [Final benchmark JSON](reports/benchmark_summary.json)

## Run locally

```sh
uv sync
uv run uvicorn app.main:app --port 8001
```

Use [.env.example](.env.example) to create a local `.env` if you want Gemini reranking. Set `GEMINI_API_KEY` and keep the release defaults `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` and `RERANKER_MODEL=off`. The service can map accounts locally with an empty API key.

## Run with Docker

Build and run from the repository root.

```sh
docker build -t fintech-mapping-engine:v0.2.0 .
docker run --rm -p 127.0.0.1:8001:8001 fintech-mapping-engine:v0.2.0
```

To enable Gemini in Docker, add `--env-file .env` before the image name. The container includes the embedding weights and index.

## Verify and connect the backend

```sh
curl http://127.0.0.1:8001/health
uv run python scripts/smoke_e2e.py
```

Point Fintech-Backend at `http://127.0.0.1:8001` with `POST /api/v1/mappings/map-accounts` and contract profile `complete-v1`. With the backend running on port `8000`, add `--backend-url http://127.0.0.1:8000` to the smoke test to verify the upload, mapping, and persistence workflow.

## Run the CSV demo

Run the v0.2.0 engine in local retrieval mode with an empty Gemini API key. Upload [demo_input.csv](demo/demo_input.csv) through Fintech-Backend using entity `PMC_01` and target schema `entity_1_ucoa`, then run mapping.

1. Approve Interest Receivable (`1618-9010`) for source `1320-200`.
2. Correct source `1601-030` to Credit Accounts Receivable Balances (`1610-1911`) to exercise the correction workflow. This example correction is pending accounting validation.
3. Record exclusion decisions for `999900` and `000004`. The first has an exclusion flag and the second has no description.
4. Download the reviewed CSV for that mapping run and compare it with [expected_reviewed_export.csv](demo/expected_reviewed_export.csv). Import account codes as text in Excel to preserve `000004`.
