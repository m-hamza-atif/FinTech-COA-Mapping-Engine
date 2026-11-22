# Mapping Engine Benchmark

This report records completed benchmark iterations. The benchmark script writes replaceable per-run results to `reports/latest/` by default.

Evaluated 1,120 golden accounts across 5,428 postable targets. The set contains 599 labeled mappings, 181 named accounts without a target label, and 340 exclusions.

The table and measurement notes below describe the baselines and the saved iteration 5 run. The accompanying [benchmark JSON](benchmark_summary.json) records the final held-out release evaluation.

| Configuration | Top-1 accuracy | Top-3 recall | Exclusion accuracy | Auto-approve precision | Labeled approvals | Happy Path LLM call reduction | Seconds per 100 accounts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline A — Lexical only | 11.02% | 18.53% | 100.00% | N/A | 0 | N/A | 1.17 |
| Baseline B — Standard RAG | 14.36% | 22.04% | 100.00% | 29.45% | 163 | N/A | 3.30 |
| System C — Hierarchy-aware hybrid RAG | 28.71% | 43.24% | 100.00% | 87.50% | 32 | 4.36% | 14.73 |

## Measurement notes

- Mapping accuracy and recall use labeled, non-excluded rows. Top-3 places the final prediction first, followed by the highest-ranked distinct retrieval candidates.
- Auto-approve precision uses labeled approvals. The baselines apply the 0.85 threshold to saturated BM25 or clipped cosine. System C reserves auto-approval for deterministic Happy Path matches. Gemini reranks stay in review with confidence capped at 0.84.
- System C searches 947 active targets first and widens to the full COA when the best score is below 0.55. Its eight-candidate pool recalled 56.76% of labeled targets.
- The Happy Path gate alone mapped 28 of 32 labeled approvals correctly (87.50%).
- All configurations use the shared exclusion rule. Happy Path call reduction is the share of non-excluded accounts resolved by the hybrid retrieval gate. The baselines have no LLM stage.
- System C used 290 usable Gemini decisions under a cap of 300 accounts. Live API requests totaled 25 and cache hits totaled 257 (257 from earlier runs and 0 from this run). Remaining ambiguous accounts used the hybrid fallback.
- Latency measures in-process mapping across batches of up to 25 with warm indexes and includes live Gemini time when applicable. Cached decisions avoid network time, so compare runs with different cache use accordingly.
- Gemini requests by model were gemini-3.5-flash-lite (15), gemini-3.6-flash (10).
- Gemini attempt diagnostics were cooldown waits (10), http 503 rows (10), http 504 rows (19), primary rate limited rows (43), retried service error rows (19). Counts can overlap across retries.

## Iteration history

All System C iterations below use BAAI/bge-small-en-v1.5 and the same golden evaluation set. The two baselines above provide the reference for each iteration. Each System C run passed all 340 exclusions. The Gemini model names are the API identifiers recorded for the runs.

| Iteration | Top-1 accuracy | Top-3 recall | Labeled auto-approve precision | Happy Path call reduction | Seconds per 100 |
|---|---:|---:|---:|---:|---:|
| 1 - Initial benchmark | 9.35% | 17.20% | 0.00% (0/1) | 0.38% | 2.54 |
| 2 - Retrieval repair | 19.03% | 32.05% | 46.88% (15/32) | 2.31% | 5.82 |
| 3 - Historical evidence and reranking | 23.54% | 42.24% | 50.98% (26/51) | 2.82% | 6.21 |
| 4 - Active-target retrieval | 27.88% | 43.07% | 47.20% (59/125) | 2.31% | 40.82 |
| 5 - Approval and latency calibration | 28.71% | 43.24% | 87.50% (28/32) | 4.36% | 14.73 |
| 6 - Expanded abbreviation dictionary | 28.05% | 43.24% | 87.88% (29/33) | 4.49% | 9.04 |

### 1 - Initial benchmark

Retrieval combined 65% cosine similarity and 35% query-normalized BM25 over full hierarchy text. It used a hard normal-balance filter, a five-candidate pool, a 0.85 approval score and a 0.12 margin. The evaluated Gemini model was `gemini-3.5-flash-lite`, with up to 12 accounts per request. The run returned 10 usable decisions for 12 eligible accounts.

### 2 - Retrieval repair

Canonical target codes and shorter hierarchy text improved candidate matching. The hybrid score used 80% cosine and 20% saturated BM25, with saturation `BM25 / (BM25 + 5)`. Soft normal-balance adjustments replaced the hard filter (0.015 for inferred balances and 0.035 for explicit balances). The gate used a 0.85 hybrid score, 0.08 margin and 0.85 historical similarity threshold. `gemini-3.5-flash-lite` returned 50 usable decisions from 100 selected accounts in eight requests, with up to 15 accounts per request.

### 3 - Historical evidence and reranking

Added a target-frequency prior from seven historical charts, five nearest historical matches, and a direct gate for three unanimous historical votes at similarity 0.88 or higher. Expanded the retrieval pool to eight candidates and added structured output with a partial retry. The run used `gemini-3.8-flash` with `gemini-3.5-flash-lite` as fallback, up to ten accounts per request and a 100-account limit. It used 80 usable decisions, including 79 cached decisions from the same experiment.

### 4 - Active-target retrieval

Searched 947 historically observed targets first and widened to all 5,428 when the best score fell below 0.55. Historical similarity rose to 0.895, with a 0.88 dense threshold and 0.08 margin. `gemini-3.6-flash` replaced the previous primary and retained `gemini-3.5-flash-lite` as fallback. The run used a shorter prompt, five accounts per request and a 300-account limit. It started with a fresh cache and returned 262 usable decisions in 108 API requests. The eight-candidate pool recalled 56.76% of labeled targets.

### 5 - Approval and latency calibration

Kept the same model pair, request size and account limit. A unanimous historical vote at similarity 0.85 or higher could qualify without the margin requirement. Other deterministic approvals retained the 0.85 hybrid score, 0.08 margin, 0.895 historical and 0.88 dense thresholds. Gemini suggestions were capped at 0.84 and routed to human review. The run used 290 usable decisions and 257 prior cache hits, with 25 live requests. Its measured latency includes those cache hits, so it is not a cold-run comparison with iteration 4.

### 6 - Expanded abbreviation dictionary

Expanded the abbreviation dictionary from 51 to 82 entries. Kept iteration 5's retrieval and approval settings, `gemini-3.6-flash` primary, `gemini-3.5-flash-lite` fallback, five accounts per request and a 300-account reranking limit. The run used 185 usable Gemini decisions, including two same-run cache hits and zero prior cache hits. It made 72 API requests (34 primary and 38 fallback). Rate limits, quota skips and service errors affected availability. Remaining accounts used the top local candidate.

## Held-out release evaluation

The release uses a fixed seed-42 split of 200 development accounts and 399 held-out accounts. These scores use the held-out split, so the earlier 599-account results use a different denominator.

| Configuration | Held-out Top-1 | Held-out Top-3 | Held-out shortlist recall |
|---|---:|---:|---:|
| BGE Small, hybrid Top 8, Gemini disabled | 24.31% | 42.36% | 57.39% |
| BGE Base, hybrid Top 8, Gemini disabled | 19.30% | 34.34% | 51.38% |
| BGE Base, cross-encoder reranking from 25 to 8, Gemini disabled | 12.78% | 26.32% | 42.36% |
| Release configuration, BGE Small, Gemini reranking from 12 candidates | 30.83% | 45.86% | 60.65% before Gemini |

The release retained BGE Small and disabled the local cross-encoder. Local Top-1 and Top-3 were 24.31% and 42.36%, respectively. Deterministic auto-approval precision was 84.62% at 6.52% coverage, and a separate audit passed all 340 exclusions.

The paced evaluation used a 4.2-second request interval and a budget of 200 accounts. Gemini returned 185 usable decisions (92.50%) with zero prior cache hits and one same-run cache hit. Paired accuracy compares the local and Gemini choices on the same labeled accounts.

| Gemini model | Usable labeled rows | Local Top-1 on usable | Gemini Top-1 on usable | API requests |
|---|---:|---:|---:|---:|
| `gemini-3.6-flash` | 5 | 20.00% | 20.00% | 21 |
| `gemini-3.5-flash-lite` | 180 | 21.11% | 35.56% | 52 |
| All | 185 | 21.08% | 35.14% | 73 |

The API recorded rate limiting, HTTP 503 and 504 responses, and one out-of-pool or unknown-index response. Counts and settings are in the [final benchmark JSON](benchmark_summary.json).

## Earlier benchmark interpretation

Iteration 6 mapped 168 of 599 labeled accounts correctly at rank one, four fewer than iteration 5. Top-3 recall held at 259 of 599, and deterministic approvals rose from 28 correct out of 32 to 29 out of 33. Both runs handled all 340 exclusions correctly. Gemini decision coverage and cache use differed between runs, so the Top-1 and timing differences cannot be attributed to the dictionary alone. The backend's separate 1,004-account test used local retrieval and verified integration and persistence.
