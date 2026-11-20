# Mapping Engine Benchmark

Evaluated 1,120 golden accounts across 5,428 postable targets. The set contains 599 labeled mappings, 181 named accounts without a target label, and 340 exclusions.

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
