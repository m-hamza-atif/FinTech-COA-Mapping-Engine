"""Evaluate lexical, dense, and hybrid retrieval on the golden test set."""

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import chromadb
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# Support both the direct script and module entry points.
ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from app.retrieval_text import code_key
from app.schemas.contracts import Account, MappingResult
from app.services.indexer import (
    ACTIVE_TARGET_THRESHOLD,
    MODEL_NAME,
    VOTE_THRESHOLD,
    Candidate,
    SearchIndex,
    collection_for,
    read_dataset,
    tokenize,
)
from app.services.llm_reranker import (
    BATCH_SIZE as GEMINI_BATCH_SIZE,
)
from app.services.llm_reranker import (
    DEFAULT_MODEL,
    SYSTEM_INSTRUCTION,
    Decision,
    GeminiReranker,
)
from app.services.normalizer import Normalizer
from app.services.retriever import (
    AUTO_APPROVE,
    HAPPY_PATH_REASON,
    MIN_APPROVAL_HISTORY,
    MIN_DENSE,
    MIN_MARGIN,
    MappingEngine,
    excluded,
    excluded_result,
    mapped_result,
)

BATCH_SIZE = 25
SEED = 42
CONFIGURATIONS = {
    "baseline_a": "Baseline A — Lexical only",
    "baseline_b": "Baseline B — Standard RAG",
    "system_c": "System C — Hierarchy-aware hybrid RAG",
}


@dataclass(frozen=True)
class Case:
    account: Account
    target_code: str | None
    expected_status: str | None


def read_cases(data_dir):
    rows, raw = read_dataset(data_dir / "golden_test_set.jsonl")
    cases = [
        Case(
            Account.model_validate({key: row[key] for key in Account.model_fields}),
            row["expected_target_code"],
            row["expected_routing_status"],
        )
        for row in rows
    ]
    if not cases:
        raise ValueError("The golden test set is empty.")
    random.Random(SEED).shuffle(cases)
    return cases, hashlib.sha256(raw).hexdigest()


class BudgetedReranker(GeminiReranker):
    """Use the production reranker with a reproducible account budget."""

    def __init__(self, api_key, model, account_limit, cache_path, dataset_hash):
        super().__init__(
            api_key if account_limit else None, model, batch_size=GEMINI_BATCH_SIZE
        )
        self.remaining = account_limit
        self.cache_path = cache_path
        self.dataset_hash = dataset_hash
        self.cache = (
            json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path.is_file()
            else {}
        )
        self.initial_cache_keys = frozenset(self.cache)
        self.prior_cache_hits = 0
        self.live_accounts = 0
        self.cached_accounts = 0
        self.successful_accounts = 0
        self.attempted_accounts = 0
        self.cache_changed = False

    @property
    def enabled(self):
        return self.remaining > 0 and (super().enabled or bool(self.cache))

    async def rerank(self, inputs, timeout):
        subset = inputs[: self.remaining]
        self.attempted_accounts += len(subset)
        results = {}
        misses = []
        for index, item in enumerate(subset):
            fingerprint = json.dumps(
                [
                    self.model,
                    SYSTEM_INSTRUCTION,
                    self.dataset_hash,
                    item.source_name,
                    item.normalized_name,
                    item.normal_balance,
                ]
            ).encode()
            key = hashlib.sha256(fingerprint).hexdigest()
            cached = self.cache.get(key)
            if cached is None or not any(
                candidate.target.code == cached["chosen_code"]
                for candidate in item.candidates
            ):
                misses.append((index, key, item))
            else:
                results[index] = Decision.model_validate({**cached, "index": index})
                self.cached_accounts += 1
                self.prior_cache_hits += key in self.initial_cache_keys

        if misses and super().enabled:
            self.live_accounts += len(misses)
            fresh = await super().rerank([item for _, _, item in misses], timeout)
            for position, decision in fresh.items():
                index, key, _ = misses[position]
                results[index] = decision.model_copy(update={"index": index})
                self.cache[key] = decision.model_dump(exclude={"index"})
                self.cache_changed = True
        self.successful_accounts += len(results)
        self.remaining -= len(subset)
        return results

    def save(self):
        if self.cache_changed:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.cache, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(self.cache_path)


def rank_baseline(accounts, targets, scores_by_name, label):
    scored = iter(
        scores_by_name(
            [account.source_name for account in accounts if not excluded(account)]
        )
    )
    results = []
    for account in accounts:
        if excluded(account):
            results.append(excluded_result(account))
            continue
        scores = next(scored)
        ranked = np.argsort(-scores, kind="stable")[:5]
        candidates = [Candidate(targets[i], float(scores[i])) for i in ranked]
        best = candidates[0]
        status = "AUTO_APPROVED" if best.score >= AUTO_APPROVE else "NEEDS_REVIEW"
        results.append(
            mapped_result(account, candidates, best, best.score, status, label)
        )
    return results


def lexical_scores(model, names):
    for name in names:
        scores = np.maximum(model.get_scores(tokenize(name)), 0)
        yield scores / (scores + 5.0)


def dense_scores(index, vectors, names):
    if names:
        yield from np.clip(index.embed(names) @ vectors.T, 0.0, 1.0)


def rate(correct, total):
    return correct / total if total else None


def summarize(cases, results: list[MappingResult], seconds, *, hybrid=False):
    counts = dict.fromkeys(
        [
            "labeled",
            "unlabeled",
            "excluded",
            "top_1",
            "top_3",
            "exclusions_correct",
            "approved",
            "labeled_approved",
            "approved_correct",
            "happy_path",
            "labeled_happy_path",
            "candidate_pool_hits",
            "happy_path_correct",
        ],
        0,
    )
    for case, result in zip(cases, results, strict=True):
        if (result.source_code, result.source_name) != (
            case.account.source_code,
            case.account.source_name,
        ):
            raise ValueError(
                "Benchmark results do not match input ordering or identity."
            )
        if case.expected_status == "EXCLUDED":
            counts["excluded"] += 1
            counts["exclusions_correct"] += (
                result.routing_status == "EXCLUDED"
                and result.confidence_score == 0.0
                and not result.top_k_candidates
                and result.predicted_target_code is None
                and result.predicted_target_name is None
                and result.target_hierarchy is None
            )
            continue
        approved = result.routing_status == "AUTO_APPROVED"
        counts["approved"] += approved
        took_happy_path = hybrid and result.reasoning == HAPPY_PATH_REASON
        counts["happy_path"] += took_happy_path
        if case.target_code is None:
            counts["unlabeled"] += 1
            continue
        expected = code_key(case.target_code)
        correct = code_key(result.predicted_target_code) == expected
        ranked = list(
            dict.fromkeys(
                code_key(code)
                for code in [
                    result.predicted_target_code,
                    *(c.code for c in result.top_k_candidates),
                ]
            )
        )
        counts["labeled"] += 1
        counts["top_1"] += correct
        counts["top_3"] += expected in [code for code in ranked if code is not None][:3]
        counts["candidate_pool_hits"] += expected in {
            code_key(c.code) for c in result.top_k_candidates
        }
        counts["labeled_approved"] += approved
        counts["approved_correct"] += approved and correct
        counts["labeled_happy_path"] += took_happy_path
        counts["happy_path_correct"] += took_happy_path and correct
    return {
        "top_1_accuracy": rate(counts["top_1"], counts["labeled"]),
        "top_3_recall": rate(counts["top_3"], counts["labeled"]),
        "candidate_pool_recall": rate(counts["candidate_pool_hits"], counts["labeled"]),
        "exception_handling_accuracy": rate(
            counts["exclusions_correct"], counts["excluded"]
        ),
        "auto_approve_precision": rate(
            counts["approved_correct"], counts["labeled_approved"]
        ),
        "llm_call_reduction": rate(
            counts["happy_path"], counts["labeled"] + counts["unlabeled"]
        )
        if hybrid
        else None,
        "happy_path_precision": rate(
            counts["happy_path_correct"], counts["labeled_happy_path"]
        ),
        "average_latency_seconds_per_100": seconds * 100 / len(cases),
        "counts": counts,
    }


async def measure(cases, predict, *, hybrid=False):
    outputs = []
    elapsed = 0.0
    for start in range(0, len(cases), BATCH_SIZE):
        batch = [case.account for case in cases[start : start + BATCH_SIZE]]
        started = perf_counter()
        results = await predict(batch) if hybrid else predict(batch)
        elapsed += perf_counter() - started
        outputs.extend(results)
        if hybrid and (len(outputs) % 100 == 0 or len(outputs) == len(cases)):
            print(
                f"System C evaluated {len(outputs):,}/{len(cases):,} accounts.",
                flush=True,
            )
    return summarize(cases, outputs, elapsed, hybrid=hybrid)


async def benchmark(args):
    cases, golden_hash = read_cases(args.data_dir)
    # Spend the live budget on accounts that contribute to accuracy metrics.
    cases.sort(key=lambda case: not bool(case.target_code))
    normalizer = Normalizer(args.data_dir)
    index = SearchIndex(args.data_dir, args.cache_dir, normalizer)
    target_rows, raw = read_dataset(args.data_dir / "target_coa.jsonl")
    names = [row["target_name"] for row in target_rows]
    lexical = BM25Okapi([tokenize(name) for name in names])
    client = chromadb.PersistentClient(path=str(args.cache_dir))
    collection = collection_for(
        client, "benchmark_names", raw, index.model, names, normalizer
    )
    documents = collection.get(include=["embeddings"])
    order = np.argsort(np.asarray(documents["ids"], dtype=int))
    vectors = np.asarray(documents["embeddings"], dtype=np.float32)[order]

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    snapshot = hashlib.sha256()
    for path in sorted(args.data_dir.iterdir()):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            snapshot.update(path.name.encode())
            snapshot.update(hashlib.sha256(path.read_bytes()).digest())
    reranker = BudgetedReranker(
        os.getenv("GEMINI_API_KEY"),
        model,
        args.llm_limit,
        args.cache_dir / "benchmark_gemini_v3.json",
        snapshot.hexdigest(),
    )
    engine = MappingEngine(normalizer, index, reranker)
    try:
        results = {
            "baseline_a": await measure(
                cases,
                lambda accounts: rank_baseline(
                    accounts,
                    index.targets,
                    lambda texts: lexical_scores(lexical, texts),
                    "Raw-name BM25.",
                ),
            ),
            "baseline_b": await measure(
                cases,
                lambda accounts: rank_baseline(
                    accounts,
                    index.targets,
                    lambda texts: dense_scores(index, vectors, texts),
                    "Raw-name cosine similarity.",
                ),
            ),
            "system_c": await measure(cases, engine.map_accounts, hybrid=True),
        }
        reranker.save()
    finally:
        await reranker.close()
    return {
        "dataset": {
            "golden_sha256": golden_hash,
            "accounts": len(cases),
            "target_accounts": len(names),
        },
        "settings": {
            "embedding_model": MODEL_NAME,
            "auto_approve_threshold": AUTO_APPROVE,
            "happy_path_dense_threshold": MIN_DENSE,
            "happy_path_historical_threshold": MIN_APPROVAL_HISTORY,
            "happy_path_margin": MIN_MARGIN,
            "happy_path_unanimous_vote_threshold": VOTE_THRESHOLD,
            "batch_size": BATCH_SIZE,
            "gemini_batch_size": GEMINI_BATCH_SIZE,
            "active_targets": len(index.active_positions),
            "active_target_threshold": ACTIVE_TARGET_THRESHOLD,
            "shuffle_seed": SEED,
        },
        "llm": {
            "model": model,
            "account_limit": args.llm_limit,
            "selected_accounts": reranker.attempted_accounts,
            "live_accounts": reranker.live_accounts,
            "cached_accounts": reranker.cached_accounts,
            "prior_cache_hits": reranker.prior_cache_hits,
            "same_run_cache_hits": reranker.cached_accounts - reranker.prior_cache_hits,
            "requests": sum(reranker.request_models.values()),
            "usable_decisions": reranker.successful_accounts,
            "requests_by_model": dict(reranker.request_models),
            "response_issues": dict(reranker.response_issues),
        },
        "results": results,
    }


def percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def write_reports(output_dir, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = summary["results"]
    counts = results["system_c"]["counts"]
    llm = summary["llm"]
    lines = [
        "# Mapping Engine Benchmark",
        "",
        (
            f"Evaluated {summary['dataset']['accounts']:,} golden accounts across "
            f"{summary['dataset']['target_accounts']:,} postable targets. "
            f"The set contains {counts['labeled']:,} labeled mappings, {counts['unlabeled']:,} "
            f"named accounts without a target label, and {counts['excluded']:,} exclusions."
        ),
        "",
        "| Configuration | Top-1 accuracy | Top-3 recall | Exclusion accuracy | Auto-approve precision | Labeled approvals | Happy Path LLM call reduction | Seconds per 100 accounts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name in CONFIGURATIONS.items():
        result = results[key]
        columns = [
            percent(result[field])
            for field in (
                "top_1_accuracy",
                "top_3_recall",
                "exception_handling_accuracy",
                "auto_approve_precision",
            )
        ]
        columns.extend(
            [
                str(result["counts"]["labeled_approved"]),
                percent(result["llm_call_reduction"]),
                f"{result['average_latency_seconds_per_100']:.2f}",
            ]
        )
        lines.append("| " + " | ".join([name, *columns]) + " |")
    lines += [
        "",
        "## Measurement notes",
        "",
        (
            "- Mapping accuracy and recall use labeled, non-excluded rows. Top-3 places the final "
            "prediction first, followed by the highest-ranked distinct retrieval candidates."
        ),
        (
            "- Auto-approve precision uses labeled approvals. The baselines apply the 0.85 threshold "
            "to saturated BM25 or clipped cosine. System C reserves auto-approval for deterministic "
            "Happy Path matches. Gemini reranks stay in review with confidence capped at 0.84."
        ),
        (
            f"- System C searches {summary['settings']['active_targets']:,} active targets first "
            f"and widens to the full COA when the best score is below {summary['settings']['active_target_threshold']:.2f}. "
            f"Its eight-candidate pool recalled {percent(results['system_c']['candidate_pool_recall'])} of labeled targets."
        ),
        (
            f"- The Happy Path gate alone mapped {counts['happy_path_correct']} of "
            f"{counts['labeled_happy_path']} labeled approvals correctly "
            f"({percent(results['system_c']['happy_path_precision'])})."
        ),
        (
            "- All configurations use the shared exclusion rule. Happy Path call reduction is the share "
            "of non-excluded accounts resolved by the hybrid retrieval gate. The baselines have no LLM stage."
        ),
        (
            f"- System C used {llm['usable_decisions']} usable Gemini decisions under a cap of "
            f"{llm['account_limit']} accounts. Live API requests totaled {llm['requests']} and "
            f"cache hits totaled {llm['cached_accounts']} "
            f"({llm.get('prior_cache_hits', llm['cached_accounts'])} from earlier runs and "
            f"{llm.get('same_run_cache_hits', 0)} from this run). "
            "Remaining ambiguous accounts used the hybrid fallback."
        ),
        (
            f"- Latency measures in-process mapping across batches of up to {summary['settings']['batch_size']} with warm indexes "
            "and includes live Gemini time when applicable. Cached decisions avoid network time, "
            "so compare runs with different cache use accordingly."
        ),
        "",
    ]
    if llm.get("requests_by_model"):
        requests = ", ".join(
            f"{model} ({count})"
            for model, count in sorted(llm["requests_by_model"].items())
        )
        lines.insert(-1, f"- Gemini requests by model were {requests}.")
    issues = {
        name: count for name, count in llm.get("response_issues", {}).items() if count
    }
    if issues:
        details = ", ".join(
            f"{name.replace('_', ' ')} ({count})"
            for name, count in sorted(issues.items())
        )
        lines.insert(
            -1,
            f"- Gemini attempt diagnostics were {details}. Counts can overlap across retries.",
        )
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".chroma")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=12,
        help="Maximum accounts eligible for Gemini reranking",
    )
    args = parser.parse_args()
    if args.llm_limit < 0:
        parser.error("--llm-limit must be nonnegative")
    load_dotenv(ROOT / ".env")
    summary = asyncio.run(benchmark(args))
    write_reports(args.output_dir, summary)
    print(f"Wrote benchmark results to {args.output_dir}")


if __name__ == "__main__":
    main()
