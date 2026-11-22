"""Evaluate lexical, dense, and hybrid retrieval on the golden test set."""

import argparse
import asyncio
import hashlib
import json
import math
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
    CANDIDATE_LIMIT,
    GEMINI_SHORTLIST_LIMIT,
    PREVIOUS_MODEL,
    RETRIEVAL_LIMIT,
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
    FALLBACK_MODEL,
    SYSTEM_INSTRUCTION,
    Decision,
    GeminiReranker,
    rerank_record,
)
from app.services.normalizer import Normalizer
from app.services.retriever import (
    AUTO_APPROVE,
    HAPPY_PATH_REASON,
    MIN_APPROVAL_HISTORY,
    MIN_DENSE,
    MIN_MARGIN,
    RERANKER_MODEL,
    LocalReranker,
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

    def __init__(
        self, api_key, model, account_limit, cache_path, dataset_hash, pace_seconds
    ):
        super().__init__(
            api_key if account_limit else None,
            model,
            batch_size=GEMINI_BATCH_SIZE,
            pace_seconds=pace_seconds,
        )
        self.remaining = account_limit
        self.cache_path = cache_path
        self.dataset_hash = dataset_hash
        self.cache = (
            json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path is not None and cache_path.is_file()
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
                    rerank_record(0, item),
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
        if self.cache_changed and self.cache_path is not None:
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
    codes = np.asarray([target.code for target in targets])
    for account in accounts:
        if excluded(account):
            results.append(excluded_result(account))
            continue
        scores = next(scored)
        ranked = np.lexsort((codes, -np.round(scores, 6)))[:5]
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


def split_cases(cases):
    labeled = [
        case
        for case in cases
        if case.target_code and case.expected_status != "EXCLUDED"
    ]
    return {
        "eval_399": labeled[200:],
        "dev_200": labeled[:200],
        "other": [
            case
            for case in cases
            if not case.target_code or case.expected_status == "EXCLUDED"
        ],
    }


async def measure_system(engine, cases, *, overall, selected_split=None):
    partitions = split_cases(cases)
    if selected_split:
        partitions = {
            selected_split: partitions[selected_split],
            "exclusion_audit": [
                case for case in cases if case.expected_status == "EXCLUDED"
            ],
        }
    summaries = {}
    all_cases, all_results = [], []
    all_seconds = 0.0
    all_hits = 0
    for name, rows in partitions.items():
        if not rows or (name == "other" and not overall):
            continue
        print(f"Evaluating {name} ({len(rows)} accounts).", flush=True)
        outputs = []
        elapsed = 0.0
        hits = 0
        shortlist_hits = local_top1 = local_top3 = 0
        paired = {}
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]

            def observe_pool(position, pool, batch=batch):
                nonlocal hits
                expected = code_key(batch[position].target_code)
                if expected and batch[position].expected_status != "EXCLUDED":
                    hits += any(
                        code_key(candidate.target.code) == expected
                        for candidate in pool
                    )

            def observe_decision(position, local, decision, batch=batch, paired=paired):
                expected = code_key(batch[position].target_code)
                if not expected:
                    return
                counts = paired.setdefault(
                    decision.model or "unknown",
                    {"usable": 0, "local_correct": 0, "llm_correct": 0},
                )
                counts["usable"] += 1
                counts["local_correct"] += code_key(local.target.code) == expected
                counts["llm_correct"] += code_key(decision.chosen_code) == expected

            def observe_shortlist(position, candidates, batch=batch):
                nonlocal shortlist_hits, local_top1, local_top3
                expected = code_key(batch[position].target_code)
                if expected and batch[position].expected_status != "EXCLUDED":
                    codes = [
                        code_key(candidate.target.code) for candidate in candidates
                    ]
                    shortlist_hits += expected in codes
                    local_top1 += expected == codes[0]
                    local_top3 += expected in codes[:3]

            started = perf_counter()
            outputs.extend(
                await engine.map_accounts(
                    [case.account for case in batch],
                    observe_pool=observe_pool,
                    observe_decision=observe_decision,
                    observe_shortlist=observe_shortlist,
                )
            )
            elapsed += perf_counter() - started
            if len(outputs) % 100 == 0:
                print(f"{name}: evaluated {len(outputs)}/{len(rows)}.", flush=True)
        summary = summarize(rows, outputs, elapsed, hybrid=True)
        summary["recall_at_25"] = rate(hits, summary["counts"]["labeled"])
        summary["local_retrieval"] = {
            "shortlist_k": engine.shortlist_limit,
            "shortlist_hits": shortlist_hits,
            "shortlist_recall": rate(shortlist_hits, summary["counts"]["labeled"]),
            "local_top1": rate(local_top1, summary["counts"]["labeled"]),
            "local_top3": rate(local_top3, summary["counts"]["labeled"]),
        }
        summary["paired_by_model"] = {
            model: {
                **counts,
                "local_top1_on_usable": rate(counts["local_correct"], counts["usable"]),
                "llm_top1_on_usable": rate(counts["llm_correct"], counts["usable"]),
            }
            for model, counts in paired.items()
        }
        summaries[name] = summary
        if overall:
            all_cases.extend(rows)
            all_results.extend(outputs)
            all_seconds += elapsed
            all_hits += hits
    if overall:
        summaries["overall"] = summarize(
            all_cases, all_results, all_seconds, hybrid=True
        )
        summaries["overall"]["recall_at_25"] = rate(
            all_hits, summaries["overall"]["counts"]["labeled"]
        )
    return summaries


async def offline_experiment(args):
    cases, golden_hash = read_cases(args.data_dir)
    normalizer = Normalizer(args.data_dir)
    base_model = "BAAI/bge-base-en-v1.5"
    reranker_model = args.reranker_model
    configurations = [
        ("Previous retrieval", PREVIOUS_MODEL, None),
        ("Experiment 1A", base_model, None),
    ]
    if reranker_model:
        configurations.append(("Experiment 1B", base_model, reranker_model))
    summary = {
        "golden_sha256": golden_hash,
        "shuffle_seed": SEED,
        "gemini_requests": 0,
        "batch_size": BATCH_SIZE,
        "active_target_threshold": ACTIVE_TARGET_THRESHOLD,
        "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "results": {},
    }
    index = None
    for name, model, reranker_name in configurations:
        print(f"Starting {name}.", flush=True)
        if index is None or index.model_name != model:
            index = SearchIndex(
                args.data_dir,
                args.cache_dir,
                normalizer,
                model_name=model,
                candidate_limit=RETRIEVAL_LIMIT,
            )
        local = LocalReranker(reranker_name) if reranker_name else None
        engine = MappingEngine(normalizer, index, GeminiReranker(None), local)
        engine.shortlist_limit = CANDIDATE_LIMIT
        summary["results"][name] = {
            "embedding_model": model,
            "reranker_model": reranker_name,
            "splits": await measure_system(engine, cases, overall=args.overall),
        }
        write_offline_report(args.output_dir, summary)


async def benchmark(args):
    cases, golden_hash = read_cases(args.data_dir)
    # Spend the live budget on accounts that contribute to accuracy metrics.
    cases.sort(key=lambda case: not bool(case.target_code))
    normalizer = Normalizer(args.data_dir)
    index = SearchIndex(
        args.data_dir,
        args.cache_dir,
        normalizer,
        candidate_limit=RETRIEVAL_LIMIT
        if args.reranker_model
        else GEMINI_SHORTLIST_LIMIT,
    )
    local_model = args.reranker_model
    local = LocalReranker(local_model) if local_model else None
    if args.split:
        return await split_benchmark(args, cases, golden_hash, normalizer, index, local)
    target_rows, raw = read_dataset(args.data_dir / "target_coa.jsonl")
    names = [row["target_name"] for row in target_rows]
    lexical = BM25Okapi([tokenize(name) for name in names])
    client = chromadb.PersistentClient(path=str(args.cache_dir))
    collection = collection_for(
        client,
        "benchmark_names",
        raw,
        index.model,
        names,
        normalizer,
        model_name=index.model_name,
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
        args.cache_dir / "benchmark_gemini_v4.json",
        snapshot.hexdigest(),
        args.pace_seconds,
    )
    engine = MappingEngine(normalizer, index, reranker, local)
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
        }
        splits = await measure_system(engine, cases, overall=True)
        results["system_c"] = splits.pop("overall")
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
            "embedding_model": index.model_name,
            "reranker_model": local_model if local else None,
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
            "pace_seconds": args.pace_seconds,
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
        "splits": splits,
    }


async def split_benchmark(args, cases, golden_hash, normalizer, index, local):
    key = os.getenv("GEMINI_API_KEY")
    if args.llm_limit and not key:
        raise ValueError("Set GEMINI_API_KEY to run a live split benchmark.")
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    reranker = BudgetedReranker(
        key, model, args.llm_limit, None, golden_hash, args.pace_seconds
    )
    engine = MappingEngine(normalizer, index, reranker, local)
    try:
        splits = await measure_system(
            engine, cases, overall=False, selected_split=args.split
        )
    finally:
        await reranker.close()
    metrics = splits[args.split]
    if index.candidate_limit < RETRIEVAL_LIMIT:
        metrics.pop("recall_at_25")
    for name in (model, FALLBACK_MODEL):
        metrics["paired_by_model"].setdefault(
            name,
            {
                "usable": 0,
                "local_correct": 0,
                "llm_correct": 0,
                "local_top1_on_usable": None,
                "llm_top1_on_usable": None,
            },
        )
    return {
        "split": args.split,
        "golden_sha256": golden_hash,
        "seed": SEED,
        "embedding_model": index.model_name,
        "reranker_model": args.reranker_model,
        "gemini": {
            "primary": model,
            "fallback": FALLBACK_MODEL,
            "pace_seconds": args.pace_seconds,
            "account_limit": args.llm_limit,
            "selected_accounts": reranker.attempted_accounts,
            "usable_decisions": reranker.successful_accounts,
            "usable_rate": rate(
                reranker.successful_accounts, reranker.attempted_accounts
            ),
            "prior_cache_hits": reranker.prior_cache_hits,
            "same_run_cache_hits": reranker.cached_accounts,
            "requests_by_model": dict(reranker.request_models),
            "response_issues": dict(reranker.response_issues),
        },
        "results": metrics,
        "exclusion_audit": splits.get("exclusion_audit"),
    }


def percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def write_split_report(output_dir, summary):
    result = summary["results"]
    local = result["local_retrieval"]
    gemini = summary["gemini"]
    audit = summary["exclusion_audit"]
    rows = [
        ("Local Top-1", local["local_top1"]),
        ("Local Top-3", local["local_top3"]),
        (f"Local Recall@{local['shortlist_k']}", local["shortlist_recall"]),
        ("Happy Path precision", result["happy_path_precision"]),
        ("Happy Path coverage", result["llm_call_reduction"]),
        ("End-to-end Top-1", result["top_1_accuracy"]),
        ("End-to-end Top-3", result["top_3_recall"]),
        ("End-to-end auto-approve precision", result["auto_approve_precision"]),
        (
            "Exclusion accuracy (separate audit)",
            audit["exception_handling_accuracy"] if audit else None,
        ),
    ]
    lines = [
        "# Final paced Gemini evaluation",
        "",
        f"Split `{summary['split']}`, seed {summary['seed']}, {result['counts']['labeled']} labeled accounts.",
        f"Embeddings `{summary['embedding_model']}`, cross-encoder `{summary['reranker_model'] or 'off'}`.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        *(f"| {label} | {percent(value)} |" for label, value in rows),
        "",
        (
            f"Gemini pace {gemini['pace_seconds']:.1f} s. "
            f"Usable decisions {gemini['usable_decisions']}/{gemini['selected_accounts']} "
            f"({percent(gemini['usable_rate'])}), with {gemini['prior_cache_hits']} prior cache hits."
        ),
        "",
        "| Model | Usable labeled rows | Local Top-1 on usable | Gemini Top-1 on usable | API requests |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, pair in result["paired_by_model"].items():
        lines.append(
            f"| {model} | {pair['usable']} | {percent(pair['local_top1_on_usable'])} | "
            f"{percent(pair['llm_top1_on_usable'])} | {gemini['requests_by_model'].get(model, 0)} |"
        )
    pairs = result["paired_by_model"].values()
    usable = sum(pair["usable"] for pair in pairs)
    local_correct = sum(pair["local_correct"] for pair in pairs)
    llm_correct = sum(pair["llm_correct"] for pair in pairs)
    lines.append(
        f"| All | {usable} | {percent(rate(local_correct, usable))} | "
        f"{percent(rate(llm_correct, usable))} | {sum(gemini['requests_by_model'].values())} |"
    )
    diagnostics = ", ".join(
        f"{key.replace('_', ' ')} ({count})"
        for key, count in sorted(gemini["response_issues"].items())
        if count
    )
    if diagnostics:
        lines.extend(
            ["", f"API diagnostics can overlap across attempts. {diagnostics}."]
        )
    lines.extend(
        [
            "",
            (
                "The usable rate uses selected accounts as the denominator. "
                "Paired accuracy compares predictions on the same labeled rows. "
                "Exclusions are audited separately from held-out mapping accuracy."
            ),
        ]
    )
    write_run_files(output_dir, summary, lines)


def write_reports(output_dir, summary):
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
            f"Its eight displayed candidates recalled {percent(results['system_c']['candidate_pool_recall'])} of labeled targets."
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
    lines.extend(
        [
            "## Paired Gemini accuracy on usable decisions",
            "",
            "| Split | Model | Usable labeled decisions | Local Top-1 | Gemini Top-1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for split, metrics in summary["splits"].items():
        for model, pair in metrics["paired_by_model"].items():
            lines.append(
                f"| {split} | {model} | {pair['usable']} | "
                f"{percent(pair['local_top1_on_usable'])} | {percent(pair['llm_top1_on_usable'])} |"
            )
    write_run_files(output_dir, summary, lines)


def write_offline_report(output_dir, summary):
    lines = [
        "# Experiment 1 - Offline retrieval comparison",
        "",
        f"Golden set SHA-256 `{summary['golden_sha256']}`. Split seed {summary['shuffle_seed']}. Gemini requests: 0.",
        "",
        "| Configuration | Split | Recall@25 | Recall@8 | Local Top-1 | Local Top-3 | Happy Path precision | Happy Path coverage | Seconds / 100 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, config in summary["results"].items():
        for split, metrics in config["splits"].items():
            if split == "other":
                continue
            columns = [
                percent(metrics[key])
                for key in (
                    "recall_at_25",
                    "candidate_pool_recall",
                    "top_1_accuracy",
                    "top_3_recall",
                    "happy_path_precision",
                    "llm_call_reduction",
                )
            ]
            columns.append(f"{metrics['average_latency_seconds_per_100']:.2f}")
            lines.append("| " + " | ".join([name, split, *columns]) + " |")
    lines.extend(["", "## Model configuration", ""])
    for name, config in summary["results"].items():
        lines.append(
            f"- {name} uses `{config['embedding_model']}` with "
            f"{config['reranker_model'] or 'the hybrid order'} for final selection."
        )
    lines.extend(
        [
            "",
            (
                "Recall@25 uses the hybrid pool before reranking. Happy Path coverage uses non-excluded accounts. "
                "Latency includes embedding, retrieval, and local reranking, with loaded model weights and indexes."
            ),
            (
                "All configurations use the same deterministic score ordering and exact historical-neighbor search. "
                "For cross-encoder runs the deterministic approval gate retains its hybrid winner."
            ),
            "",
        ]
    )
    if "Experiment 1B" in summary["results"]:
        metrics = summary["results"]["Experiment 1B"]["splits"].get("eval_399")
        if metrics:
            passed = (
                metrics["candidate_pool_recall"] >= 0.618
                and metrics["average_latency_seconds_per_100"] < 10
            )
            lines.append(
                f"Held-out success criterion (Recall@8 ≥ 61.8% and < 10 seconds / 100): "
                f"{'met' if passed else 'missed'}."
            )
    write_run_files(output_dir, summary, lines)


def write_run_files(output_dir, summary, lines):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "benchmark_run_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".chroma")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "latest",
        help="Directory for the generated report and JSON, replaced on each run",
    )
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=12,
        help="Maximum accounts eligible for Gemini reranking",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run the three offline retrieval configurations",
    )
    parser.add_argument(
        "--overall",
        action="store_true",
        help="Include all golden accounts in the offline results",
    )
    parser.add_argument(
        "--no-cross-encoder",
        action="store_true",
        help="Disable local cross-encoder reranking",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=0.0,
        help="Minimum interval between Gemini request starts",
    )
    parser.add_argument(
        "--split",
        choices=("dev_200", "eval_399"),
        help="Evaluate only this labeled split and audit exclusions",
    )
    args = parser.parse_args()
    if args.llm_limit < 0:
        parser.error("--llm-limit must be nonnegative")
    if not math.isfinite(args.pace_seconds) or args.pace_seconds < 0:
        parser.error("--pace-seconds must be finite and nonnegative")
    load_dotenv(ROOT / ".env")
    if args.no_llm and args.split:
        args.llm_limit = 0
    local_model = os.getenv(
        "RERANKER_MODEL", RERANKER_MODEL if args.no_llm and not args.split else "off"
    ).strip()
    args.reranker_model = (
        None
        if args.no_cross_encoder
        or local_model.casefold() in {"", "off", "none", "disabled"}
        else local_model
    )
    if args.split:
        write_split_report(args.output_dir, asyncio.run(benchmark(args)))
    elif args.no_llm:
        asyncio.run(offline_experiment(args))
    else:
        write_reports(args.output_dir, asyncio.run(benchmark(args)))
    print(f"Wrote benchmark results to {args.output_dir}")


if __name__ == "__main__":
    main()
