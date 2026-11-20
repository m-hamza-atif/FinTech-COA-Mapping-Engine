"""Compare Happy Path thresholds on a name-disjoint slice of the training KB."""

import random
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np

from app.retrieval_text import code_key
from app.schemas.contracts import Account
from app.services.indexer import VOTE_THRESHOLD, SearchIndex, tokenize
from app.services.normalizer import Normalizer
from app.services.retriever import (
    AUTO_APPROVE,
    MIN_APPROVAL_HISTORY,
    MIN_DENSE,
    MIN_MARGIN,
)

ROOT = Path(__file__).resolve().parents[1]


def prepare_dev(index, normalizer):
    groups = defaultdict(set)
    for row in index.example_rows:
        groups[tuple(tokenize(normalizer.expand(row["source_name"])))].add(
            code_key(row["target_code"])
        )

    clear = []
    other = []
    for row in index.example_rows:
        account = Account(
            source_code=row["source_code"],
            source_name=row["source_name"],
            normal_balance=None,
            ending_balance=None,
            exclude=False,
        )
        query = normalizer.account_query(account)
        tokens = tuple(tokenize(query.text))
        if query.has_red_flag or len(groups[tokens]) != 1:
            continue
        group = clear if len(index.exact_names.get(tokens, [])) == 1 else other
        group.append((tokens, query, code_key(row["target_code"])))

    rng = random.Random(42)
    clear = list({item[0]: item for item in clear}.values())
    other = list({item[0]: item for item in other}.values())
    return rng.sample(clear, min(80, len(clear))), rng.sample(
        other, min(180, len(other))
    )


def remove_dev_names(index, normalizer, names):
    old = index.examples.get(include=["embeddings"])
    keep = [
        i
        for i, identifier in enumerate(old["ids"])
        if tuple(
            tokenize(
                normalizer.expand(index.example_rows[int(identifier)]["source_name"])
            )
        )
        not in names
    ]
    retained = [index.example_rows[int(old["ids"][i])] for i in keep]
    collection = chromadb.EphemeralClient().create_collection(
        name="calibration_examples",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.add(
        ids=[str(i) for i in range(len(retained))], embeddings=old["embeddings"][keep]
    )
    index.examples = collection
    index.configure_history(retained, normalizer)


def measure(
    ranked,
    cases,
    clear_count,
    dense_threshold,
    margin,
    history_threshold,
    vote_threshold,
):
    approved = correct = clear_approved = 0
    for i, (candidates, (_, _, expected)) in enumerate(zip(ranked, cases, strict=True)):
        best = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        evidence = best.historical_similarity >= history_threshold and (
            best.exact_name or best.dense_similarity >= dense_threshold
        )
        if (best.unanimous_vote and best.vote_similarity >= vote_threshold) or (
            best.score >= AUTO_APPROVE
            and best.score - second_score >= margin
            and evidence
        ):
            approved += 1
            correct += code_key(best.target.code) == expected
            clear_approved += i < clear_count
    precision = correct / approved if approved else 0
    coverage = clear_approved / clear_count if clear_count else 0
    return approved, correct, precision, clear_approved, coverage


def main():
    data = ROOT / "data" / "processed"
    normalizer = Normalizer(data)
    index = SearchIndex(data, ROOT / ".chroma", normalizer)
    clear, other = prepare_dev(index, normalizer)
    cases = clear + other
    remove_dev_names(index, normalizer, {item[0] for item in cases})
    queries = [item[1] for item in cases]
    ranked = list(index.search_batch(queries, index.embed([q.text for q in queries])))
    values = np.quantile([candidates[0].score for candidates in ranked], [0, 0.5, 1])
    print(
        f"Dev rows {len(cases)}, clear rows {len(clear)}, top hybrid scores min/median/max {values}"
    )
    print(
        "History | Dense | Margin | Vote | Approved | Correct | Precision | Clear approved | Clear coverage"
    )
    trials = []
    for history in (threshold / 1000 for threshold in range(850, 946, 5)):
        for dense in (0.82, 0.85, 0.88, 0.90):
            for margin in (0.04, 0.06, 0.08, 0.10, 0.12):
                for vote in (VOTE_THRESHOLD, 0.88, 0.90, 0.92, 0.94, 1.01):
                    metrics = measure(
                        ranked, cases, len(clear), dense, margin, history, vote
                    )
                    trials.append((history, dense, margin, vote, *metrics))
    for (
        history,
        dense,
        margin,
        vote,
        approved,
        correct,
        precision,
        clear_approved,
        coverage,
    ) in sorted(
        trials,
        key=lambda t: (t[6] >= 0.85 and t[8] >= 0.15, t[6], t[8], t[4]),
        reverse=True,
    )[:12]:
        print(
            f"{history:.3f} | {dense:.2f} | {margin:.2f} | {vote:.2f} | {approved} | {correct} | "
            f"{precision:.2%} | {clear_approved} | {coverage:.2%}"
        )
    print(
        "Active gate approved, correct, precision, clear approved, clear coverage:",
        measure(
            ranked,
            cases,
            len(clear),
            MIN_DENSE,
            MIN_MARGIN,
            MIN_APPROVAL_HISTORY,
            VOTE_THRESHOLD,
        ),
    )


if __name__ == "__main__":
    main()
