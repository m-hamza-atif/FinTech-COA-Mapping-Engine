import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from app.retrieval_text import code_key, hierarchy_text
from app.services.normalizer import Normalizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HISTORICAL_THRESHOLD = 0.85
VOTE_THRESHOLD = 0.85
CANDIDATE_LIMIT = 8
ACTIVE_TARGET_THRESHOLD = 0.55


@dataclass(frozen=True)
class Target:
    code: str
    name: str
    hierarchy: str
    normal_balance: str


@dataclass(frozen=True)
class Candidate:
    target: Target
    score: float
    dense_similarity: float | None = None
    historical_similarity: float = 0.0
    exact_consensus: bool = False
    exact_name: bool = False
    unanimous_vote: bool = False
    active_count: int = 0
    vote_similarity: float = 0.0


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold())


def read_dataset(path: Path):
    raw = path.read_bytes()
    return [json.loads(line) for line in raw.splitlines() if line.strip()], raw


def collection_for(client, name, raw, model, documents, normalizer, metadata=None):
    rules = json.dumps(normalizer.abbreviations, sort_keys=True).encode()
    digest = hashlib.sha256(MODEL_NAME.encode() + rules + raw).hexdigest()[:24]
    collection = client.get_or_create_collection(
        name=f"{name}_{digest}",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    ids = [str(index) for index in range(len(documents))]
    count = collection.count()
    if count == len(documents):
        return collection
    existing = set(collection.get(include=[])["ids"]) if count else set()
    missing = [
        (identifier, text)
        for identifier, text in zip(ids, documents, strict=True)
        if identifier not in existing
    ]
    for start in range(0, len(missing), 256):
        batch_ids, batch = map(list, zip(*missing[start : start + 256], strict=True))
        vectors = model.encode(
            batch, normalize_embeddings=True, show_progress_bar=False
        )
        collection.upsert(
            ids=batch_ids,
            documents=batch,
            embeddings=vectors,
            metadatas=[metadata[int(identifier)] for identifier in batch_ids]
            if metadata
            else None,
        )
    return collection


class SearchIndex:
    def __init__(
        self, data_dir: Path, cache_dir: Path, normalizer: Normalizer, model=None
    ):
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME, device="cpu")
        self.model = model
        self.model_lock = Lock()
        targets, target_raw = read_dataset(data_dir / "target_coa.jsonl")
        examples, _ = read_dataset(data_dir / "few_shot_kb.jsonl")
        example_raw = "".join(
            json.dumps(
                {k: v for k, v in row.items() if k != "training_pmc_count"},
                ensure_ascii=False,
            )
            + "\n"
            for row in examples
        ).encode()
        if not targets:
            raise ValueError("The target COA is empty.")
        self.targets = [
            Target(
                row["target_code"],
                row["target_name"],
                row["parent_path"],
                row["normal_balance"],
            )
            for row in targets
        ]
        self.example_rows = examples
        target_docs = [
            normalizer.expand(hierarchy_text(target.name, target.hierarchy))
            for target in self.targets
        ]
        example_docs = [normalizer.expand(row["source_name"]) for row in examples]
        self.target_positions = {
            code_key(target.code): i for i, target in enumerate(self.targets)
        }
        client = chromadb.PersistentClient(path=str(cache_dir))
        target_collection = collection_for(
            client,
            "target",
            target_raw + b"hierarchy-metadata-v1",
            model,
            target_docs,
            normalizer,
            metadata=[
                {
                    "target_code": t.code,
                    "target_name": t.name,
                    "parent_path": t.hierarchy,
                    "normal_balance": t.normal_balance,
                }
                for t in self.targets
            ],
        )
        self.examples = collection_for(
            client, "examples", example_raw, model, example_docs, normalizer
        )
        stored = target_collection.get(include=["embeddings"])
        order = np.argsort([int(identifier) for identifier in stored["ids"]])
        self.vectors = np.asarray(stored["embeddings"], dtype=np.float32)[order]
        self.configure_history(examples, normalizer)
        name_tokens = [tuple(tokenize(normalizer.expand(t.name))) for t in self.targets]
        self.exact_names = defaultdict(list)
        for position, tokens in enumerate(name_tokens):
            if tokens:
                self.exact_names[tokens].append(position)
        self.bm25 = BM25Okapi(name_tokens)
        self.balance_masks = {
            balance: np.array(
                [target.normal_balance == balance for target in self.targets]
            )
            for balance in ("Debit", "Credit")
        }

    def configure_history(self, examples, normalizer):
        self.example_rows = examples
        self.exact_history = defaultdict(set)
        self.active_target_counts = Counter()
        positions = set()
        for row in examples:
            position = self.target_positions.get(code_key(row["target_code"]))
            if position is not None:
                positions.add(position)
                tokens = tuple(tokenize(normalizer.expand(row["source_name"])))
                self.exact_history[tokens].add(position)
                self.active_target_counts[self.targets[position].code] += row.get(
                    "training_pmc_count", 1
                )
        counts = np.array([self.active_target_counts[t.code] for t in self.targets])
        log_frequency = np.log(np.maximum(counts, 1))
        self.active_priors = np.where(
            counts > 0,
            0.08 + 0.07 * log_frequency / max(float(log_frequency.max()), 1e-9),
            0.0,
        )
        self.active_positions = np.array(sorted(positions), dtype=int)
        self.active_vectors = self.vectors[self.active_positions]

    def embed(self, texts):
        if not texts:
            return []
        with self.model_lock:
            return self.model.encode(
                [QUERY_PREFIX + text for text in texts],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def search_batch(self, queries, vectors):
        for start in range(0, len(queries), 128):
            batch = queries[start : start + 128]
            batch_vectors = vectors[start : start + len(batch)]
            similarities = batch_vectors @ self.active_vectors.T
            evidence = self.historical_matches(batch_vectors)
            for query, vector, cosine, matches in zip(
                batch, batch_vectors, similarities, evidence, strict=True
            ):
                candidates = self.search(query, cosine, matches, self.active_positions)
                if (
                    max((candidate.score for candidate in candidates), default=0.0)
                    < ACTIVE_TARGET_THRESHOLD
                ):
                    candidates = self.search(query, self.vectors @ vector, matches)
                yield candidates

    def historical_matches(self, vectors):
        count = min(5, len(self.example_rows))
        if not count:
            return [[] for _ in vectors]
        result = self.examples.query(
            query_embeddings=np.asarray(vectors), n_results=count, include=["distances"]
        )
        matches = []
        for ids, distances in zip(result["ids"], result["distances"], strict=True):
            neighbors = []
            for identifier, distance in zip(ids, distances, strict=True):
                row = self.example_rows[int(identifier)]
                position = self.target_positions.get(code_key(row["target_code"]))
                similarity = min(1.0, 1.0 - distance)
                neighbors.append((position, similarity))
            matches.append(neighbors)
        return matches

    def search(self, query, cosine, neighbors, positions=None):
        if positions is not None and not len(positions):
            return []
        tokens = tuple(tokenize(query.text))
        cosine = np.clip(cosine, 0.0, 1.0)
        selected = slice(None) if positions is None else positions
        lexical = np.maximum(
            self.bm25.get_scores(tokens)
            if positions is None
            else self.bm25.get_batch_scores(tokens, positions),
            0,
        )
        local_scores = (
            0.80 * cosine
            + 0.20 * lexical / (lexical + 5.0)
            + self.active_priors[selected]
        )
        scores = np.full(len(self.targets), -np.inf)
        scores[selected] = local_scores
        dense = np.zeros(len(self.targets))
        dense[selected] = cosine
        mask = self.balance_masks.get(query.normal_balance)
        if mask is not None:
            weight = 0.035 if query.explicit_balance else 0.015
            scores += np.where(mask, weight, -weight)
        exact = self.exact_history.get(tokens, set())
        exact_consensus = exact if len(exact) == 1 else set()
        matches = {}
        votes = Counter()
        for i, similarity in neighbors:
            if i is not None and similarity >= HISTORICAL_THRESHOLD:
                matches[i] = max(similarity, matches.get(i, 0.0))
                votes[i] += 1
        for i, similarity in matches.items():
            scores[i] += 0.10 + 0.80 * (similarity - HISTORICAL_THRESHOLD)
            scores[i] += 0.02 * (votes[i] - 1)
        for i in exact_consensus:
            scores[i] += 0.20
        exact_names = self.exact_names.get(tokens, [])
        scores[exact_names] += 0.12
        # Sort before clipping so very strong candidates still have a stable lead.
        eligible = np.arange(len(self.targets)) if positions is None else positions
        ranked = list(eligible[np.argsort(-scores[eligible], kind="stable")])
        nearest = neighbors[:3]
        unanimous = None
        if (
            len(nearest) == 3
            and nearest[0][0] is not None
            and np.isfinite(scores[nearest[0][0]])
            and all(
                i == nearest[0][0] and similarity >= VOTE_THRESHOLD
                for i, similarity in nearest
            )
        ):
            unanimous = nearest[0][0]
            ranked.remove(unanimous)
            ranked.insert(0, unanimous)
        # Keep the best hybrid hits and reserve room for distinct historical targets.
        voted = sorted(
            (i for i in votes if np.isfinite(scores[i])),
            key=lambda i: (-votes[i], -matches[i], i),
        )
        ranked = list(dict.fromkeys([*ranked[:5], *voted, *ranked[:CANDIDATE_LIMIT]]))[
            :CANDIDATE_LIMIT
        ]
        return [
            Candidate(
                self.targets[i],
                float(np.clip(scores[i], 0.0, 1.0)),
                float(dense[i]),
                matches.get(i, 0.0),
                int(i) in exact_consensus,
                len(exact_names) == 1 and i == exact_names[0],
                i == unanimous,
                self.active_target_counts[self.targets[i].code],
                min(similarity for _, similarity in nearest) if i == unanimous else 0.0,
            )
            for i in ranked
        ]

    def similar_examples(self, vectors):
        count = min(3, len(self.example_rows))
        if not count:
            return [[] for _ in vectors]
        result = self.examples.query(
            query_embeddings=np.asarray(vectors), n_results=count, include=[]
        )
        return [
            [self.example_rows[int(identifier)] for identifier in ids]
            for ids in result["ids"]
        ]
