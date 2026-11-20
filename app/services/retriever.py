import asyncio
from collections.abc import Sequence

from app.schemas.contracts import Account, MappingResult, TopCandidate
from app.services.indexer import Candidate, SearchIndex
from app.services.llm_reranker import REVIEW_CONFIDENCE_CAP, GeminiReranker, RerankInput
from app.services.normalizer import Normalizer

AUTO_APPROVE = 0.85
MIN_MARGIN = 0.08
MIN_DENSE = 0.88
MIN_APPROVAL_HISTORY = 0.895
REQUEST_BUDGET = 50.0
HAPPY_PATH_REASON = "Deterministic retrieval evidence met the approval criteria."


def happy_path(candidates, has_red_flag):
    best = candidates[0]
    if best.unanimous_vote:
        return not has_red_flag
    second = candidates[1].score if len(candidates) > 1 else 0.0
    return (
        not has_red_flag
        and best.score >= AUTO_APPROVE
        and best.score - second >= MIN_MARGIN
        and best.historical_similarity >= MIN_APPROVAL_HISTORY
        and (best.exact_name or (best.dense_similarity or 0.0) >= MIN_DENSE)
    )


def excluded(account: Account) -> bool:
    return account.exclude or not (account.source_name or "").strip()


def excluded_result(account: Account) -> MappingResult:
    return MappingResult(
        source_code=account.source_code,
        source_name=account.source_name,
        predicted_target_code=None,
        predicted_target_name=None,
        target_hierarchy=None,
        confidence_score=0.0,
        routing_status="EXCLUDED",
        reasoning="Source account is excluded or has no description.",
    )


def mapped_result(account, candidates, selected, confidence, status, reasoning):
    return MappingResult(
        source_code=account.source_code,
        source_name=account.source_name,
        predicted_target_code=selected.target.code,
        predicted_target_name=selected.target.name,
        target_hierarchy=selected.target.hierarchy or None,
        confidence_score=confidence,
        routing_status=status,
        top_k_candidates=[
            TopCandidate(code=c.target.code, name=c.target.name, score=c.score)
            for c in candidates
        ],
        reasoning=reasoning,
    )


class MappingEngine:
    def __init__(
        self, normalizer: Normalizer, index: SearchIndex, reranker: GeminiReranker
    ):
        self.normalizer = normalizer
        self.index = index
        self.reranker = reranker

    def _retrieve(self, accounts):
        active = [
            (position, account)
            for position, account in enumerate(accounts)
            if not excluded(account)
        ]
        queries = [self.normalizer.account_query(account) for _, account in active]
        vectors = self.index.embed([query.text for query in queries])
        return [
            (position, query, vector, candidates)
            for (position, _), query, vector, candidates in zip(
                active,
                queries,
                vectors,
                self.index.search_batch(queries, vectors),
                strict=True,
            )
        ]

    async def map_accounts(self, accounts: Sequence[Account]) -> list[MappingResult]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + REQUEST_BUDGET
        results = {
            position: excluded_result(account)
            for position, account in enumerate(accounts)
            if excluded(account)
        }
        if len(results) == len(accounts):
            return list(results.values())
        retrieved = await asyncio.to_thread(self._retrieve, accounts)
        pending = []
        for position, query, vector, candidates in retrieved:
            best = candidates[0]
            if happy_path(candidates, query.has_red_flag):
                results[position] = mapped_result(
                    accounts[position],
                    candidates,
                    best,
                    max(best.score, AUTO_APPROVE)
                    if best.unanimous_vote
                    else best.score,
                    "AUTO_APPROVED",
                    HAPPY_PATH_REASON,
                )
            else:
                pending.append((position, query, vector, candidates))

        if pending and self.reranker.enabled and loop.time() < deadline:
            examples = await asyncio.to_thread(
                self.index.similar_examples, [vector for _, _, vector, _ in pending]
            )
            rerank_inputs = [
                RerankInput(
                    accounts[position].source_name,
                    query.text,
                    query.normal_balance,
                    candidates,
                    matches,
                )
                for (position, query, _, candidates), matches in zip(
                    pending, examples, strict=True
                )
            ]
            decisions = await self.reranker.rerank(
                rerank_inputs, deadline - loop.time()
            )
        else:
            decisions = {}

        for offset, (position, query, _, candidates) in enumerate(pending):
            decision = decisions.get(offset)
            selected: Candidate = candidates[0]
            confidence = selected.score
            reason = (
                "Gemini reranking was unavailable. Showing the top hybrid candidate."
            )
            if decision is not None:
                selected = next(
                    candidate
                    for candidate in candidates
                    if candidate.target.code == decision.chosen_code
                )
                confidence = decision.review_confidence(selected.score)
                reason = decision.reasoning
            if query.has_red_flag:
                confidence = min(confidence, REVIEW_CONFIDENCE_CAP)
                reason = f"Accounting risk term requires review. {reason}"[:4096]
            results[position] = mapped_result(
                accounts[position],
                candidates,
                selected,
                confidence,
                "NEEDS_REVIEW",
                reason,
            )
        return [results[position] for position in range(len(accounts))]
