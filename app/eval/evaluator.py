"""
Offline evaluation harness.

Metrics:
   - Retrieval : Recall@k, MRR
   - Generation : Groundedness (LLM-as-judge, Optional)
   - Latency : total pipeline time per case
"""

from __future__ import annotations
import asyncio
import stat
import statistics
import time
from dataclasses import dataclass, field
from app.services.query_processor import QueryProcessor
from app.services.retriever import HybridRetriever
from app.services.reranker import get_reranker
from app.services.context_builder import ContextBuilder
from app.services.llm_router import LLMRouter
from app.config import get_settings
import structlog

log = structlog.get_logger()
_settings = get_settings()

@dataclass
class EvalCase:
    query: str
    tenant_id: str
    expected_doc_ids: list[str] # ground-truth relevant document ids
    reference_answer: str = ""

@dataclass
class EvalResult:
    query: str
    recall_at_k: float
    mrr: float
    groundedness: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)

class RAGEvaluator:
    def __init__(self) -> None:

        self._processor = QueryProcessor()
        self._retriever = HybridRetriever()
        self._reranker = get_reranker()
        self._ctx = ContextBuilder(max_tokens=_settings.max_context_tokens)
        self._llm = LLMRouter()

    async def evaluate_case(self, case: EvalCase, k: int = 5) -> EvalResult:

        t0 = time.perf_counter()

        rewrites = await self._processor.rewrite(case.query)

        all_hits = await asyncio.gather(*[
            self._retriever.retrieve(q, case.tenant_id)
            for q in rewrites
        ])

        seen, merged = set(), []
        for hits in all_hits:
            for h in hits:
                if h.id not in seen:
                    seen.add(h.id)
                    merged.append(h)

        reranked = await self._reranker.rerank(
            case.query, merged[:40], top_n=k
        )

        latency_ms = {
            "total_ms": round((time.perf_counter() - t0) * 1000, 1)
        }

        retrieved_ids = [
            c.metadata.get("document_id", c.id) for c in reranked
        ]

        relevant = set(case.expected_doc_ids)

        # Recall@k
        recall = len(relevant & set(retrieved_ids[:k])) / max(len(relevant), 1)

        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                mrr = 1.0 / rank
                break

        return EvalResult(
            query=case.query,
            recall_at_k=recall,
            mrr=mrr,
            latency_ms=latency_ms,
        )

    async def evaluate_suite(self, cases: list[EvalCase]) -> dict:
        """Run all cases concurrently and return aggregate metrics."""

        results = await asyncio.gather(*[
            self.evaluate_case(c) for c in cases
        ])

        recalls = [r.recall_at_k for r in results]
        mrrs = [r.mrr for r in results]

        log.info(
            "eval_suite_complete",
            n=len(results),
            recall=round(statistics.mean(recalls), 4),
            mrr=round(statistics.mean(mrrs), 4),
        )

        return {
            "n": len(results),
            "recall@5_mean": round(statistics.mean(recalls), 4),
            "mrr_mean": round(statistics.mean(mrrs), 4),
            "latency_p50_ms": round(
                statistics.median([
                    r.latency_ms.get("total_ms", 0) for r in results
                ]), 1
            ),
        }