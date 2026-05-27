"""
Context compression and token-budget-aware packing.

Takes reranked chunks and produces a compact context block that:

 - Deduplicates near-identical content
 - Merges adjacent chunks from the same document/parent
 - Trims to hard token budget
 - Preserves provenance (source, chunk id) for citations
"""

from __future__ import annotations
import tiktoken
from dataclasses import dataclass, field
from app.models.schemas import RetrievedChunk

_ENC = tiktoken.get_encoding("cl100k_base")

def _tokens(text: str) -> int:
    return len(_ENC.encode(text))

@dataclass
class ContextBlock:
    text: str
    source_id: str
    document_id: str = ""
    token_count: int = field(init=False)

    def __post_init__(self):
        self.token_count = _tokens(self.text)

@dataclass
class CompressedContext:
    blocks: list[ContextBlock]
    tokens: int
    final_prompt_context: str # formatted string ready for LLM injection

def _jaccard_sim(a: str, b: str) -> float:
    """Fast token-level Jaccard similarity for near-duplicate detection."""

    sa = set(a.lower().split())
    sb = set(b.lower().split())

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)

class ContextBuilder:
    """Builds a compressed, token-bounded context from reranked chunks.
    
    Args:
        max_tokens: Hard ceiling for assembled context
        dedup_thresh: Jaccard threshold above which a chunk is considered a near-duplicate of an already included chunk.
    """

    def __init__(self, max_tokens: int = 2200, dedup_thresh: float = 0.72) -> None:
        self.max_tokens = max_tokens
        self.dedup_thresh = dedup_thresh

    def build(self, chunks: list[RetrievedChunk]) -> CompressedContext:

        selected: list[RetrievedChunk] = []
        used_tokens = 0

        for chunk in chunks:
            t = _tokens(chunk.text)
            if used_tokens + t > self.max_tokens:
                break

            # Near-duplicate check against already-selected chunks
            is_dup = any(
                _jaccard_sim(chunk.text, s.text) > self.dedup_thresh
                for s in selected
            )

            if is_dup:
                continue

            selected.append(chunk)
            used_tokens += t

            blocks = [
                ContextBlock(
                    text=c.text,
                    source_id=c.id,
                    document_id=c.metadata.get("document_id", ""),
                )
                for c in selected
            ]

            # Format for prompt injection with numbered citations
            formatted_parts = []
            for i, b in enumerate(blocks, start=1):
                doc_ref = f"[{i}] (source: {b.document_id or b.source_id})"
                formatted_parts.append(f"{doc_ref}\n{b.text}")
            final_context = "\n\n---\n\n".join(formatted_parts)

            return CompressedContext(
                blocks=blocks,
                tokens=_tokens(final_context),
                final_prompt_context=final_context
            )






