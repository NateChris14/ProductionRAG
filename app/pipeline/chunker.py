"""
Semantic + dynamic chunker.

Strategy:
- Split text into sentences using regex-based splitter
- Compute consecutive-sentence embedding similarity
- Place a boundary wherever similarity drops below threshold.
- Merge tiny segments up to max_tokens; split oversized ones on paragraph.
"""

from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
import tiktoken
from app.config import get_settings

from sentence_transformers import SentenceTransformer

_settings = get_settings()
_ENC = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))

def _split_sentences(text: str) -> list[str]:
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]

@dataclass
class SemanticChunkResult:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    token_count: int = 0
    chunk_index: int = 0
    parent_id: Optional[str] = None

class SemanticChunker:
    """
    Produces semantically coherent chunks with adaptive sizing.

    Args:
        min_tokens: Floor - merge chunks below this.
        max_tokens: Ceiling - hard-split chunks above this.
        similarity_thresh: Cosine similarity threshold for boundary detection.
                           Lower = more splits.  Default 0.75 works well.
        overlap_sentences: How many trailing sentences to repeat in next chunk for discourse continuity.
    """

    def __init__(
        self,
        min_tokens: int = 150,
        max_tokens: int = 700,
        similarity_thresh: float = 0.75,
        overlap_sentences: int = 1
    ) -> None:
        
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.similarity_thresh = similarity_thresh
        self.overlap_sentences = overlap_sentences
        self._embed_fn = None

    def _get_embed_fn(self):
        if self._embed_fn is None:
            _model = SentenceTransformer("all-MiniLM-L6-v2")

            def embed(texts: list[str]) -> list[list[float]]:
                return _model.encode(texts, normalize_embeddings=True).tolist()

            self._embed_fn = embed
        return self._embed_fn

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return dot # already L2 normalized

    def chunk(self, text: str, document_id: str = "") -> list[SemanticChunkResult]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        embed = self._get_embed_fn()
        vecs = embed(sentences)

        # Detect split points where similarity drops
        boundaries: list[int] = [0]
        for i in range(1, len(sentences)):
            sim = self._cosine(vecs[i-1], vecs[i])
            if sim < self.similarity_thresh:
                boundaries.append(i)
        boundaries.append(len(sentences))

        # Build raw segments from boundaries
        segments: list[str] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segments.append(" ".join(sentences[start:end]))

        # Merge under-sized segments upward
        merged: list[str] = []
        buf = ""

        for seg in segments:
            candidate = (buf + " " + seg).strip()
            if _count_tokens(candidate) < self.min_tokens:
                buf = candidate
            else:
                if buf:
                    merged.append(buf)
                buf = seg
        if buf:
            merged.append(buf)

        # Hard-split oversized segments
        final: list[str] = []
        for seg in merged:
            if _count_tokens(seg) <= self.max_tokens:
                final.append(seg)

            else:
                for para in re.split(r'\n{2,}', seg):
                    if _count_tokens(para) <= self.max_tokens:
                        final.append(para.strip())
                    else:
                        tokens = _ENC.encode(para)
                        for i in range(0, len(tokens), self.max_tokens):
                            window = tokens[i: i + self.max_tokens]
                            final.append(_ENC.decode(window))

        
        # Add sentence-level overlap between chunks
        results: list[SemanticChunkResult] = []
        prev_tail: list[str] = []
        parent_id = str(uuid.uuid4())

        for idx, seg_text in enumerate(final):
            if prev_tail and self.overlap_sentences:
                seg_text = " ".join(prev_tail) + " " + seg_text
            tail_sentences = _split_sentences(seg_text)
            prev_tail = tail_sentences[-self.overlap_sentences:] if tail_sentences else []

            results.append(SemanticChunkResult(
                text=seg_text.strip(),
                token_count=_count_tokens(seg_text),
                chunk_index=idx,
                parent_id=parent_id,
            ))

        return results

        


        

            