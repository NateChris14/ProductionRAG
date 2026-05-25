"""
Query normalization, rewriting, and expansion.

Produces multiple query variants to improve recall in hybrid retrieval:

- Canonical: cleaned, spell-corrected, acronym-expanded
- Semantic paraphrase: different wording, same intent
- Keyword-dense: focused on most important concepts
"""

from __future__ import annotations
import re
import json
import openai
from app.config import get_settings
import structlog

log = structlog.get_logger()
_settings = get_settings()

_REWRITE_PROMPT = """You are a search query optimizer for a RAG system.
Given the user query, produce 2 rewrites:
1. A semantically equivalent but differently worded version.
2. A keyword-dense version focused on the most important concepts.

Return only a JSON array of 2 strings, e.g. ["query1", "query2"]. No explanation.

Query: {query}"""

class QueryProcessor:
    """
    Normalizes and expands a raw user query before retrieval.

    Returns:
        canonical: cleaned original query
        rewrites: LLM-generated variants (2 by default)
        all_queries: [canonical] + rewrites - feed all into retrieval
    """

    def __init__(self):
        self._client = openai.AsyncOpenAI(api_key=_settings.openai_api_key)

    @staticmethod
    def normalize(query: str) -> str:
        """Basic normalization: strip extra whitespace, fix unicode."""

        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        query = (
            query
            .replace('\u2019',"'")
            .replace('\u201c','"')
            .replace('\u201d','"')
        )
        return query

    async def rewrite(self, query: str) -> list[str]:
        """
        Returns [canonical] + LLM rewrites.
        Falls back gracefully to [canonical] on any error.
        """

        canonical = self.normalize(query)
        try:
            response = await self._client.chat.completions.create(
                model=_settings.llm_fast,
                messages=[
                    {
                        "role": "user",
                        "content": _REWRITE_PROMPT.format(query=canonical),
                    }
                ],

                temperature=0.3,
                max_tokens=256,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content or "[]"
            parsed = json.loads(raw)

            # Model may return {"queries": [...]} or just [...]
            if isinstance(parsed, dict):
                variants = list(parsed.values())[0] if parsed else []
            else:
                variants = parsed

            variants = [
                v for v in variants
                if isinstance(v, str) and v.strip()
            ]

        except Exception as e:
            log.warning("query_rewrite_failed", error=str(e))
            variants = []

        all_queries = [canonical] + variants[:2]
        log.info("query_rewrites", count=len(all_queries))
        return all_queries