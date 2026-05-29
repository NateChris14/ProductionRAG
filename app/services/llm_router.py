"""
LLM router + streaming generator.

Routes queries to the appropriate model based on:
 - Context token count
 - Tenant tier
 - Response latency target

Supports OpenAI streaming via async generator + SSE-compatible chunks.
"""

from __future__ import annotations
from typing import AsyncIterator
import openai
from app.config import get_settings
from app.services.context_builder import CompressedContext
import structlog

log = structlog.get_logger()
_settings = get_settings()

_SYSTEM_PROMPT = """You are a knowledgeable assistant with access to retrieved context.

Answer the user's question based ONLY on the provided context.
If the context does not contain enough information, say so clearly.
Always cite the source numbers [1], [2], etc. when referencing context.
Be concise, accurate, and direct.
"""

def _select_model(context_tokens: int, tenant_tier: str = "standard") -> str:
    """
    Simple routing heuristic. Replace with a learned router in production.
    - Pro tenant always get the strong model.
    - Large context uses the strong model.
    - Short factual queries use the fast model.
    """

    if tenant_tier == "pro":
        return _settings.llm_strong
    if context_tokens > 1200:
        return _settings.llm_strong
    return _settings.llm_fast

class LLMRouter:
    def __init__(self) -> None:
        self._client = openai.AsyncOpenAI(api_key=_settings.openai_api_key)

    def route(
        self,
        context: CompressedContext,
        tenant_tier: str = "standard",
    ) -> str:
        
        model = _select_model(context.tokens, tenant_tier)
        log.info("model_routed", model=model, context_tokens=context.tokens)
        return model

    async def stream(self, query: str, context: CompressedContext, chat_history: list[dict], tenant_tier: str = "standard",) -> AsyncIterator[str]:
        """
        Yields text tokens as they arrive from the LLM.
        Caller wraps this in SSE or WebSocket events.
        """

        model = self.route(context, tenant_tier)
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(chat_history[-6:]) # last 3 turns
        messages.append({
            "role": "user",
            "content": (
                f"Context: \n{context.final_prompt_context}\n\n"
                f"Question: {query}"
            ),
        })

        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.2,
            max_tokens=1024,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def complete(self, query: str, context: CompressedContext, chat_history: list[dict], tenant_tier: str = "standard") -> str:
        """
        Non-streaming path for background / eval tasks.
        """

        model = self.route(context, tenant_tier)
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(chat_history[-6:])
        messages.append({
            "role": "user",
            "content": (
                f"Context: \n{context.final_prompt_context}\n\n"
                f"Question: {query}"
            ),
        })

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )

        return response.choices[0].message.content or ""

