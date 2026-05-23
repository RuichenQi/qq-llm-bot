"""Web search backend.

Currently wraps Tavily (the simplest LLM-oriented search API). The abstract
`WebSearchProvider` interface makes it easy to swap in Brave / SerpAPI / a
local search engine later without changing the tool wrapper or handler.

`WebSearchUnavailable` is the canonical "search isn't configured" signal — the
tool handler catches it and returns a polite message so the LLM can still
compose a useful reply ("I tried to look this up but search isn't set up").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import httpx

from bot.logger import get_logger
from config import CONFIG

log = get_logger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0


class WebSearchUnavailable(RuntimeError):
    """Raised when search is asked for but isn't configured."""


class WebSearchProvider(Protocol):
    name: str

    async def search(
        self, query: str, *, max_results: int = 5,
    ) -> List[SearchResult]: ...


class TavilyProvider:
    """Tavily search API client. https://docs.tavily.com/

    Free tier: 1000 searches/month. Designed for LLM grounding — results are
    short, deduplicated, and ranked by relevance. Single POST per query.
    """

    name = "tavily"

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.tavily.com",
    ) -> None:
        self._api_key = api_key or CONFIG.tavily_api_key
        if not self._api_key:
            raise WebSearchUnavailable(
                "TAVILY_API_KEY not set; web search is disabled"
            )
        self._client = client or httpx.AsyncClient(timeout=20.0)
        self._url = f"{base_url.rstrip('/')}/search"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(
        self, query: str, *, max_results: int = 5,
    ) -> List[SearchResult]:
        body = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max(1, min(int(max_results), 10)),
            "search_depth": "basic",  # "advanced" costs more credits
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            resp = await self._client.post(self._url, json=body)
        except httpx.HTTPError as e:
            raise WebSearchUnavailable(f"Tavily network error: {e}") from e
        if resp.status_code == 401:
            raise WebSearchUnavailable("Tavily rejected the API key (401)")
        if resp.status_code >= 400:
            raise WebSearchUnavailable(
                f"Tavily HTTP {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json() or {}
        raw_results = data.get("results") or []
        out: List[SearchResult] = []
        for r in raw_results:
            title = str(r.get("title") or "").strip()
            url = str(r.get("url") or "").strip()
            snippet = str(r.get("content") or "").strip()
            if not title and not snippet:
                continue
            try:
                score = float(r.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            out.append(SearchResult(
                title=title, url=url, snippet=snippet, score=score,
            ))
        log.info("tavily query=%r → %d result(s)", query[:80], len(out))
        return out


def build_provider() -> Optional[WebSearchProvider]:
    """Return a ready-to-use search provider per CONFIG, or None if disabled.

    Treats "no API key" as "disabled" (returns None) rather than raising —
    lets the bot boot without search configured.
    """
    if not CONFIG.web_search_enabled:
        return None
    name = (CONFIG.web_search_provider or "tavily").lower()
    try:
        if name == "tavily":
            return TavilyProvider()
        log.warning("unknown WEB_SEARCH_PROVIDER=%r — search disabled", name)
        return None
    except WebSearchUnavailable as e:
        log.warning("web search disabled: %s", e)
        return None


def format_results(results: List[SearchResult], *, max_chars: int = 1800) -> str:
    """Render results into a compact block for the LLM. Capped so a verbose
    search doesn't blow the next chat call's prompt budget."""
    if not results:
        return "No results."
    lines: List[str] = []
    used = 0
    for i, r in enumerate(results, 1):
        block = f"[{i}] {r.title}\n{r.url}\n{r.snippet}".strip()
        if used + len(block) + 2 > max_chars:
            break
        lines.append(block)
        used += len(block) + 2
    return "\n\n".join(lines)
