"""Web search: Tavily client (mocked) + result formatting + provider builder."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import config as cfg
from providers.web_search import (
    SearchResult,
    TavilyProvider,
    WebSearchUnavailable,
    build_provider,
    format_results,
)


def _mock_transport(handler):
    """Wrap a request handler in MockTransport and give back an AsyncClient."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_format_results_empty():
    assert format_results([]) == "No results."


def test_format_results_renders_blocks():
    rows = [
        SearchResult(title="A", url="https://a", snippet="aa"),
        SearchResult(title="B", url="https://b", snippet="bb"),
    ]
    out = format_results(rows)
    assert "[1] A" in out and "[2] B" in out
    assert "https://a" in out and "https://b" in out


def test_format_results_respects_max_chars():
    rows = [SearchResult(title="X" * 100, url="https://x", snippet="y" * 500)]
    out = format_results(rows, max_chars=50)
    assert len(out) <= 50 + 5  # leave a tiny slack for join


def test_tavily_requires_api_key(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "tavily_api_key", "", raising=False)
    with pytest.raises(WebSearchUnavailable):
        TavilyProvider()


def test_tavily_search_happy_path(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "tavily_api_key", "test-key", raising=False)

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["query"] == "雷军 外貌"
        assert body["api_key"] == "test-key"
        assert body["max_results"] == 3
        return httpx.Response(200, json={
            "results": [
                {"title": "雷军百度百科", "url": "https://baike", "content": "小米创始人", "score": 0.9},
                {"title": "雷军微博", "url": "https://weibo", "content": "经常穿牛仔", "score": 0.7},
            ],
        })

    client = _mock_transport(_handler)
    provider = TavilyProvider(client=client, api_key="test-key")

    async def go():
        results = await provider.search("雷军 外貌", max_results=3)
        await provider.aclose()
        return results

    out = asyncio.run(go())
    assert len(out) == 2
    assert out[0].title == "雷军百度百科"
    assert out[1].snippet == "经常穿牛仔"


def test_tavily_skips_empty_rows(monkeypatch):
    """Rows with neither title nor snippet are dropped."""
    monkeypatch.setattr(cfg.CONFIG, "tavily_api_key", "test-key", raising=False)

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "results": [
                {"title": "", "url": "", "content": ""},  # skip
                {"title": "real", "url": "u", "content": "snippet"},
            ],
        })

    client = _mock_transport(_handler)
    provider = TavilyProvider(client=client, api_key="test-key")

    async def go():
        results = await provider.search("q")
        await provider.aclose()
        return results

    out = asyncio.run(go())
    assert len(out) == 1
    assert out[0].title == "real"


def test_tavily_401_becomes_unavailable(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "tavily_api_key", "bad", raising=False)

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    client = _mock_transport(_handler)
    provider = TavilyProvider(client=client, api_key="bad")

    async def go():
        with pytest.raises(WebSearchUnavailable, match="401"):
            await provider.search("q")
        await provider.aclose()

    asyncio.run(go())


def test_build_provider_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "web_search_enabled", False, raising=False)
    assert build_provider() is None


def test_build_provider_returns_none_without_key(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "web_search_enabled", True, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "tavily_api_key", "", raising=False)
    assert build_provider() is None


def test_build_provider_unknown_backend_returns_none(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "web_search_enabled", True, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "web_search_provider", "bogus", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "tavily_api_key", "anything", raising=False)
    assert build_provider() is None
