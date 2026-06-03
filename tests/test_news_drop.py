"""News drop: `/news` force-trigger + daily `send_news_to_all_groups` fan-out.

Verifies the new shape (post-refactor):
  - `/news` is available to ANY user (not just superusers)
  - `/news` with no args uses CONFIG.news_query (broad search)
  - `/news <topic>` overrides the search query for that call
  - `send_news_to_all_groups` posts to every eligible group in one fire
  - Paused groups + within-cooldown groups are skipped
  - No search backend → graceful skip
"""
from __future__ import annotations

import asyncio
import types
from typing import List, Tuple

import config as cfg
from bot.command_handler import Handler
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


def _event(text: str, *, user_id: int = 42, group_id: int = 1):
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": text,
        "message": [
            {"type": "at", "data": {"qq": "10000"}},
            {"type": "text", "data": {"text": text}},
        ],
        "sender": {"user_id": user_id, "nickname": "x"},
    }


class _FakeSearch:
    """Captures every query then returns canned headlines."""
    name = "fake"

    def __init__(self, results=None):
        self.calls = []
        from providers.web_search import SearchResult
        self._results = results if results is not None else [
            SearchResult(title="某航天公司发射成功", url="https://example.com/a",
                         snippet="发射于 03:14, 载荷送达预定轨道"),
            SearchResult(title="新发现的系外行星", url="https://example.com/b",
                         snippet="距地 30 光年, 大气富含水汽"),
        ]

    async def search(self, q, *, max_results=5):
        self.calls.append((q, max_results))
        return list(self._results)


def _make_handler(monkeypatch, *, web_search=None) -> Tuple[Handler, List[Tuple[int, str]]]:
    sent: List[Tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    stub = types.SimpleNamespace(name="stub")

    async def chat(messages, **_kw):
        from providers.base import TextReply
        return TextReply(text="刚刷到，挺有意思的，发射成功了，还有个新行星。", model="stub")

    stub.chat = chat

    async def aclose():
        return None

    stub.aclose = aclose

    router = types.SimpleNamespace()

    async def decide(text, *, has_image=False, was_at_bot=False, has_file=False, **_kw):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "stub", text)

    router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1, 2, 3}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "tool_use_enabled", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "news_enabled", True, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "news_query", "今日热点新闻", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "news_min_interval_hours", 20, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "news_search_max_results", 7, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 0, raising=False)

    return Handler(
        deepseek=stub, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
        web_search=web_search,
    ), sent


# ---------- /news force-trigger ----------
def test_news_command_works_for_any_user(monkeypatch):
    """Non-superuser can call /news; no superuser gate."""
    fake = _FakeSearch()
    handler, sent = _make_handler(monkeypatch, web_search=fake)
    asyncio.run(handler.handle(parse_event(_event("/news", user_id=9999))))
    asyncio.run(handler.aclose())
    assert fake.calls, "web_search should run regardless of caller's role"
    assert sent and "发射" in sent[-1][1]


def test_news_command_no_topic_uses_default_query(monkeypatch):
    """`/news` with no args searches CONFIG.news_query verbatim."""
    fake = _FakeSearch()
    handler, sent = _make_handler(monkeypatch, web_search=fake)
    asyncio.run(handler.handle(parse_event(_event("/news"))))
    asyncio.run(handler.aclose())
    assert fake.calls and fake.calls[0][0] == "今日热点新闻"


def test_news_command_with_custom_topic(monkeypatch):
    """`/news <custom>` overrides the default query for that call only."""
    fake = _FakeSearch()
    handler, sent = _make_handler(monkeypatch, web_search=fake)
    asyncio.run(handler.handle(parse_event(_event("/news 最近的 GPU 价格新闻"))))
    asyncio.run(handler.aclose())
    assert fake.calls and fake.calls[0][0] == "最近的 GPU 价格新闻"


def test_news_command_no_results_replies_politely(monkeypatch):
    """Empty search results → polite skip, not crash."""
    handler, sent = _make_handler(
        monkeypatch, web_search=_FakeSearch(results=[]),
    )
    asyncio.run(handler.handle(parse_event(_event("/news"))))
    asyncio.run(handler.aclose())
    assert sent and "没刷到" in sent[-1][1]


def test_news_command_search_failure_replies_politely(monkeypatch):
    """Search backend raising must not crash the handler."""
    class BrokenSearch:
        name = "broken"
        async def search(self, q, **_kw):
            raise RuntimeError("network down")

    handler, sent = _make_handler(monkeypatch, web_search=BrokenSearch())
    asyncio.run(handler.handle(parse_event(_event("/news"))))
    asyncio.run(handler.aclose())
    assert sent and "没刷到" in sent[-1][1]


def test_news_command_no_backend_says_not_configured(monkeypatch):
    handler, sent = _make_handler(monkeypatch, web_search=None)
    asyncio.run(handler.handle(parse_event(_event("/news"))))
    asyncio.run(handler.aclose())
    assert sent and ("TAVILY" in sent[0][1] or "联网" in sent[0][1])


# ---------- sensitive-content gate on news ----------
def test_news_command_sensitive_topic_refused_without_searching(monkeypatch):
    """If the user passes a CN-political topic to /news, we refuse before
    even hitting the search backend."""
    fake = _FakeSearch()
    handler, sent = _make_handler(monkeypatch, web_search=fake)
    asyncio.run(handler.handle(parse_event(_event("/news 六四 三十周年"))))
    asyncio.run(handler.aclose())
    assert fake.calls == [], "sensitive query must not hit the search API"
    assert sent and "我不方便聊" in sent[-1][1]


def test_bare_news_never_shows_refusal_even_if_default_query_blocked(monkeypatch):
    """The polite "我不方便聊" message is reserved for user-typed sensitive
    topics. A bare `/news` (no args) must NEVER say that — even if the
    default NEWS_QUERY is misconfigured to a sensitive phrase, the user
    didn't ask for it, so they shouldn't see a refusal aimed at them."""
    fake = _FakeSearch(results=[])  # no results → triggers "今早没刷到" fallback
    handler, sent = _make_handler(monkeypatch, web_search=fake)
    # Force the default query to something that trips _query_is_blocked.
    monkeypatch.setattr(cfg.CONFIG, "news_query", "六四 真相", raising=False)
    asyncio.run(handler.handle(parse_event(_event("/news"))))  # bare /news
    asyncio.run(handler.aclose())
    assert sent
    body = sent[-1][1]
    assert "我不方便聊" not in body, (
        "bare /news must not look like a refusal aimed at the caller"
    )
    # And the user gets the generic fallback instead.
    assert "没刷到" in body


def test_send_news_searches_even_when_query_string_is_blocked(monkeypatch):
    """A flagged word in the query string itself must NOT short-circuit the
    search. We still hit Tavily and lean on the per-result filter — so a
    daily auto-fire with one stray sensitive result still posts the other
    items, instead of silently bailing the whole group."""
    from providers.web_search import SearchResult

    fake_calls: List[str] = []

    class MixedSearch:
        name = "mixed"
        async def search(self, q, **_kw):
            fake_calls.append(q)
            # First result is sensitive (would have tripped belt #1 alone);
            # the remaining two are safe.
            return [
                SearchResult(title="习近平 出席气候峰会", url="https://x", snippet="..."),
                SearchResult(title="James Webb 拍到新星云", url="https://y", snippet="space"),
                SearchResult(title="某新硬件 发售", url="https://z", snippet="gadget"),
            ]

    chat_calls: List[str] = []

    async def chat(messages, **_kw):
        from providers.base import TextReply
        chat_calls.append(messages[-1].content)
        return TextReply(text="今早看到 James Webb 拍到星云，还有新硬件发售。", model="stub")

    handler, sent = _make_handler(monkeypatch, web_search=MixedSearch())
    handler.deepseek.chat = chat  # type: ignore[attr-defined]

    # Query string itself trips _query_is_blocked, but we expect search to
    # still run and the safe results to make it to the LLM.
    ok = asyncio.run(handler.send_news(group_id=1, query="某新闻 关于敏感词 评价"))
    asyncio.run(handler.aclose())
    assert ok is True
    # Tavily was actually invoked — the old belt #1 would have skipped this.
    assert fake_calls == ["某新闻 关于敏感词 评价"]
    body = chat_calls[0]
    # The sensitive headline ("习近平 出席气候峰会") was filtered out — its
    # distinguishing words (the unique part of the title) must not reach
    # the LLM. The safe headlines did.
    assert "出席气候峰会" not in body, (
        "sensitive headline must be filtered out of the prompt"
    )
    assert "James Webb" in body or "星云" in body
    assert "新硬件" in body or "硬件" in body
    # And something landed in the group.
    assert sent


def test_news_results_all_sensitive_bails(monkeypatch):
    """If EVERY search result trips the sensitive-content filter we bail
    silently rather than try to thread the needle."""
    from providers.web_search import SearchResult

    class SensitiveSearch:
        name = "sensitive"
        async def search(self, q, **_kw):
            return [
                SearchResult(title="习近平 出席会议", url="https://x", snippet="..."),
                SearchResult(title="新疆 集中营 报告", url="https://y", snippet="..."),
                SearchResult(title="香港 抗议 新闻", url="https://z", snippet="..."),
            ]

    handler, sent = _make_handler(monkeypatch, web_search=SensitiveSearch())
    asyncio.run(handler.handle(parse_event(_event("/news 今日要闻"))))
    asyncio.run(handler.aclose())
    # Bot tells the group "today nothing to share" rather than posting any of
    # those results.
    assert sent and "没刷到" in sent[-1][1]


def test_news_partial_filter_keeps_safe_results(monkeypatch):
    """When SOME results are sensitive and some aren't, only the safe ones
    reach the LLM."""
    from providers.web_search import SearchResult

    forwarded: List[str] = []

    class MixedSearch:
        name = "mixed"
        async def search(self, q, **_kw):
            return [
                SearchResult(title="习近平 讲话", url="https://x", snippet="political"),
                SearchResult(title="James Webb 拍到新星云", url="https://y", snippet="space"),
                SearchResult(title="某新游戏发售", url="https://z", snippet="game"),
            ]

    async def chat(messages, **_kw):
        from providers.base import TextReply
        # Capture the user-side prompt so we can verify what made it through.
        forwarded.append(messages[-1].content)
        return TextReply(text="今早看到 James Webb 拍到新星云，还有个新游戏发售。", model="stub")

    handler, sent = _make_handler(monkeypatch, web_search=MixedSearch())
    handler.deepseek.chat = chat  # type: ignore[attr-defined]

    asyncio.run(handler.send_news(group_id=1))
    asyncio.run(handler.aclose())

    assert forwarded, "deepseek chat should still be invoked for the safe results"
    body = forwarded[0]
    assert "习近平" not in body, "sensitive headline must be filtered"
    assert "James Webb" in body or "星云" in body
    assert "新游戏" in body or "游戏" in body


# ---------- send_news prompt content (LLM filter rules) ----------
def test_send_news_prompt_includes_filter_rules(monkeypatch):
    """The chat call's prompt must instruct the LLM to filter results by
    interestingness — skip CN politics, gossip, disasters; prefer tech /
    science / quirky. Otherwise we're back to the old hardcoded-topic shape."""
    fake = _FakeSearch()
    captured: List[List] = []

    async def chat(messages, **_kw):
        from providers.base import TextReply
        captured.append(list(messages))
        return TextReply(text="ok", model="stub")

    handler, _ = _make_handler(monkeypatch, web_search=fake)
    handler.deepseek.chat = chat  # type: ignore[attr-defined]

    asyncio.run(handler.send_news(group_id=1))
    asyncio.run(handler.aclose())

    assert captured, "deepseek.chat should be called"
    user_msg = next(m for m in captured[0] if m.role == "user")
    body = user_msg.content
    # The filter explicitly excludes certain categories.
    assert "中国大陆政治" in body
    assert "灾难" in body or "凶杀" in body
    # And actively biases toward the persona-preferred categories.
    assert "科技" in body or "科学" in body
    # And tells the LLM to bail rather than hallucinate when nothing fits.
    assert "没刷到啥能聊的" in body


# ---------- send_news_to_all_groups (daily fan-out) ----------
def test_send_news_to_all_groups_posts_once_per_group(monkeypatch):
    """A single daily fire reaches every allowed, non-paused group exactly
    once. Subsequent fires within the per-group cooldown are no-ops."""
    fake = _FakeSearch()
    handler, sent = _make_handler(monkeypatch, web_search=fake)

    async def go():
        # First daily fire: posts to all three.
        n1 = await handler.send_news_to_all_groups()
        # Second daily fire (simulated same day): cooldown blocks every group.
        n2 = await handler.send_news_to_all_groups()
        await handler.aclose()
        return n1, n2

    n1, n2 = asyncio.run(go())
    assert n1 == 3, "every allowed group should receive one post on first fire"
    assert n2 == 0, "second same-day fire is fully cooldown-blocked"
    delivered = {gid for gid, _ in sent}
    assert delivered == {1, 2, 3}


def test_send_news_to_all_groups_skips_paused(monkeypatch):
    """`/stop`-paused groups are silenced even on the daily fire."""
    from bot import allowlist
    fake = _FakeSearch()
    handler, sent = _make_handler(monkeypatch, web_search=fake)

    async def go():
        await allowlist.pause(1, by_user_id=42)
        n = await handler.send_news_to_all_groups()
        await handler.aclose()
        return n

    n = asyncio.run(go())
    assert n == 2
    assert {gid for gid, _ in sent} == {2, 3}


def test_send_news_to_all_groups_no_backend_returns_zero(monkeypatch):
    handler, sent = _make_handler(monkeypatch, web_search=None)
    n = asyncio.run(handler.send_news_to_all_groups())
    asyncio.run(handler.aclose())
    assert n == 0
    assert sent == []


# ---------- main.py daily-schedule helper ----------
def test_seconds_until_news_time_returns_next_occurrence(monkeypatch):
    """The schedule helper should pick the NEXT NEWS_TIME in NEWS_TIME_TZ
    (today if still ahead, otherwise tomorrow). Sanity-check the return
    value lies in the (0, 24h] band."""
    from main import _seconds_until_news_time
    monkeypatch.setattr(cfg.CONFIG, "news_time", "09:00", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "news_time_tz", "Asia/Shanghai", raising=False)
    s = _seconds_until_news_time()
    assert 0 < s <= 24 * 3600


def test_seconds_until_news_time_handles_bad_tz(monkeypatch):
    """A typo in NEWS_TIME_TZ should fall back to UTC and not crash."""
    from main import _seconds_until_news_time
    monkeypatch.setattr(cfg.CONFIG, "news_time", "09:00", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "news_time_tz", "Not/A/Real/Zone", raising=False)
    s = _seconds_until_news_time()
    assert 0 < s <= 24 * 3600
