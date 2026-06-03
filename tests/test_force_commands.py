"""Force-trigger commands: /search, /teach, /forget, /news.

These verify the user-facing commands exist, route to the right handler,
and respect the sensitive-content gate. The underlying providers (web
search, lesson classifier) are stubbed.
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


def _event(text: str, user_id: int = 42, group_id: int = 1):
    # Commands always need @bot; auto-add for test convenience.
    segs = []
    if text.startswith("/"):
        segs.append({"type": "at", "data": {"qq": "10000"}})
    segs.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": text,
        "message": segs,
        "sender": {"user_id": user_id, "nickname": "x"},
    }


def _make_handler(monkeypatch, *, web_search=None, lessons=None) -> Tuple[Handler, List[Tuple[int, str]]]:
    sent: List[Tuple[int, str]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    stub = types.SimpleNamespace(name="stub")

    async def chat(messages, **_kw):
        from providers.base import TextReply
        # Echo last user content so we can verify what was passed in.
        last = messages[-1].content if messages else ""
        return TextReply(text=f"ok:{last[:40]}", model="stub")

    stub.chat = chat

    async def aclose():
        return None

    stub.aclose = aclose

    router = types.SimpleNamespace()

    async def decide(text, *, has_image=False, was_at_bot=False, has_file=False, **_kw):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "stub", text)

    router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "tool_use_enabled", False, raising=False)

    handler = Handler(
        deepseek=stub, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
        web_search=web_search, lessons=lessons,
    )
    return handler, sent


# ---------- /help ----------
def test_help_lists_new_commands(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/help"))))
    asyncio.run(handler.aclose())
    assert sent, "should reply"
    body = sent[0][1]
    for cmd in ("/ask", "/search", "/teach", "/forget", "/file", "/vision",
                "/image", "/edit", "/recap", "/recall", "/timewarp",
                "/remember", "/balance", "/clear"):
        assert cmd in body, f"{cmd} missing from /help"
    # /reset is no longer a top-level command — it lives under /admin reset.
    assert "/admin reset" in body


# ---------- /search ----------
def test_search_without_query_asks_for_one(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/search"))))
    asyncio.run(handler.aclose())
    assert sent and "搜啥" in sent[0][1]


def test_search_without_backend_says_not_configured(monkeypatch):
    handler, sent = _make_handler(monkeypatch, web_search=None)
    asyncio.run(handler.handle(parse_event(_event("/search 雷军"))))
    asyncio.run(handler.aclose())
    assert sent and "TAVILY" in sent[0][1] or "联网" in sent[0][1]


def test_search_blocked_sensitive_query_dodges(monkeypatch):
    """Sensitive queries don't hit the search backend — bot dodges politely."""
    calls: List[str] = []

    class FakeSearch:
        name = "fake"

        async def search(self, q, *, max_results=5):
            calls.append(q)
            return []

    handler, sent = _make_handler(monkeypatch, web_search=FakeSearch())
    asyncio.run(handler.handle(parse_event(_event("/search 六四 真相"))))
    asyncio.run(handler.aclose())
    assert calls == [], "blocked query must not reach the search backend"
    assert sent, "bot should still reply with a dodge"
    assert any("我们聊点别的" in t or "不太懂" in t for _, t in sent)


def test_search_runs_when_enabled(monkeypatch):
    """A normal query reaches the backend AND triggers a chat call."""
    backend_calls: List[str] = []

    class FakeSearch:
        name = "fake"

        async def search(self, q, *, max_results=5):
            backend_calls.append(q)
            from providers.web_search import SearchResult
            return [SearchResult(title="t", url="https://x", snippet="content")]

    handler, sent = _make_handler(monkeypatch, web_search=FakeSearch())
    asyncio.run(handler.handle(parse_event(_event("/search python 3.13 新特性"))))
    asyncio.run(handler.aclose())
    assert backend_calls == ["python 3.13 新特性"]
    assert sent and sent[0][1].startswith("ok:")


# ---------- /teach ----------
def test_teach_without_text_explains_usage(monkeypatch):
    handler, sent = _make_handler(monkeypatch)
    asyncio.run(handler.handle(parse_event(_event("/teach"))))
    asyncio.run(handler.aclose())
    assert sent and "用法" in sent[0][1]


def test_teach_persists_literal_text(monkeypatch):
    """`/teach` goes through teach_raw (no classifier), so the exact wording
    is what gets persisted and later injected into this group's chats."""
    received: List[dict] = []

    class FakeLessons:
        async def teach_raw(self, *, group_id, user_id, text):
            received.append({
                "group_id": group_id, "user_id": user_id, "text": text,
            })
            return 42

        async def maybe_learn(self, *, group_id, user_id, text, addressed=False):
            raise AssertionError("/teach must bypass the LLM classifier")

    monkeypatch.setattr(cfg.CONFIG, "lessons_enabled", True, raising=False)
    handler, sent = _make_handler(monkeypatch, lessons=FakeLessons())
    asyncio.run(handler.handle(parse_event(_event(
        "/teach 群里有人发666你也跟一个"
    ))))
    asyncio.run(handler.aclose())
    assert received == [{
        "group_id": 1, "user_id": 42,
        "text": "群里有人发666你也跟一个",
    }]
    assert sent and "#42" in sent[0][1]
    assert "本群" in sent[0][1], "ack should clarify scope is this group"


def test_teach_rejects_oversized_input(monkeypatch):
    """teach_raw refuses > 500 char text; /teach reports the limit."""
    class FakeLessons:
        async def teach_raw(self, *, group_id, user_id, text):
            return 0  # the real implementation refuses; mirror that here

    monkeypatch.setattr(cfg.CONFIG, "lessons_enabled", True, raising=False)
    handler, sent = _make_handler(monkeypatch, lessons=FakeLessons())
    asyncio.run(handler.handle(parse_event(_event("/teach " + "啊" * 600))))
    asyncio.run(handler.aclose())
    assert sent and "太长" in sent[0][1]


def test_teach_disabled_says_so(monkeypatch):
    monkeypatch.setattr(cfg.CONFIG, "lessons_enabled", False, raising=False)
    handler, sent = _make_handler(monkeypatch, lessons=None)
    asyncio.run(handler.handle(parse_event(_event("/teach 别用 emoji"))))
    asyncio.run(handler.aclose())
    assert sent and "功能注入" in sent[0][1]


# ---------- /forget ----------
def test_forget_aliases_remember_cancel(monkeypatch):
    cancelled: List[List[int]] = []

    class FakeLessons:
        async def list_pending(self, *a, **k):
            return []

        async def cancel(self, lesson_id, group_id):
            cancelled.append([lesson_id])
            return True

        async def cancel_many(self, ids, group_id):
            cancelled.append(list(ids))
            return len(ids)

        async def cancel_all(self, group_id, kind=None):
            cancelled.append(["all" if kind is None else kind])
            return 0

    monkeypatch.setattr(cfg.CONFIG, "lessons_enabled", True, raising=False)
    handler, sent = _make_handler(monkeypatch, lessons=FakeLessons())
    asyncio.run(handler.handle(parse_event(_event("/forget 7"))))
    asyncio.run(handler.aclose())
    # `/forget 7` parses as a single-id list and routes to cancel_many.
    assert cancelled == [[7]]


# ---------- /news ----------
def test_news_without_backend_says_not_configured(monkeypatch):
    """Without a Tavily key /news politely degrades — but it's still
    available to anyone (no superuser gate any more)."""
    handler, sent = _make_handler(monkeypatch, web_search=None)
    asyncio.run(handler.handle(parse_event(_event("/news", user_id=9999))))
    asyncio.run(handler.aclose())
    assert sent and ("TAVILY" in sent[0][1] or "联网" in sent[0][1])
    # Crucially, the message is NOT "仅限超级用户" — non-superusers reach
    # the same code path as everyone else.
    assert "超级用户" not in sent[0][1]
