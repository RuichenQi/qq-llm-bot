"""End-to-end proof: text saved via /teach actually reaches the LLM's
system prompt on every subsequent chat call in that group.

Stubbed: the OneBot transport (we call handler.handle directly with parsed
events) and the DeepSeek `chat()` method (we capture the messages it sees).
Real: SQLite storage + the lessons module's read path + the persona / lessons
ordering in `_run_text`.
"""
from __future__ import annotations

import asyncio
import types
from typing import List, Tuple

import config as cfg
from bot.command_handler import Handler
from bot.lessons import Lessons
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


def _event(text: str, *, user_id: int = 42, group_id: int = 1) -> dict:
    segs = [
        {"type": "at", "data": {"qq": "10000"}},
        {"type": "text", "data": {"text": text}},
    ]
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


def _make_handler(monkeypatch) -> Tuple[Handler, List[List]]:
    """Return a handler whose chat() captures every message list it was called
    with, so the test can assert what the LLM actually saw."""
    captured: List[List] = []

    stub = types.SimpleNamespace(name="stub")

    async def chat(messages, **_kw):
        from providers.base import TextReply
        captured.append(list(messages))
        return TextReply(text="ok", model="stub")

    stub.chat = chat

    async def aclose():
        return None

    stub.aclose = aclose

    router = types.SimpleNamespace()

    async def decide(text, *, has_image=False, was_at_bot=False, has_file=False, **_kw):
        from bot.router import RouteDecision
        return RouteDecision("deepseek_chat", 1.0, "stub", text)

    router.decide = decide

    async def send_text(gid, text):
        pass

    async def send_image(gid, img):
        pass

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1, 2}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "tool_use_enabled", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "lessons_enabled", True, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "lessons_inject_limit", 12, raising=False)
    # Tests below send multiple chats from the same group — disable cooldown
    # so the second/third chat aren't silently swallowed.
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 0, raising=False)

    return Handler(
        deepseek=stub, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
        # Provide a real Lessons module so the chat path's active_for_user()
        # query goes through to SQLite (and surfaces the /teach rows).
        lessons=Lessons(stub),  # type: ignore[arg-type]
    ), captured


# ---------- core proof: /teach text reaches the chat's system prompt ----------
def test_teach_text_appears_in_subsequent_chat_system_prompt(monkeypatch):
    """The user teaches a rule, then chats. The chat call's system prompts
    must contain the literal text the user passed to /teach."""
    handler, captured = _make_handler(monkeypatch)

    async def go():
        # Step 1: teach.
        await handler.handle(parse_event(_event(
            "/teach 本群讨论 ML 论文，回复时多引用 paper"
        )))
        # Step 2: a separate chat turn.
        await handler.handle(parse_event(_event("这篇论文你怎么看")))
        await handler.aclose()

    asyncio.run(go())

    # Find the chat call. The background lesson classifier ALSO calls
    # stub.chat with a prompt that *contains* the user's text inside its
    # own JSON-classification template, so substring matching would grab
    # the classifier call. Use exact equality to home in on the chat path.
    chat_calls = [
        msgs for msgs in captured
        if any(m.role == "user" and m.content == "这篇论文你怎么看" for m in msgs)
    ]
    assert chat_calls, "the user's chat turn must hit deepseek.chat"
    msgs = chat_calls[0]

    # Look at all `system` content concatenated.
    system_text = "\n".join(m.content for m in msgs if m.role == "system")

    # The literal /teach text is in there.
    assert "本群讨论 ML 论文，回复时多引用 paper" in system_text, (
        f"/teach text missing from system prompt. Got:\n{system_text}"
    )
    # And it landed in the dedicated strong-rules block (which has the
    # override language), not the softer advisory block.
    assert "强制规则" in system_text


# ---------- per-group isolation ----------
def test_teach_in_group_a_does_not_leak_into_group_b(monkeypatch):
    """A rule taught in group 1 must NOT appear in group 2's chat prompts."""
    handler, captured = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event(
            "/teach 本群是程序员群，回复尽量技术化", group_id=1,
        )))
        # Chat in group 2 — the rule should NOT be injected.
        await handler.handle(parse_event(_event("早", group_id=2)))
        await handler.aclose()

    asyncio.run(go())

    # Exact equality — see comment in the first test.
    chat_calls = [
        msgs for msgs in captured
        if any(m.role == "user" and m.content == "早" for m in msgs)
    ]
    # Filter out the classifier call (its user content embeds "早" in a
    # bigger template; we want only the chat call whose user msg IS "早").
    chat_calls = [
        msgs for msgs in chat_calls
        if any(m.role == "system"
               and ("行为规则" in m.content or "{nickname}" in m.content
                    or "QQ 群" in m.content) for m in msgs)
    ]
    assert chat_calls, "group 2's chat must hit deepseek.chat with persona/lessons"
    system_text = "\n".join(m.content for m in chat_calls[0] if m.role == "system")
    assert "本群是程序员群" not in system_text, (
        "group 1's /teach text leaked into group 2's prompt"
    )


# ---------- persistence across multiple chat turns ----------
def test_teach_persists_across_many_chats(monkeypatch):
    """A single /teach should keep injecting on every subsequent chat —
    not just the first one — until /forget removes it."""
    handler, captured = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event(
            "/teach 你回复时要带括号注释"
        )))
        for q in ("早", "在干啥", "今天天气咋样"):
            await handler.handle(parse_event(_event(q)))
        await handler.aclose()

    asyncio.run(go())

    seen = set()
    for q in ("早", "在干啥", "今天天气咋样"):
        chat_call = next(
            (msgs for msgs in captured
             if any(m.role == "user" and m.content == q for m in msgs)),
            None,
        )
        assert chat_call is not None, f"no chat call captured for {q!r}"
        seen.add(q)
        system_text = "\n".join(m.content for m in chat_call if m.role == "system")
        assert "你回复时要带括号注释" in system_text, (
            f"teach rule missing from chat for {q!r}; system was:\n{system_text}"
        )
    assert seen == {"早", "在干啥", "今天天气咋样"}


# ---------- forget removes the injection ----------
def test_forget_removes_injection_from_subsequent_chats(monkeypatch):
    handler, captured = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event(
            "/teach 你说话要简短"
        )))
        # Confirm one chat sees it.
        await handler.handle(parse_event(_event("hi")))
        # Find the inserted row id from the SQLite side so we can /forget it.
        active = await handler.lessons.active_for_user(1, 42, limit=10)
        teach_row = next(a for a in active if a.content == "你说话要简短")
        await handler.handle(parse_event(_event(f"/forget {teach_row.id}")))
        # Chat again — rule should be gone.
        await handler.handle(parse_event(_event("hi again")))
        await handler.aclose()

    asyncio.run(go())

    pre_forget = next(
        msgs for msgs in captured
        if any(m.role == "user" and m.content == "hi" for m in msgs)
    )
    post_forget = next(
        msgs for msgs in captured
        if any(m.role == "user" and m.content == "hi again" for m in msgs)
    )
    assert post_forget is not pre_forget
    pre_text = "\n".join(m.content for m in pre_forget if m.role == "system")
    post_text = "\n".join(m.content for m in post_forget if m.role == "system")
    assert "你说话要简短" in pre_text
    assert "你说话要简短" not in post_text
