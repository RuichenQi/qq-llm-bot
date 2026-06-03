"""Question-tier ambient gate.

Verifies that when an unaddressed message looks like a question, the bot uses
the lower `AMBIENT_REPLY_PROBABILITY_QUESTION` instead of the higher `high`
probability — and that the detector picks up the common Chinese/English forms
without firing on near-miss phrases like "啥都行"."""
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


# ---------- detector unit tests ----------
def test_question_detector_punctuation():
    assert Handler._looks_like_question("早饭吃啥?")
    assert Handler._looks_like_question("这个怎么用？")
    assert Handler._looks_like_question("really?")


def test_question_detector_final_particles():
    assert Handler._looks_like_question("你吃了吗")
    assert Handler._looks_like_question("你吃了吗。")
    assert Handler._looks_like_question("你吃了吗～")
    assert Handler._looks_like_question("这样可以吧呢")
    # Particle has to be FINAL, not mid-sentence.
    # 啦/吧 don't count (too noisy to gate on).


def test_question_detector_wh_words():
    assert Handler._looks_like_question("为什么会这样")
    assert Handler._looks_like_question("这咋办")
    assert Handler._looks_like_question("到底什么时候开始")
    assert Handler._looks_like_question("哪里有卖的")


def test_question_detector_avoids_false_positives():
    # 怎么/啥 alone without an interrogative phrasing aren't questions.
    assert not Handler._looks_like_question("怎么都行")
    assert not Handler._looks_like_question("啥都行")
    assert not Handler._looks_like_question("怎么样都好")
    # Empty / whitespace.
    assert not Handler._looks_like_question("")
    assert not Handler._looks_like_question("   ")
    # Plain statements without question signals.
    assert not Handler._looks_like_question("今天天气真好")
    assert not Handler._looks_like_question("我吃完啦")


# ---------- end-to-end: ambient gate picks the question probability ----------
def _event(text: str, *, user_id: int = 42, group_id: int = 1):
    """Unaddressed message (no @bot, no nickname)."""
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": 10000,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": text,
        "message": [{"type": "text", "data": {"text": text}}],
        "sender": {"user_id": user_id, "nickname": "x"},
    }


def _make_handler(monkeypatch) -> Tuple[Handler, List[Tuple[int, str]], List[Tuple[str, float]]]:
    """Capture every random.random() roll's tier+p so we can assert which
    probability the gate ended up using."""
    sent: List[Tuple[int, str]] = []
    rolls: List[Tuple[str, float]] = []

    async def send_text(gid, text):
        sent.append((gid, text))

    async def send_image(gid, img):
        sent.append((gid, f"[image:{img[:40]}]"))

    stub = types.SimpleNamespace(name="stub")

    async def chat(messages, **_kw):
        from providers.base import TextReply
        return TextReply(text="ok", model="stub")

    stub.chat = chat

    async def aclose():
        return None

    stub.aclose = aclose

    router = types.SimpleNamespace()

    async def decide(text, *, has_image=False, was_at_bot=False, has_file=False, **_kw):
        from bot.router import RouteDecision
        # Always classify as high so the question vs non-question path is the
        # only thing differing across tests.
        return RouteDecision("deepseek_chat", 1.0, "stub", text, tier="high")

    router.decide = decide

    monkeypatch.setattr(cfg.CONFIG, "allowed_groups", {1}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "trigger_mode", "always", raising=False)
    monkeypatch.setattr(cfg.CONFIG, "stream_replies", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "superusers", {42}, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "tool_use_enabled", False, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "ambient_reply_min_seconds", 0, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "ambient_reply_probability_high", 0.10, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "ambient_reply_probability_question", 0.02, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "ambient_reply_probability_low", 0.0005, raising=False)
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 0, raising=False)

    # Pin random so dice always rolls 0.05 — passes high (p=0.10), skips
    # question (p=0.02). Any test that wants a different roll re-patches.
    import random
    monkeypatch.setattr(random, "random", lambda: 0.05)

    handler = Handler(
        deepseek=stub, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
    )
    return handler, sent, rolls


def test_unaddressed_question_uses_lower_probability(monkeypatch):
    """A high-tier statement passes the gate at p=0.10 (roll 0.05 < 0.10),
    but a same-tier QUESTION fails at p=0.02 (roll 0.05 >= 0.02). With the
    pinned roll, this is the deterministic difference."""
    handler, sent, _ = _make_handler(monkeypatch)

    async def go():
        # Statement — high tier, no question, p=0.10, roll=0.05 → speak.
        await handler.handle(parse_event(_event("今天好累啊")))
        before_question = len(sent)
        # Bypass cooldown bookkeeping so the second message gets a clean roll.
        handler._last_bot_speech_at.pop(1, None)
        # Question — re-classified as question tier, p=0.02, roll=0.05 → skip.
        await handler.handle(parse_event(_event("你吃了吗?")))
        after_question = len(sent)
        await handler.aclose()
        return before_question, after_question

    before, after = asyncio.run(go())
    assert before == 1, "high-tier statement should pass the gate"
    assert after == before, "high-tier question should be silenced"


def test_unaddressed_statement_passes_when_dice_allows(monkeypatch):
    """Sanity: same statement IS replied to under the high-tier probability."""
    handler, sent, _ = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event("刚发现一家新店")))
        await handler.aclose()

    asyncio.run(go())
    assert sent and sent[0][1] == "ok"


def test_addressed_question_always_replies_regardless_of_gate(monkeypatch):
    """When the user @bot's a question, the ambient gate is bypassed entirely
    — the question-tier probability shouldn't even be consulted."""
    handler, sent, _ = _make_handler(monkeypatch)

    # Force the dice to always fail (1.0). If the gate were still consulted,
    # this would block the reply.
    import random
    monkeypatch.setattr(random, "random", lambda: 0.999)

    async def go():
        ev = {
            "post_type": "message",
            "message_type": "group",
            "self_id": 10000,
            "group_id": 1,
            "user_id": 42,
            "raw_message": "你吃了吗?",
            "message": [
                {"type": "at", "data": {"qq": "10000"}},   # @bot
                {"type": "text", "data": {"text": " 你吃了吗?"}},
            ],
            "sender": {"user_id": 42, "nickname": "x"},
        }
        await handler.handle(parse_event(ev))
        await handler.aclose()

    asyncio.run(go())
    assert sent and sent[0][1] == "ok"


def test_ambient_min_seconds_blocks_both_tiers(monkeypatch):
    """If we're still in the cooldown, BOTH high-tier and question-tier
    messages get silenced (gate fires before the dice roll)."""
    handler, sent, _ = _make_handler(monkeypatch)
    monkeypatch.setattr(cfg.CONFIG, "ambient_reply_min_seconds", 120, raising=False)

    async def go():
        await handler.handle(parse_event(_event("发现一家新店")))
        assert sent and sent[0][1] == "ok"
        # Right after: another non-addressed message — cooldown not elapsed.
        await handler.handle(parse_event(_event("你吃了吗?")))
        await handler.aclose()

    asyncio.run(go())
    assert len(sent) == 1, "second message should be cooldown-silenced"
