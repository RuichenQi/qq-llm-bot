"""Two regressions:

1. After `/forget`, a /teach'd rule must NOT linger anywhere the bot can
   re-find it on the next chat. Verified end-to-end through the chat path:
   the captured chat() call's system prompt is empty of the cancelled rule
   AND the lessons row is physically gone from SQLite.

2. /teach'd rules are tagged with `teach` and get rendered in a STRONG
   system block with override language. Auto-classified rules (from the
   background classifier) land in a softer advisory block. Both are
   group-scoped.
"""
from __future__ import annotations

import asyncio
import time
import types
from typing import List, Tuple

import config as cfg
from bot.command_handler import Handler
from bot.lessons import ActiveLesson, Lessons
from bot.memory import Memory
from bot.message_parser import parse_event
from bot.quota import Quota
from bot.rate_limit import RateLimiter


# ---------- format_strong_rules / format_advisory_lessons ----------
def test_strong_rules_only_picks_teach_tagged_rules():
    rows = [
        ActiveLesson(1, "rule", None, "本群讨论 ML 论文", 0.8, "teach", None),
        ActiveLesson(2, "rule", None, "说话简短", 0.6, "", None),  # auto-classified
        ActiveLesson(3, "fact", 42, "对花生过敏", 0.7, "", None),
    ]
    strong = Lessons.format_strong_rules(rows)
    advisory = Lessons.format_advisory_lessons(rows, speaker_user_id=42)
    assert strong is not None
    assert "本群讨论 ML 论文" in strong
    assert "说话简短" not in strong          # auto-rule stays in advisory
    assert "强制规则" in strong               # strong header marker
    assert "最高优先级" in strong             # explicit priority claim
    assert advisory is not None
    assert "说话简短" in advisory
    assert "对花生过敏" in advisory
    assert "本群讨论 ML 论文" not in advisory  # taught rule excluded from advisory


def test_strong_rules_none_when_no_teach_rows():
    rows = [
        ActiveLesson(1, "rule", None, "说话简短", 0.6, "", None),
        ActiveLesson(2, "fact", 1, "x", 0.5, "", None),
    ]
    assert Lessons.format_strong_rules(rows) is None
    # advisory still renders.
    assert Lessons.format_advisory_lessons(rows, speaker_user_id=1) is not None


def test_advisory_none_when_only_teach_rules():
    rows = [
        ActiveLesson(1, "rule", None, "用英文回复", 0.8, "teach", None),
    ]
    assert Lessons.format_strong_rules(rows) is not None
    assert Lessons.format_advisory_lessons(rows, speaker_user_id=1) is None


def test_format_for_prompt_backward_compat_combines_both():
    rows = [
        ActiveLesson(1, "rule", None, "本群讲英文", 0.8, "teach", None),
        ActiveLesson(2, "fact", 42, "对花生过敏", 0.7, "", None),
    ]
    block = Lessons.format_for_prompt(rows, speaker_user_id=42)
    assert block is not None
    assert "本群讲英文" in block
    assert "对花生过敏" in block


# ---------- end-to-end: chat() sees TWO distinct system messages ----------
def _event(text: str, *, user_id: int = 42, group_id: int = 1) -> dict:
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


def _make_handler(monkeypatch) -> Tuple[Handler, List[List]]:
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
    monkeypatch.setattr(cfg.CONFIG, "reply_cooldown_seconds", 0, raising=False)

    return Handler(
        deepseek=stub, openai=None, router=router,
        memory=Memory(), quota=Quota(), rate=RateLimiter(per_minute=999),
        send_text=send_text, send_image=send_image,
        lessons=Lessons(stub),  # type: ignore[arg-type]
    ), captured


def test_teach_emits_dedicated_strong_system_message(monkeypatch):
    """The chat call's `messages` list should contain a system message whose
    content matches the strong block — separate from persona, separate from
    any advisory block."""
    handler, captured = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event("/teach 本群只用英文回复")))
        await handler.handle(parse_event(_event("hi")))
        await handler.aclose()

    asyncio.run(go())
    chat_msgs = next(
        msgs for msgs in captured
        if any(m.role == "user" and m.content == "hi" for m in msgs)
    )
    system_texts = [m.content for m in chat_msgs if m.role == "system"]
    # Exactly one of the system messages is the strong-rules block.
    strong_blocks = [t for t in system_texts if "强制规则" in t]
    assert len(strong_blocks) == 1
    assert "本群只用英文回复" in strong_blocks[0]
    assert "最高优先级" in strong_blocks[0]
    # And it's a SEPARATE message from the persona one.
    persona_blocks = [t for t in system_texts if "{nickname}" in t or "我是" in t or "QQ 群" in t]
    assert persona_blocks, "persona system message should also be present"
    assert strong_blocks[0] not in persona_blocks


def test_teach_then_cancel_no_residue_anywhere(monkeypatch):
    """Reproduces the exact user-reported bug:
       1) /teach a rule
       2) verify it's in the next chat's system prompt
       3) /forget it
       4) verify NEXT chat has zero residue (no system block mentions it,
          and the SQLite row is physically gone).
    """
    handler, captured = _make_handler(monkeypatch)

    async def go():
        await handler.handle(parse_event(_event(
            "/teach 本群讨论 ML 论文，回复时多引用 paper"
        )))
        await handler.handle(parse_event(_event("第一次聊天")))
        active = await handler.lessons.active_for_user(1, 42, limit=10)
        target = next(a for a in active if "ML 论文" in a.content)
        await handler.handle(parse_event(_event(f"/forget {target.id}")))
        await handler.handle(parse_event(_event("第二次聊天")))
        await handler.aclose()
        # Direct DB check.
        from bot.storage import Storage
        store = await Storage.get()
        async with store._conn.execute(  # type: ignore[union-attr]
            "SELECT COUNT(*) FROM lessons WHERE id=?", (target.id,),
        ) as cur:
            (count,) = await cur.fetchone()
        return target.id, count

    target_id, db_count = asyncio.run(go())

    pre_chat = next(
        msgs for msgs in captured
        if any(m.role == "user" and m.content == "第一次聊天" for m in msgs)
    )
    post_chat = next(
        msgs for msgs in captured
        if any(m.role == "user" and m.content == "第二次聊天" for m in msgs)
    )
    pre_text = "\n".join(m.content for m in pre_chat if m.role == "system")
    post_text = "\n".join(m.content for m in post_chat if m.role == "system")
    # Before /forget: rule appears in the chat's system prompt.
    assert "本群讨论 ML 论文，回复时多引用 paper" in pre_text
    # After /forget: rule is gone from BOTH the system prompt AND the DB.
    assert "本群讨论 ML 论文，回复时多引用 paper" not in post_text
    assert db_count == 0, "lessons row must be physically deleted, not just hidden"


def test_taught_and_auto_rules_land_in_different_blocks(monkeypatch):
    """A taught rule and an auto-classified rule should end up in SEPARATE
    system messages (strong vs advisory)."""
    handler, captured = _make_handler(monkeypatch)

    async def go():
        # /teach saves with tags="teach".
        await handler.lessons.teach_raw(
            group_id=1, user_id=42, text="本群只用英文回复",
        )
        # Directly insert an "auto-classified" rule (kind=rule, no teach tag),
        # so we don't need to mock the classifier.
        from bot.storage import Storage
        store = await Storage.get()
        await store.lesson_insert(
            group_id=1, kind="rule", subject_user_id=None,
            content="说话别太正式", importance=0.5, tags="",
            trigger_at=None, recurrence=None, expires_at=None,
            source_user_id=42, source_text="说话别太正式",
            created_at=time.time(),
        )
        await handler.handle(parse_event(_event("hi")))
        await handler.aclose()

    asyncio.run(go())
    chat_msgs = next(
        msgs for msgs in captured
        if any(m.role == "user" and m.content == "hi" for m in msgs)
    )
    sys_msgs = [m.content for m in chat_msgs if m.role == "system"]
    strong = [t for t in sys_msgs if "强制规则" in t]
    advisory = [t for t in sys_msgs if "自动学到" in t]
    assert len(strong) == 1
    assert "本群只用英文回复" in strong[0]
    assert "说话别太正式" not in strong[0]
    assert len(advisory) == 1
    assert "说话别太正式" in advisory[0]
    assert "本群只用英文回复" not in advisory[0]


def test_per_group_strong_rule_isolation(monkeypatch):
    """A strong rule taught in group 1 must not appear (in any tier) in
    group 2's chat prompts."""
    handler, captured = _make_handler(monkeypatch)

    async def go():
        await handler.lessons.teach_raw(
            group_id=1, user_id=42, text="只用英文回复",
        )
        await handler.handle(parse_event(_event("早", group_id=2)))
        await handler.aclose()

    asyncio.run(go())
    chat_msgs = next(
        msgs for msgs in captured
        if any(m.role == "user" and m.content == "早" for m in msgs)
    )
    sys_text = "\n".join(m.content for m in chat_msgs if m.role == "system")
    assert "只用英文回复" not in sys_text, (
        "group 1's strong rule leaked into group 2's system prompt"
    )
