"""Central configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
IMAGE_DIR = DATA_DIR / "images"
MEMORY_FILE = DATA_DIR / "memory.json"   # legacy; migrated on first start
QUOTA_FILE = DATA_DIR / "quota.json"     # legacy; migrated on first start
DB_FILE = DATA_DIR / "state.db"

load_dotenv(ROOT_DIR / ".env")


def _csv_ints(value: str | None) -> Set[int]:
    if not value:
        return set()
    out: Set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            out.add(int(chunk))
    return out


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Limits:
    openai_text_group: int = _env_int("DAILY_OPENAI_TEXT_GROUP", 20)
    openai_text_user: int = _env_int("DAILY_OPENAI_TEXT_USER", 3)
    openai_image_group: int = _env_int("DAILY_OPENAI_IMAGE_GROUP", 10)
    openai_image_user: int = _env_int("DAILY_OPENAI_IMAGE_USER", 2)
    openai_image_edit_group: int = _env_int("DAILY_OPENAI_IMAGE_EDIT_GROUP", 5)
    openai_image_edit_user: int = _env_int("DAILY_OPENAI_IMAGE_EDIT_USER", 1)
    openai_vision_group: int = _env_int("DAILY_OPENAI_VISION_GROUP", 20)
    openai_vision_user: int = _env_int("DAILY_OPENAI_VISION_USER", 5)
    # Hard cap for auto-captioning images that float through the group (only
    # consulted when AUTO_VISION_GROUP_IMAGES=1).
    auto_vision_group: int = _env_int("AUTO_VISION_DAILY_MAX", 100)
    rate_limit_per_min: int = _env_int("RATE_LIMIT_USER_PER_MIN", 15)
    max_reply_chars: int = _env_int("MAX_REPLY_CHARS", 1800)
    max_history_turns: int = _env_int("MAX_HISTORY_TURNS", 64)


@dataclass
class Config:
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_chat_model: str = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    deepseek_reasoner_model: str = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_text_model: str = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
    openai_vision_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    # gpt-image-1-mini + low quality is the cheapest path at $0.005/image
    # (vs $0.016 for dall-e-2 256x256). Min size is 1024x1024.
    openai_image_model: str = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")
    openai_image_size: str = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
    # gpt-image-* only. Accepted: low | medium | high | auto. Ignored by dall-e-*.
    openai_image_quality: str = os.getenv("OPENAI_IMAGE_QUALITY", "low")
    # Uploaded images are downscaled to fit within this box (px). 512 fits one
    # OpenAI "low detail" tile — flat 85 tokens (~$0.000013) regardless of
    # input size. Going smaller doesn't save tokens AND breaks recognition
    # (model refuses with "I can't view images" below ~256px).
    max_vision_input_size: int = _env_int("MAX_VISION_INPUT_SIZE", 512)

    onebot_ws_url: str = os.getenv("ONEBOT_WS_URL", "ws://127.0.0.1:3001")
    onebot_access_token: str = os.getenv("ONEBOT_ACCESS_TOKEN", "")
    onebot_mode: str = os.getenv("ONEBOT_MODE", "forward").lower()  # forward | reverse
    onebot_reverse_host: str = os.getenv("ONEBOT_REVERSE_HOST", "0.0.0.0")
    onebot_reverse_port: int = _env_int("ONEBOT_REVERSE_PORT", 3001)
    onebot_reverse_path: str = os.getenv("ONEBOT_REVERSE_PATH", "/onebot/v11/ws")

    trigger_mode: str = os.getenv("TRIGGER_MODE", "always").lower()  # always | mention | prefix
    trigger_prefix: str = os.getenv("TRIGGER_PREFIX", "#")
    # After the bot replies in a group, silently ignore further non-command
    # messages in that group for this many seconds. 0 disables.
    reply_cooldown_seconds: int = _env_int("REPLY_COOLDOWN_SECONDS", 0)

    image_cache_ttl: int = _env_int("IMAGE_CACHE_TTL", 600)  # seconds

    # Daily report. Optional: a group id that receives a summary right before midnight.
    daily_report_group: int = _env_int("DAILY_REPORT_GROUP", 0)
    # Time of day to send the daily report, 24h "HH:MM" local time.
    daily_report_time: str = os.getenv("DAILY_REPORT_TIME", "23:55")

    # ----- Long-term memory (daily recap of every allowed group) -----
    daily_recap_enabled: bool = (
        os.getenv("DAILY_RECAP_ENABLED", "1") not in ("0", "false", "False")
    )
    # When the daily-recap task fires (local time). Default early morning so
    # yesterday's data is complete and the group is quiet.
    daily_recap_time: str = os.getenv("DAILY_RECAP_TIME", "03:30")
    # Recaps older than this are pruned.
    daily_recap_keep_days: int = _env_int("DAILY_RECAP_KEEP_DAYS", 365)
    # How many recent daily recaps to inject as "long memory" into each chat
    # call. 0 disables long-memory injection (but recaps are still saved).
    long_memory_inject_days: int = _env_int("LONG_MEMORY_INJECT_DAYS", 5)

    # Streaming. When true, DeepSeek replies are streamed in chunks into QQ.
    stream_replies: bool = (os.getenv("STREAM_REPLIES", "1") not in ("0", "false", "False"))
    stream_flush_chars: int = _env_int("STREAM_FLUSH_CHARS", 220)

    allowed_groups: Set[int] = field(default_factory=lambda: _csv_ints(os.getenv("ALLOWED_GROUPS")))
    superusers: Set[int] = field(default_factory=lambda: _csv_ints(os.getenv("SUPERUSERS")))

    # The bot's QQ nickname. Used by the router to decide if a message is
    # "directed at the bot" (in addition to @-mentions).
    bot_nickname: str = os.getenv("BOT_NICKNAME", "小笨蛋")

    # Optional path to a custom persona file. If empty/missing, the default
    # persona in bot/persona.py is used. The file can use {nickname} which
    # will be replaced with bot_nickname.
    bot_persona_file: str = os.getenv("BOT_PERSONA_FILE", "")

    # ----- Group-wide memory (lets the bot "hear" the whole group) -----
    # Max rows kept per group (oldest pruned).
    group_memory_max: int = _env_int("GROUP_MEMORY_MAX", 1000)
    # How many recent group messages to inject into each chat call as context.
    group_context_turns: int = _env_int("GROUP_CONTEXT_TURNS", 40)

    # ----- Human-pacing of replies -----
    # Split each bot reply into N short messages with a random delay between
    # each. Set human_send_enabled=0 to disable and revert to instant one-shot.
    human_send_enabled: bool = (
        os.getenv("HUMAN_SEND_ENABLED", "1") not in ("0", "false", "False")
    )
    human_send_max_chunks: int = _env_int("HUMAN_SEND_MAX_CHUNKS", 3)
    human_send_delay_min: float = float(os.getenv("HUMAN_SEND_DELAY_MIN", "0.6"))
    human_send_delay_max: float = float(os.getenv("HUMAN_SEND_DELAY_MAX", "2.0"))

    # ----- Emoji post-filter on bot output -----
    # After the LLM replies, every emoji has this probability of SURVIVING.
    # 0.10 = strip ~90% of emojis the model produces. Lower = more aggressive.
    emoji_keep_probability: float = float(os.getenv("EMOJI_KEEP_PROBABILITY", "0.10"))

    # ----- Auto-vision for group images -----
    # When ON, every image that floats through the group gets a quick caption
    # via OpenAI vision and the caption is stored in group_memory. Costs about
    # $0.000028 per image; daily cap (limits.auto_vision_group) prevents
    # runaway spend.
    auto_vision_group_images: bool = (
        os.getenv("AUTO_VISION_GROUP_IMAGES", "0") not in ("0", "false", "False")
    )

    # ----- Important-memory layer (LLM-judged facts / reminders / decisions) -----
    important_memory_enabled: bool = (
        os.getenv("IMPORTANT_MEMORY_ENABLED", "1") not in ("0", "false", "False")
    )
    # Top-N memories injected into each chat turn's system prompt. 0 disables
    # context-injection (reminders still fire).
    important_memory_recall_limit: int = _env_int("IMPORTANT_MEMORY_RECALL_LIMIT", 6)
    # Reminder loop tick (seconds). 30 = up-to-30s late on a "9:00pm" reminder.
    reminder_tick_seconds: int = _env_int("REMINDER_TICK_SECONDS", 30)
    # Combined maintenance loop tick (seconds). Runs daily-recap refresh +
    # memories dedup + expiry on every tick.
    maintenance_tick_seconds: int = _env_int("MAINTENANCE_TICK_SECONDS", 1800)

    # ----- Proactive interjection -----
    # When a message is NOT addressed to the bot, the bot may occasionally
    # decide on its own to chime in. Default to OFF until tuned.
    proactive_enabled: bool = (
        os.getenv("PROACTIVE_ENABLED", "1") not in ("0", "false", "False")
    )
    # Probability (0..1) each non-addressed message is even considered.
    # The actual decision is then made by an LLM judge that mostly says skip.
    proactive_probability: float = float(os.getenv("PROACTIVE_PROBABILITY", "0.08"))
    # Minimum seconds since the bot last spoke in this group before another
    # proactive interjection is allowed.
    proactive_min_seconds: int = _env_int("PROACTIVE_MIN_SECONDS", 90)
    # Minimum non-bot messages observed in the group since the bot last spoke.
    # Prevents firing back-to-back with a normal reply.
    proactive_min_new_messages: int = _env_int("PROACTIVE_MIN_NEW_MESSAGES", 3)

    # ----- Ambient engagement (post-router throttle for unaddressed chat) -----
    # When the router approves a NON-addressed message (no @, no nickname),
    # roll this probability before actually replying. 1.0 disables the gate.
    ambient_reply_probability: float = float(os.getenv("AMBIENT_REPLY_PROBABILITY", "0.1"))
    # Per-group minimum seconds between unaddressed replies. Directly-addressed
    # messages (@ / nickname) bypass this entirely.
    ambient_reply_min_seconds: int = _env_int("AMBIENT_REPLY_MIN_SECONDS", 60)

    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    limits: Limits = field(default_factory=Limits)


CONFIG = Config()

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
