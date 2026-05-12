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
    rate_limit_per_min: int = _env_int("RATE_LIMIT_USER_PER_MIN", 15)
    max_reply_chars: int = _env_int("MAX_REPLY_CHARS", 1800)
    max_history_turns: int = _env_int("MAX_HISTORY_TURNS", 8)


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
    # Uploaded images are downscaled to fit within this box before being sent
    # to the vision API — cuts token cost dramatically (~free).
    max_vision_input_size: int = _env_int("MAX_VISION_INPUT_SIZE", 128)

    onebot_ws_url: str = os.getenv("ONEBOT_WS_URL", "ws://127.0.0.1:3001")
    onebot_access_token: str = os.getenv("ONEBOT_ACCESS_TOKEN", "")
    onebot_mode: str = os.getenv("ONEBOT_MODE", "forward").lower()  # forward | reverse
    onebot_reverse_host: str = os.getenv("ONEBOT_REVERSE_HOST", "0.0.0.0")
    onebot_reverse_port: int = _env_int("ONEBOT_REVERSE_PORT", 3001)
    onebot_reverse_path: str = os.getenv("ONEBOT_REVERSE_PATH", "/onebot/v11/ws")

    trigger_mode: str = os.getenv("TRIGGER_MODE", "always").lower()  # always | mention | prefix
    trigger_prefix: str = os.getenv("TRIGGER_PREFIX", "#")

    image_cache_ttl: int = _env_int("IMAGE_CACHE_TTL", 600)  # seconds

    # Daily report. Optional: a group id that receives a summary right before midnight.
    daily_report_group: int = _env_int("DAILY_REPORT_GROUP", 0)
    # Time of day to send the daily report, 24h "HH:MM" local time.
    daily_report_time: str = os.getenv("DAILY_REPORT_TIME", "23:55")

    # Streaming. When true, DeepSeek replies are streamed in chunks into QQ.
    stream_replies: bool = (os.getenv("STREAM_REPLIES", "1") not in ("0", "false", "False"))
    stream_flush_chars: int = _env_int("STREAM_FLUSH_CHARS", 220)

    allowed_groups: Set[int] = field(default_factory=lambda: _csv_ints(os.getenv("ALLOWED_GROUPS")))
    superusers: Set[int] = field(default_factory=lambda: _csv_ints(os.getenv("SUPERUSERS")))

    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    limits: Limits = field(default_factory=Limits)


CONFIG = Config()

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
