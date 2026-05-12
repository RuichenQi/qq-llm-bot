"""Logging helpers. Writes to stdout and a daily rotating file under data/logs."""
from __future__ import annotations

import logging
import re
from logging.handlers import TimedRotatingFileHandler

from config import CONFIG, LOG_DIR

_API_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")


class _RedactFilter(logging.Filter):
    """Replace any token that looks like an API key with sk-***."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _API_KEY_RE.sub("sk-***", record.msg)
        if record.args:
            try:
                record.args = tuple(
                    _API_KEY_RE.sub("sk-***", a) if isinstance(a, str) else a
                    for a in record.args
                )
            except Exception:
                pass
        return True


_INITIALIZED = False


def setup_logging() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    root = logging.getLogger()
    root.setLevel(CONFIG.log_level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    redact = _RedactFilter()

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    stream.addFilter(redact)
    root.addHandler(stream)

    file_handler = TimedRotatingFileHandler(
        LOG_DIR / "bot.log", when="midnight", backupCount=7, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(redact)
    root.addHandler(file_handler)

    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
