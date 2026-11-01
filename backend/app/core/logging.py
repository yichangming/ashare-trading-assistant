"""Logging setup with best-effort secret redaction."""

from __future__ import annotations

import logging
import re

_SENSITIVE = re.compile(
    r"(?i)\b(token|password|passwd|secret|authorization|api[_-]?key)\b"
    r"(\s*[=:]\s*)([^\s,;]+)"
)


def redact(text: str) -> str:
    return _SENSITIVE.sub(lambda match: f"{match.group(1)}{match.group(2)}***", text)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - never break logging
            return True
        record.msg = redact(message)
        record.args = ()
        return True


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_ashare_redaction", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.addFilter(RedactionFilter())
    handler._ashare_redaction = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
