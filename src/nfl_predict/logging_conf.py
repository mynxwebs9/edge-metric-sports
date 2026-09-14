"""Structured logging setup, shared by every module (see docs/ARCHITECTURE.md#logging-conventions).

Usage::

    from nfl_predict.logging_conf import get_logger

    logger = get_logger(__name__)
    logger.info("ingested games", extra={"season": 2023, "count": 272})

Modules must not call ``logging.basicConfig()`` or construct their own handlers/formatters
— configuration happens once, here, the first time :func:`get_logger` is called anywhere in
the process.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_RESERVED_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

_configured = False


class ContextFormatter(logging.Formatter):
    """Formats a log line as ``timestamp level logger: message  {extra key=value pairs}``."""

    def format(self, record: logging.LogRecord) -> str:
        # Capture extras before super().format() mutates the record (it sets
        # record.message, and record.asctime if the format string uses %(asctime)s) —
        # otherwise those computed attributes get misread as caller-supplied extra context.
        context: dict[str, Any] = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS
        }
        base = super().format(record)
        if not context:
            return base
        rendered_context = " ".join(f"{key}={value!r}" for key, value in sorted(context.items()))
        return f"{base}  {rendered_context}"


def _configure(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        ContextFormatter(fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name``, configuring process-wide logging on first use.

    Reads the initial log level from ``nfl_predict.config.get_settings().log_level`` if
    the config module can be imported without error; falls back to ``INFO`` otherwise
    (e.g. during very early bootstrap before config is available) so logging never blocks
    on configuration being ready.
    """
    if not _configured:
        try:
            from nfl_predict.config import get_settings

            level = get_settings().log_level
        except Exception:
            level = "INFO"
        _configure(level=level)
    return logging.getLogger(name)
