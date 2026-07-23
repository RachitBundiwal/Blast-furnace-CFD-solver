"""
Structured logging configuration.

Produces JSON-formatted log records on STDOUT suitable for ingestion by
Cloud Logging, Datadog, ELK, etc.  Errors are logged **without** exposing
tracebacks by default (per Google SRE / OWASP guidance); a debug flag flips
this for development.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter compatible with Cloud Logging structured fields."""

    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge extra structured fields
        for key, value in record.__dict__.items():
            if key in {"args", "asctime", "created", "exc_info", "exc_text",
                       "filename", "funcName", "levelname", "levelno",
                       "lineno", "message", "module", "msecs", "msg",
                       "name", "pathname", "process", "processName",
                       "relativeCreated", "thread", "threadName",
                       "stack_info", "taskName"}:
                continue
            base[key] = value
        # Include traceback only in debug mode
        if self.debug and record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)


def configure_logging(level: str = "INFO", debug: bool = False) -> logging.Logger:
    """Initialise root logger and return a scoped solver logger."""
    root = logging.getLogger()
    # Reset to guarantee idempotence in notebook / test environments
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(debug=debug))
    root.addHandler(handler)
    root.setLevel(level.upper())
    return logging.getLogger("bf2d")
