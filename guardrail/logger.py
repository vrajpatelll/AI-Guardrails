"""
Async structured logger for guardrail decisions.

Uses Python's `logging.handlers.QueueHandler` and `QueueListener` to push
log formatting and I/O to a background thread, ensuring that writing logs
never blocks the hot request path.

Logs are emitted as JSON lines to standard output, making them easily
ingestible by Datadog, ELK, CloudWatch, etc.
"""

from __future__ import annotations

import atexit
import json
import logging
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import Any

from guardrail.schema import GuardrailResponse


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        # Build the structured payload
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Include any extra kwargs passed to the logger
        if hasattr(record, "structured_data"):
            payload.update(record.structured_data)  # type: ignore

        # Include exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def _setup_async_logger() -> logging.Logger:
    """Initialize the async logger and background listener."""
    logger = logging.getLogger("guardrail.decisions")
    
    # Only set up once
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't pass up to root logger

    # Unbounded queue for log events
    log_queue: Queue[logging.LogRecord] = Queue()

    # The QueueHandler pushes records to the queue (runs in calling thread)
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # The StreamHandler writes to stdout (runs in background thread)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JsonFormatter())

    # The QueueListener pops from queue and passes to StreamHandler
    listener = QueueListener(log_queue, stream_handler, respect_handler_level=True)
    listener.start()

    # Ensure the listener flushes on exit
    atexit.register(listener.stop)

    return logger


_DECISION_LOGGER = _setup_async_logger()


def log_decision(verdict: GuardrailResponse, error: Exception | None = None) -> None:
    """
    Log a guardrail decision asynchronously.

    Args:
        verdict: The final GuardrailResponse.
        error: Optional exception if the pipeline failed.
    """
    sr = verdict.sanitization_result
    
    # Extract category hits for easier querying in the log backend
    hits = [
        cat for cat, res in sr.filter_results.items()
        if res.match_state.value == "MATCH_FOUND"
    ]

    structured_data: dict[str, Any] = {
        "guardrail": {
            "request_id": verdict.request_id,
            "direction": verdict.direction.value,
            "action": sr.action.value,
            "latency_ms": sr.latency_ms,
            "cache_hit": sr.sanitization_metadata.cache_hit,
            "policy_version": sr.sanitization_metadata.policy_version,
            "fallback_applied": sr.sanitization_metadata.fallback_applied,
            "hits": hits,
        }
    }

    if error:
        _DECISION_LOGGER.error(
            "Guardrail pipeline error",
            extra={"structured_data": structured_data},
            exc_info=error,
        )
    else:
        # Use log level WARN for BLOCK, INFO for others, just as an example
        level = logging.WARNING if sr.action.value == "BLOCK" else logging.INFO
        _DECISION_LOGGER.log(
            level,
            f"Guardrail decision: {sr.action.value}",
            extra={"structured_data": structured_data},
        )
