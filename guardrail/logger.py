"""
Async structured logger for guardrail decisions.

Uses Python's `logging.handlers.QueueHandler` and `QueueListener` to push
log formatting and I/O to a background thread, ensuring that writing logs
never blocks the hot request path.

Logs are emitted as JSON lines to standard output (for Datadog, ELK,
CloudWatch, etc.) AND appended to a local JSONL file so the dashboard
(guardrail/dashboard.py) has a durable, greppable data source without
standing up a separate log pipeline. Set GUARDRAIL_LOG_FILE to change the
path; set it to an empty string to disable the file sink entirely.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import Queue
from typing import Any

from guardrail.schema import GuardrailResponse

#: Default location for the JSONL decision log consumed by the dashboard.
DEFAULT_LOG_FILE = "logs/decisions.jsonl"


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
    handlers: list[logging.Handler] = [stream_handler]

    # The FileHandler appends the same JSON lines to a local file, so the
    # dashboard can tail/aggregate decisions without a separate log pipeline.
    log_file = os.environ.get("GUARDRAIL_LOG_FILE", DEFAULT_LOG_FILE)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)

    # The QueueListener pops from queue and fans out to every handler
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()

    # Ensure the listener flushes on exit
    atexit.register(listener.stop)

    return logger


_DECISION_LOGGER = _setup_async_logger()


def build_decision_payload(verdict: GuardrailResponse) -> dict[str, Any]:
    """
    Build the JSON-serialisable "guardrail" payload for one decision.

    Shared by log_decision() (live traffic, stamped with wall-clock time by
    the logging framework) and scripts/generate_sample_data.py (which writes
    the same shape directly to the JSONL file with backdated timestamps, to
    seed the dashboard with a realistic-looking history).

    NEVER includes raw request/response text — only category names,
    confidence scores, and (for Tier 2) the model's short natural-language
    reason. Tier 1 detections (pii, secrets) never carry raw matched
    values here, only counts/confidence, so a secret can never leak into
    the decision log itself.
    """
    sr = verdict.sanitization_result

    # Extract category hits for easier querying in the log backend
    hits = [
        cat for cat, res in sr.filter_results.items()
        if res.match_state.value == "MATCH_FOUND"
    ]

    # Per-category detail (confidence/count/reason) — powers the dashboard's
    # category breakdown and "high risk prompts" views without ever
    # surfacing the matched text itself.
    categories: dict[str, Any] = {}
    risk_score = 0.0
    for cat_name, res in sr.filter_results.items():
        confidences = [d.confidence for d in res.detections]
        max_confidence = max(confidences, default=0.0)
        if res.match_state.value == "MATCH_FOUND":
            risk_score = max(risk_score, max_confidence)
        entry: dict[str, Any] = {
            "match": res.match_state.value == "MATCH_FOUND",
            "confidence": round(max_confidence, 4),
            "count": len(res.detections),
        }
        # Tier 2 rules carry a short model-generated reason (e.g.
        # "qwen2.5-0.5b: jailbreak attempt to bypass safety guidelines") —
        # safe to log since it describes the *classification*, not the text.
        if res.tier == 2 and res.detections:
            entry["reason"] = res.detections[0].rule
        categories[cat_name] = entry

    return {
        "request_id": verdict.request_id,
        "direction": verdict.direction.value,
        "action": sr.action.value,
        "latency_ms": sr.latency_ms,
        "cache_hit": sr.sanitization_metadata.cache_hit,
        "policy_version": sr.sanitization_metadata.policy_version,
        "fallback_applied": sr.sanitization_metadata.fallback_applied,
        "hits": hits,
        "categories": categories,
        "risk_score": round(risk_score, 4),
    }


def log_decision(verdict: GuardrailResponse, error: Exception | None = None) -> None:
    """
    Log a guardrail decision asynchronously.

    Args:
        verdict: The final GuardrailResponse.
        error: Optional exception if the pipeline failed.
    """
    sr = verdict.sanitization_result
    structured_data: dict[str, Any] = {"guardrail": build_decision_payload(verdict)}

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
