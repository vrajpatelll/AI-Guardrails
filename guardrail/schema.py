"""
Pydantic v2 models for the Guardrail request/response schema.
Mirrors the shape defined in output-schema.md exactly.

Key design decisions:
- sanitizedText is a top-level field on the response (not buried in
  filterResults) so the SDK wrapper can use it directly as the text
  to forward/return when action=REDACT.
- policyVersion is carried on every response so callers can detect
  a policy reload without polling.
- filterResults is a dict[str, CategoryResult] keyed by category name
  (e.g. "pii", "secrets", "harmful_content", "prompt_injection") so
  new categories added in policy.yaml appear automatically without
  schema changes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class FilterMatchState(str, Enum):
    MATCH_FOUND = "MATCH_FOUND"
    NO_MATCH_FOUND = "NO_MATCH_FOUND"


class InvocationResult(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    TIMEOUT_FALLBACK = "TIMEOUT_FALLBACK"


class Action(str, Enum):
    BLOCK = "BLOCK"
    ALLOW = "ALLOW"
    REDACT = "REDACT"


class ExecutionState(str, Enum):
    EXECUTION_SUCCESS = "EXECUTION_SUCCESS"
    NOT_EVALUATED = "NOT_EVALUATED"    # category disabled in policy
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    TIMEOUT = "TIMEOUT"


class MatchState(str, Enum):
    MATCH_FOUND = "MATCH_FOUND"
    NO_MATCH_FOUND = "NO_MATCH_FOUND"
    NA = "N/A"


# ---------------------------------------------------------------------------
# Detection / per-category models
# ---------------------------------------------------------------------------

class Detection(BaseModel):
    """A single matched entity inside a category."""
    category: str                          # e.g. "EMAIL_ADDRESS", "AWS_KEY"
    matched_span: tuple[int, int]          # [start, end] char offsets in normalized text
    rule: str                              # e.g. "presidio.email_regex"
    confidence: float = Field(ge=0.0, le=1.0)
    redacted: bool = False                 # True when action=REDACT was applied


class CategoryResult(BaseModel):
    """Result for one policy category (pii, secrets, harmful_content, etc.)."""
    execution_state: ExecutionState
    match_state: MatchState
    tier: int                              # 1 = Presidio/deterministic, 2 = Qwen3Guard
    model: str | None = None               # e.g. "qwen3guard-4b" when tier=2
    detections: list[Detection] = Field(default_factory=list)
    reason: str | None = None              # human-readable skip/error reason


# ---------------------------------------------------------------------------
# Sanitization metadata
# ---------------------------------------------------------------------------

class SanitizationMetadata(BaseModel):
    cache_hit: bool = False
    policy_version: int = 0               # increments on every policy.yaml reload
    fallback_applied: bool = False         # True when latency budget was exceeded
    error_code: str | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Top-level sanitization result (mirrors output-schema.md)
# ---------------------------------------------------------------------------

class SanitizationResult(BaseModel):
    filter_match_state: FilterMatchState
    invocation_result: InvocationResult
    action: Action
    latency_ms: int = 0

    # Populated when action=REDACT; the SDK wrapper forwards/returns this
    # text instead of the original.  None when action=BLOCK or ALLOW.
    sanitized_text: str | None = None

    filter_results: dict[str, CategoryResult] = Field(default_factory=dict)
    sanitization_metadata: SanitizationMetadata = Field(
        default_factory=SanitizationMetadata
    )


# ---------------------------------------------------------------------------
# Request / Response wrappers (the guardrail HTTP contract)
# ---------------------------------------------------------------------------

class GuardrailRequest(BaseModel):
    """
    Guardrail-internal request envelope.
    `data.text` is the text to evaluate (after the SDK extracts it from
    the provider's messages array).
    """
    request_id: str
    direction: Direction
    policy_template: str = "default-strict"  # maps to a named policy in policy.yaml
    data: dict[str, Any]                     # {"text": "<prompt or response text>"}


class GuardrailResponse(BaseModel):
    """Full guardrail response returned to the SDK wrapper / HTTP caller."""
    request_id: str
    direction: Direction
    sanitization_result: SanitizationResult
