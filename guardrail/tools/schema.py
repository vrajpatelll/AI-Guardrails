"""
Pydantic schema for the agentic tool-calling guardrail.

Deliberately separate from guardrail/schema.py (the text-scanning
guardrail's schema): a tool call is structured JSON arguments to a named
function, not free text, and the attack surface is different (SQL
injection, SSRF, command injection, schema violations) from what Tier 1/
Tier 2 look for (PII, secrets, harmful content, prompt injection in prose).

The three-outcome routing (ALLOW / BLOCK / HUMAN_APPROVAL) also doesn't map
onto the text guardrail's ALLOW/REDACT/BLOCK — there's no sensible way to
"redact" a tool call, and some tools (run_system_command) must pause for a
human based on risk tier alone, independent of whether the parameters look
clean.

Per-tool parameter models below enforce explicit types and bounds
(min/max length) via Pydantic — the first guardrail layer, before any
content-specific check runs (see guardrail/tools/checks.py).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolAction(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"


class RiskTier(str, Enum):
    LOW = "low"        # read-only, no side effects
    MEDIUM = "medium"   # side effects possible, but scoped
    HIGH = "high"        # state-changing or system-level


class Finding(BaseModel):
    """One guardrail check's result. Empty findings list = clean."""
    check: str                # e.g. "schema", "pii", "sql_safety", "ssrf", "command_injection"
    severity: str              # "block" | "review"
    message: str


class ToolCallVerdict(BaseModel):
    """Result of evaluate_tool_call() — what to do with one proposed call."""
    request_id: str
    tool_name: str
    parameters: dict[str, Any]
    action: ToolAction
    risk_tier: RiskTier
    findings: list[Finding] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-tool parameter models
# ---------------------------------------------------------------------------

class GetWeatherParams(BaseModel):
    location: str = Field(..., min_length=1, max_length=100)


class ExecuteDbQueryParams(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class FetchExternalUrlParams(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)
    method: str = Field(default="GET", pattern="^(GET|POST)$")


class RunSystemCommandParams(BaseModel):
    command: str = Field(..., min_length=1, max_length=500)
