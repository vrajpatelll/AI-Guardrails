"""
Agentic tool-call guardrail: intercepts a tool call an LLM wants to make and
routes it to ALLOW / BLOCK / HUMAN_APPROVAL before anything executes.

Multi-layer evaluation, in order:
  1. Registry lookup — unknown tool name -> BLOCK immediately.
  2. Schema validation (Pydantic) — explicit types, min/max lengths. A
     malformed call never reaches a content-specific check.
  3. PII/secrets inspection — applies to every tool's string parameters.
  4. Tool-specific checks — SQL safety for execute_db_query, SSRF/domain
     allowlist for fetch_external_url, shell-injection for
     run_system_command (guardrail/tools/checks.py).

Precedence, the same shape as the text guardrail's BLOCK > REDACT > ALLOW
(guardrail/verdict.py:determine_action): any "block"-severity finding wins
outright, even for a tool that would otherwise just need human approval —
a SQL-injection payload doesn't get a "maybe a human says yes" chance.
Short of that, a "review"-severity finding or a tool flagged
always_requires_approval routes to HUMAN_APPROVAL. A clean call auto-allows.

evaluate_tool_call() never raises for bad input — a validation failure or
unknown tool name comes back as an ordinary BLOCK verdict with an
explanatory Finding, the same fail-safe posture as the rest of this
guardrail.
"""

from __future__ import annotations

import logging
from typing import Callable

from pydantic import BaseModel, ValidationError

from guardrail.tools.checks import (
    check_command_injection,
    check_pii,
    check_sql_safety,
    check_ssrf,
)
from guardrail.tools.registry import TOOL_REGISTRY, ToolSpec
from guardrail.tools.schema import Finding, RiskTier, ToolAction, ToolCallVerdict
from guardrail.verdict import new_request_id

logger = logging.getLogger("guardrail.tools")

# Dispatch table: tool name -> (attribute to read the relevant string param
# from, check function). Keeps evaluate_tool_call() from hardcoding an
# if/elif chain that grows with every new tool.
_TOOL_SPECIFIC_CHECKS: dict[str, tuple[str, Callable[[str], list[Finding]]]] = {
    "execute_db_query": ("query", check_sql_safety),
    "fetch_external_url": ("url", check_ssrf),
    "run_system_command": ("command", check_command_injection),
}


def _run_checks(spec: ToolSpec, params: BaseModel) -> list[Finding]:
    findings: list[Finding] = []

    for value in params.model_dump().values():
        if isinstance(value, str):
            findings.extend(check_pii(value))

    dispatch = _TOOL_SPECIFIC_CHECKS.get(spec.name)
    if dispatch is not None:
        attr, check_fn = dispatch
        findings.extend(check_fn(getattr(params, attr)))

    return findings


def evaluate_tool_call(tool_name: str, raw_parameters: dict) -> ToolCallVerdict:
    """Evaluate one LLM-proposed tool call and return the routing verdict."""
    request_id = new_request_id()
    spec = TOOL_REGISTRY.get(tool_name)

    if spec is None:
        logger.warning("[%s] unknown tool %r — BLOCK", request_id, tool_name)
        return ToolCallVerdict(
            request_id=request_id,
            tool_name=tool_name,
            parameters=raw_parameters,
            action=ToolAction.BLOCK,
            risk_tier=RiskTier.HIGH,
            findings=[Finding(
                check="schema", severity="block",
                message=f"Unknown tool '{tool_name}' is not in the registry.",
            )],
        )

    try:
        params = spec.param_model.model_validate(raw_parameters)
    except ValidationError as exc:
        logger.warning("[%s] tool=%s schema validation failed — BLOCK", request_id, tool_name)
        return ToolCallVerdict(
            request_id=request_id,
            tool_name=tool_name,
            parameters=raw_parameters,
            action=ToolAction.BLOCK,
            risk_tier=spec.risk_tier,
            findings=[Finding(check="schema", severity="block", message=str(exc))],
        )

    findings = _run_checks(spec, params)
    has_block = any(f.severity == "block" for f in findings)
    has_review = any(f.severity == "review" for f in findings)

    if has_block:
        action = ToolAction.BLOCK
    elif has_review or spec.always_requires_approval:
        action = ToolAction.HUMAN_APPROVAL
    else:
        action = ToolAction.ALLOW

    logger.info(
        "[%s] tool=%s action=%s findings=%d",
        request_id, tool_name, action.value, len(findings),
    )

    return ToolCallVerdict(
        request_id=request_id,
        tool_name=tool_name,
        parameters=params.model_dump(),
        action=action,
        risk_tier=spec.risk_tier,
        findings=findings,
    )
