"""
Verdict builder: constructs stub / no-op GuardrailResponse objects.

Day 1: every layer downstream calls build_stub_verdict() to get a
fully schema-valid response with no real detections.  Days 2-5 will
replace stub calls with real detector output by filling in the
CategoryResult objects before passing to combine_verdicts().

Key rules (from CRITICAL constraints):
- Tier 1 (pii, secrets) and Tier 2 (harmful_content, prompt_injection)
  are INDEPENDENT.  Both always run unless the category is disabled.
  There is NO cascade skip based on Tier 1 results.
- BLOCK always beats REDACT when combining category actions.
- When the combined action is REDACT, sanitized_text must be set.
"""

from __future__ import annotations

import time
import uuid

from guardrail.policy import PolicyConfig, PolicyAction
from guardrail.schema import (
    Action,
    CategoryResult,
    Direction,
    ExecutionState,
    FilterMatchState,
    GuardrailResponse,
    InvocationResult,
    MatchState,
    SanitizationMetadata,
    SanitizationResult,
)


# ---------------------------------------------------------------------------
# Tier / category metadata (used by later layers to build CategoryResult)
# ---------------------------------------------------------------------------

#: Which tier owns each built-in category.  Custom categories added via
#: policy.yaml default to tier 2 (guard model) if not recognised here.
CATEGORY_TIER: dict[str, int] = {
    "pii": 1,
    "secrets": 1,
    "harmful_content": 2,
    "prompt_injection": 2,
}

CATEGORY_MODEL: dict[str, str | None] = {
    "pii": None,
    "secrets": None,
    "harmful_content": "qwen3guard-4b",
    "prompt_injection": "qwen3guard-4b",
}


# ---------------------------------------------------------------------------
# Stub builder (Day 1 — no real detection yet)
# ---------------------------------------------------------------------------

def build_stub_verdict(
    request_id: str,
    direction: Direction,
    policy: PolicyConfig,
    original_text: str,
    latency_ms: int = 0,
    cache_hit: bool = False,
) -> GuardrailResponse:
    """
    Return a fully schema-valid no-op verdict.

    Every enabled category appears in filterResults with:
    - execution_state = EXECUTION_SUCCESS
    - match_state = NO_MATCH_FOUND
    - detections = []

    Every disabled category appears with:
    - execution_state = NOT_EVALUATED
    - reason = "skipped: category disabled in policy"

    The combined action is always ALLOW and sanitized_text is None
    until real detection logic populates detections in Days 2-4.
    """
    filter_results: dict[str, CategoryResult] = {}

    for cat_name, cat_policy in policy.categories.items():
        tier = CATEGORY_TIER.get(cat_name, 2)
        model = CATEGORY_MODEL.get(cat_name)

        if not cat_policy.enabled:
            filter_results[cat_name] = CategoryResult(
                execution_state=ExecutionState.NOT_EVALUATED,
                match_state=MatchState.NA,
                tier=tier,
                model=model,
                reason="skipped: category disabled in policy",
            )
        else:
            filter_results[cat_name] = CategoryResult(
                execution_state=ExecutionState.EXECUTION_SUCCESS,
                match_state=MatchState.NO_MATCH_FOUND,
                tier=tier,
                model=model,
                detections=[],
            )

    return GuardrailResponse(
        request_id=request_id,
        direction=direction,
        sanitization_result=SanitizationResult(
            filter_match_state=FilterMatchState.NO_MATCH_FOUND,
            invocation_result=InvocationResult.SUCCESS,
            action=Action.ALLOW,
            latency_ms=latency_ms,
            sanitized_text=None,
            filter_results=filter_results,
            sanitization_metadata=SanitizationMetadata(
                cache_hit=cache_hit,
                policy_version=policy.policy_version,
                fallback_applied=False,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Verdict combiner (shell — filled in on Day 3/5 with real logic)
# ---------------------------------------------------------------------------

def combine_verdicts(
    input_verdict: GuardrailResponse,
    output_verdict: GuardrailResponse | None = None,
) -> GuardrailResponse:
    """
    Merge input + output verdicts into one final response.

    Day 1: returns input_verdict unchanged (output guardrail is Day 5).

    BLOCK always beats REDACT (enforced here when real detections arrive):
    - If ANY enabled category fires BLOCK → top-level action = BLOCK
    - If no BLOCK but ANY fires REDACT → top-level action = REDACT
    - Otherwise → ALLOW
    """
    # TODO (Day 5): merge output_verdict filter_results in
    return input_verdict


# ---------------------------------------------------------------------------
# Convenience: generate a request ID
# ---------------------------------------------------------------------------

def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"
