"""
Verdict builder: constructs GuardrailResponse objects from detector output.

Day 1: build_stub_verdict() — fully schema-valid no-op (still used in tests).
Day 2: assemble_tier1_verdict() — real CategoryResult from Tier1Results.
       determine_action() — BLOCK > REDACT > ALLOW per policy.
Day 4: assemble_verdict() — both tiers concurrent, BLOCK > REDACT > ALLOW.

Key rules (from CRITICAL constraints):
- Tier 1 (pii, secrets) and Tier 2 (harmful_content, prompt_injection)
  are INDEPENDENT. Both always run unless the category is disabled.
  There is NO cascade skip based on Tier 1 results.
- BLOCK always beats REDACT when combining category actions.
- When the combined action is REDACT, sanitized_text must be set.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from guardrail.policy import PolicyConfig, PolicyAction
from guardrail.schema import (
    Action,
    CategoryResult,
    Detection,
    Direction,
    ExecutionState,
    FilterMatchState,
    GuardrailResponse,
    InvocationResult,
    MatchState,
    SanitizationMetadata,
    SanitizationResult,
)

if TYPE_CHECKING:
    from guardrail.detectors.tier1 import DetectionResult, Tier1Results
    from guardrail.detectors.tier2 import Tier2Results


# ---------------------------------------------------------------------------
# Tier / category metadata
# ---------------------------------------------------------------------------

#: Which tier owns each built-in category.
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
# Action determination
# ---------------------------------------------------------------------------

def determine_action(
    filter_results: dict[str, CategoryResult],
    policy: PolicyConfig,
) -> Action:
    """
    Walk all CategoryResult entries and return the strongest action.

    Precedence: BLOCK > REDACT > ALLOW.
    Only considers categories that have match_state=MATCH_FOUND and
    execution_state=EXECUTION_SUCCESS.
    """
    has_block = False
    has_redact = False

    for cat_name, result in filter_results.items():
        if result.execution_state != ExecutionState.EXECUTION_SUCCESS:
            continue
        if result.match_state != MatchState.MATCH_FOUND:
            continue

        cat_policy = policy.categories.get(cat_name)
        if cat_policy is None or not cat_policy.enabled:
            continue

        if cat_policy.action == PolicyAction.BLOCK:
            has_block = True
        elif cat_policy.action == PolicyAction.REDACT:
            has_redact = True

    if has_block:
        return Action.BLOCK
    if has_redact:
        return Action.REDACT
    return Action.ALLOW


# ---------------------------------------------------------------------------
# Tier 1 verdict assembler (Day 2)
# ---------------------------------------------------------------------------

def _detection_results_to_schema(
    detections: list["DetectionResult"],
    threshold: float,
    action_is_redact: bool,
) -> list[Detection]:
    """
    Convert internal DetectionResult objects to schema Detection objects,
    applying confidence threshold filtering.
    """
    out: list[Detection] = []
    for d in detections:
        if d.confidence < threshold:
            continue
        out.append(Detection(
            category=d.category,
            matched_span=(d.start, d.end),
            rule=d.rule,
            confidence=d.confidence,
            redacted=action_is_redact,
        ))
    return out


def assemble_tier1_verdict(
    request_id: str,
    direction: Direction,
    policy: PolicyConfig,
    tier1_results: "Tier1Results",
    normalised_text: str,
    latency_ms: int = 0,
    cache_hit: bool = False,
) -> tuple[GuardrailResponse, list["DetectionResult"]]:
    """
    Build a GuardrailResponse from Tier 1 detector output.

    Returns:
        (verdict, redactable_detections)

        redactable_detections: all detections whose category action = REDACT
        (used by the caller to build sanitized_text via build_redacted_text).

    Tier 2 categories (harmful_content, prompt_injection) are set to
    NOT_EVALUATED here — they are wired in on Day 4.
    """
    # --- Build per-category results ----------------------------------------

    filter_results: dict[str, CategoryResult] = {}

    # PII
    pii_policy = policy.categories.get("pii")
    if pii_policy and pii_policy.enabled:
        # We'll determine redacted flag after we know the action
        pii_threshold = pii_policy.confidence_threshold
        pii_hits_above_threshold = [
            d for d in tier1_results.pii if d.confidence >= pii_threshold
        ]
        pii_match = bool(pii_hits_above_threshold)
        filter_results["pii"] = CategoryResult(
            execution_state=ExecutionState.EXECUTION_SUCCESS,
            match_state=MatchState.MATCH_FOUND if pii_match else MatchState.NO_MATCH_FOUND,
            tier=1,
            model=None,
            detections=[],  # filled in after action is determined
        )
    elif pii_policy and not pii_policy.enabled:
        filter_results["pii"] = CategoryResult(
            execution_state=ExecutionState.NOT_EVALUATED,
            match_state=MatchState.NA,
            tier=1,
            reason="skipped: category disabled in policy",
        )

    # Secrets
    secrets_policy = policy.categories.get("secrets")
    if secrets_policy and secrets_policy.enabled:
        secrets_threshold = secrets_policy.confidence_threshold
        secrets_hits_above_threshold = [
            d for d in tier1_results.secrets if d.confidence >= secrets_threshold
        ]
        secrets_match = bool(secrets_hits_above_threshold)
        filter_results["secrets"] = CategoryResult(
            execution_state=ExecutionState.EXECUTION_SUCCESS,
            match_state=MatchState.MATCH_FOUND if secrets_match else MatchState.NO_MATCH_FOUND,
            tier=1,
            model=None,
            detections=[],  # filled in after action is determined
        )
    elif secrets_policy and not secrets_policy.enabled:
        filter_results["secrets"] = CategoryResult(
            execution_state=ExecutionState.NOT_EVALUATED,
            match_state=MatchState.NA,
            tier=1,
            reason="skipped: category disabled in policy",
        )

    # Tier 2 categories — stub NOT_EVALUATED until Day 4
    for cat_name in ("harmful_content", "prompt_injection"):
        cat_policy = policy.categories.get(cat_name)
        if cat_policy and not cat_policy.enabled:
            filter_results[cat_name] = CategoryResult(
                execution_state=ExecutionState.NOT_EVALUATED,
                match_state=MatchState.NA,
                tier=2,
                model=CATEGORY_MODEL.get(cat_name),
                reason="skipped: category disabled in policy",
            )
        else:
            filter_results[cat_name] = CategoryResult(
                execution_state=ExecutionState.NOT_EVALUATED,
                match_state=MatchState.NA,
                tier=2,
                model=CATEGORY_MODEL.get(cat_name),
                reason="skipped: tier 2 not yet wired (Day 4)",
            )

    # --- Determine combined action -----------------------------------------
    action = determine_action(filter_results, policy)

    # --- Now populate Detection objects with correct redacted flag ----------
    redactable_detections: list[DetectionResult] = []

    if "pii" in filter_results and filter_results["pii"].execution_state == ExecutionState.EXECUTION_SUCCESS:
        pii_pol = policy.categories.get("pii")
        threshold = pii_pol.confidence_threshold if pii_pol else 0.0
        pii_action_is_redact = (pii_pol.action == PolicyAction.REDACT) if pii_pol else False
        pii_above = [d for d in tier1_results.pii if d.confidence >= threshold]
        filter_results["pii"].detections = _detection_results_to_schema(
            pii_above, threshold=0.0, action_is_redact=pii_action_is_redact
        )
        if pii_action_is_redact and action != Action.BLOCK:
            redactable_detections.extend(pii_above)

    if "secrets" in filter_results and filter_results["secrets"].execution_state == ExecutionState.EXECUTION_SUCCESS:
        sec_pol = policy.categories.get("secrets")
        threshold = sec_pol.confidence_threshold if sec_pol else 0.0
        sec_action_is_redact = (sec_pol.action == PolicyAction.REDACT) if sec_pol else False
        sec_above = [d for d in tier1_results.secrets if d.confidence >= threshold]
        filter_results["secrets"].detections = _detection_results_to_schema(
            sec_above, threshold=0.0, action_is_redact=sec_action_is_redact
        )
        if sec_action_is_redact and action != Action.BLOCK:
            redactable_detections.extend(sec_above)

    # --- Top-level match state --------------------------------------------
    any_match = any(
        r.match_state == MatchState.MATCH_FOUND
        for r in filter_results.values()
        if r.execution_state == ExecutionState.EXECUTION_SUCCESS
    )

    return GuardrailResponse(
        request_id=request_id,
        direction=direction,
        sanitization_result=SanitizationResult(
            filter_match_state=(
                FilterMatchState.MATCH_FOUND if any_match
                else FilterMatchState.NO_MATCH_FOUND
            ),
            invocation_result=InvocationResult.SUCCESS,
            action=action,
            latency_ms=latency_ms,
            sanitized_text=None,   # caller sets this after calling build_redacted_text
            filter_results=filter_results,
            sanitization_metadata=SanitizationMetadata(
                cache_hit=cache_hit,
                policy_version=policy.policy_version,
                fallback_applied=False,
            ),
        ),
    ), redactable_detections


# ---------------------------------------------------------------------------
# Combined verdict assembler — Tier 1 + Tier 2 (Day 4)
# ---------------------------------------------------------------------------

def assemble_verdict(
    request_id: str,
    direction: Direction,
    policy: PolicyConfig,
    tier1_results: "Tier1Results",
    tier2_results: "Tier2Results",
    normalised_text: str,
    latency_ms: int = 0,
    cache_hit: bool = False,
) -> tuple[GuardrailResponse, list["DetectionResult"]]:
    """
    Build a GuardrailResponse from BOTH tier detectors running concurrently.

    Tier 1 (pii, secrets) and Tier 2 (harmful_content, prompt_injection)
    are INDEPENDENT.  Both always run unless the category is disabled in
    policy.yaml.  There is NO cascade skip based on Tier 1 results.

    Returns:
        (verdict, redactable_detections)
    """
    filter_results: dict[str, CategoryResult] = {}

    # Tier 1: PII
    pii_policy = policy.categories.get("pii")
    if pii_policy and pii_policy.enabled:
        threshold = pii_policy.confidence_threshold
        pii_hits = [d for d in tier1_results.pii if d.confidence >= threshold]
        filter_results["pii"] = CategoryResult(
            execution_state=ExecutionState.EXECUTION_SUCCESS,
            match_state=MatchState.MATCH_FOUND if pii_hits else MatchState.NO_MATCH_FOUND,
            tier=1, model=None, detections=[],
        )
    elif pii_policy:
        filter_results["pii"] = CategoryResult(
            execution_state=ExecutionState.NOT_EVALUATED,
            match_state=MatchState.NA, tier=1,
            reason="skipped: category disabled in policy",
        )

    # Tier 1: Secrets
    sec_policy = policy.categories.get("secrets")
    if sec_policy and sec_policy.enabled:
        threshold = sec_policy.confidence_threshold
        sec_hits = [d for d in tier1_results.secrets if d.confidence >= threshold]
        filter_results["secrets"] = CategoryResult(
            execution_state=ExecutionState.EXECUTION_SUCCESS,
            match_state=MatchState.MATCH_FOUND if sec_hits else MatchState.NO_MATCH_FOUND,
            tier=1, model=None, detections=[],
        )
    elif sec_policy:
        filter_results["secrets"] = CategoryResult(
            execution_state=ExecutionState.NOT_EVALUATED,
            match_state=MatchState.NA, tier=1,
            reason="skipped: category disabled in policy",
        )

    # Tier 2: Harmful Content
    hc_policy = policy.categories.get("harmful_content")
    if hc_policy and hc_policy.enabled:
        threshold = hc_policy.confidence_threshold
        hc_hits = [d for d in tier2_results.harmful if d.confidence >= threshold]
        filter_results["harmful_content"] = CategoryResult(
            execution_state=ExecutionState.EXECUTION_SUCCESS,
            match_state=MatchState.MATCH_FOUND if hc_hits else MatchState.NO_MATCH_FOUND,
            tier=2,
            model=CATEGORY_MODEL.get("harmful_content"),
            detections=[Detection(
                category=d.category, matched_span=(d.start, d.end),
                rule=d.rule, confidence=d.confidence, redacted=False,
            ) for d in hc_hits],
        )
    elif hc_policy:
        filter_results["harmful_content"] = CategoryResult(
            execution_state=ExecutionState.NOT_EVALUATED,
            match_state=MatchState.NA, tier=2,
            model=CATEGORY_MODEL.get("harmful_content"),
            reason="skipped: category disabled in policy",
        )

    # Tier 2: Prompt Injection
    pi_policy = policy.categories.get("prompt_injection")
    if pi_policy and pi_policy.enabled:
        threshold = pi_policy.confidence_threshold
        pi_hits = [d for d in tier2_results.injection if d.confidence >= threshold]
        filter_results["prompt_injection"] = CategoryResult(
            execution_state=ExecutionState.EXECUTION_SUCCESS,
            match_state=MatchState.MATCH_FOUND if pi_hits else MatchState.NO_MATCH_FOUND,
            tier=2,
            model=CATEGORY_MODEL.get("prompt_injection"),
            detections=[Detection(
                category=d.category, matched_span=(d.start, d.end),
                rule=d.rule, confidence=d.confidence, redacted=False,
            ) for d in pi_hits],
        )
    elif pi_policy:
        filter_results["prompt_injection"] = CategoryResult(
            execution_state=ExecutionState.NOT_EVALUATED,
            match_state=MatchState.NA, tier=2,
            model=CATEGORY_MODEL.get("prompt_injection"),
            reason="skipped: category disabled in policy",
        )

    # Combined action
    action = determine_action(filter_results, policy)

    # Populate Tier 1 Detection objects (need final action for redacted flag)
    redactable_detections: list["DetectionResult"] = []

    if "pii" in filter_results and filter_results["pii"].execution_state == ExecutionState.EXECUTION_SUCCESS:
        pii_pol = policy.categories.get("pii")
        threshold = pii_pol.confidence_threshold if pii_pol else 0.0
        pii_action_is_redact = (pii_pol.action == PolicyAction.REDACT) if pii_pol else False
        pii_above = [d for d in tier1_results.pii if d.confidence >= threshold]
        filter_results["pii"].detections = _detection_results_to_schema(
            pii_above, threshold=0.0, action_is_redact=pii_action_is_redact
        )
        if pii_action_is_redact and action != Action.BLOCK:
            redactable_detections.extend(pii_above)

    if "secrets" in filter_results and filter_results["secrets"].execution_state == ExecutionState.EXECUTION_SUCCESS:
        sec_pol = policy.categories.get("secrets")
        threshold = sec_pol.confidence_threshold if sec_pol else 0.0
        sec_action_is_redact = (sec_pol.action == PolicyAction.REDACT) if sec_pol else False
        sec_above = [d for d in tier1_results.secrets if d.confidence >= threshold]
        filter_results["secrets"].detections = _detection_results_to_schema(
            sec_above, threshold=0.0, action_is_redact=sec_action_is_redact
        )
        if sec_action_is_redact and action != Action.BLOCK:
            redactable_detections.extend(sec_above)

    # Top-level match state
    any_match = any(
        r.match_state == MatchState.MATCH_FOUND
        for r in filter_results.values()
        if r.execution_state == ExecutionState.EXECUTION_SUCCESS
    )

    return GuardrailResponse(
        request_id=request_id,
        direction=direction,
        sanitization_result=SanitizationResult(
            filter_match_state=(
                FilterMatchState.MATCH_FOUND if any_match
                else FilterMatchState.NO_MATCH_FOUND
            ),
            invocation_result=InvocationResult.SUCCESS,
            action=action,
            latency_ms=latency_ms,
            sanitized_text=None,
            filter_results=filter_results,
            sanitization_metadata=SanitizationMetadata(
                cache_hit=cache_hit,
                policy_version=policy.policy_version,
                fallback_applied=False,
            ),
        ),
    ), redactable_detections


# ---------------------------------------------------------------------------
# Stub builder (Day 1 — still used in tests and as fallback)
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
    Used in tests and as a fallback if Presidio is unavailable.
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
# Verdict combiner (shell — filled in on Day 5 with real logic)
# ---------------------------------------------------------------------------

def combine_verdicts(
    input_verdict: GuardrailResponse,
    output_verdict: GuardrailResponse | None = None,
) -> GuardrailResponse:
    """
    Merge input + output verdicts into one final response.

    Day 1/2: returns input_verdict unchanged (output guardrail is Day 5).
    BLOCK always beats REDACT (enforced by determine_action).
    """
    # TODO (Day 5): merge output_verdict filter_results
    return input_verdict


# ---------------------------------------------------------------------------
# Fallback / timeout builder (Day 3)
# ---------------------------------------------------------------------------

def build_timeout_verdict(
    request_id: str,
    direction: Direction,
    policy: PolicyConfig,
    latency_ms: int,
) -> GuardrailResponse:
    """
    Return a fallback verdict when the latency budget is exceeded.
    The action (ALLOW or BLOCK) depends on policy.on_timeout.
    """
    if policy.on_timeout.value == "fail_open":
        action = Action.ALLOW
    else:
        action = Action.BLOCK

    return GuardrailResponse(
        request_id=request_id,
        direction=direction,
        sanitization_result=SanitizationResult(
            filter_match_state=FilterMatchState.NO_MATCH_FOUND,
            invocation_result=InvocationResult.TIMEOUT_FALLBACK,
            action=action,
            latency_ms=latency_ms,
            sanitized_text=None,
            filter_results={},  # no detections
            sanitization_metadata=SanitizationMetadata(
                cache_hit=False,
                policy_version=policy.policy_version,
                fallback_applied=True,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Convenience: generate a request ID
# ---------------------------------------------------------------------------

def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"
