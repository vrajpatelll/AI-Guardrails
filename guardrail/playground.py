"""
Interactive "try it" page — type a prompt, run it through the real
GuardrailMiddleware pipeline (Tier 1 + Tier 2, verdict cache, hot-reloaded
policy.yaml, redaction), and see the verdict in a browser instead of the
`examples/day3_live.py` CLI demo.

Mounted into guardrail/dashboard.py's FastAPI app, so the whole guardrail
web UI (log dashboard + live tester) runs from one command:
    uv run uvicorn guardrail.dashboard:app --reload --port 8050

Reuses GuardrailMiddleware as-is rather than re-implementing Tier 1/Tier 2
orchestration a third time (middleware.py and scripts/generate_sample_data.py
already do this) — this module is just the HTTP form + rendering around it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from html import escape
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from guardrail.config import GuardrailConfig
from guardrail.detectors.tier1 import DenyListDetector, DetectionResult
from guardrail.middleware import GuardrailBlockedError, GuardrailMiddleware
from guardrail.normalizer import normalise
from guardrail.schema import GuardrailResponse

logger = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "pii": "PII",
    "secrets": "Secrets",
    "harmful_content": "Harmful content",
    "prompt_injection": "Prompt injection",
}
ACTION_COLORS = {"ALLOW": "#0ca30c", "REDACT": "#fab219", "BLOCK": "#d03b3b"}


# ---------------------------------------------------------------------------
# Middleware lifecycle — built once at process startup (see dashboard.py's
# lifespan handler), not per-request: Tier 2's model load alone is ~15-20s.
# ---------------------------------------------------------------------------

_middleware: GuardrailMiddleware | None = None
_init_error: str | None = None
_deny_list = DenyListDetector()


def init_middleware() -> None:
    global _middleware, _init_error
    try:
        cfg = GuardrailConfig.from_env()
        _middleware = GuardrailMiddleware(cfg)
        logger.info("Playground: guardrail middleware ready (model=%s)", cfg.llm_model)
    except Exception as exc:
        _init_error = str(exc)
        logger.exception("Playground: failed to initialise guardrail middleware")


def shutdown_middleware() -> None:
    if _middleware is not None:
        _middleware.shutdown()


def setup_error() -> str | None:
    return _init_error


# ---------------------------------------------------------------------------
# Human-in-the-loop keyword gate
#
# Runs BEFORE the real Tier1+Tier2 pipeline and before any LLM call. Tier 2's
# semantic judgment on prompt_injection is only as good as a 0.5B model's
# instruction-following (see the tier2.py parsing fix — recall on real
# jailbreaks isn't something to rely on alone). A deterministic keyword match
# against known jailbreak phrasing routes the request to a human reviewer
# instead of an automatic decision: approve and it still goes through the
# full guardrail pipeline as normal; reject and nothing is sent anywhere.
#
# Keywords live in policy.yaml under categories.prompt_injection.deny_patterns
# (reusing the same DenyListDetector/deny_patterns mechanism Tier 1 uses for
# pii/secrets) so they hot-reload with the rest of the policy - no restart
# needed to add a new phrase.
# ---------------------------------------------------------------------------

def check_injection_keywords(text: str) -> list[DetectionResult]:
    if _middleware is None:
        return []
    cat_policy = _middleware.policy.categories.get("prompt_injection")
    if not cat_policy or not cat_policy.enabled or not cat_policy.deny_patterns:
        return []
    norm_text = normalise(text)
    return _deny_list.run(norm_text, "prompt_injection", cat_policy.deny_patterns)


# ---------------------------------------------------------------------------
# Running a prompt through the pipeline
# ---------------------------------------------------------------------------

@dataclass
class ScanOutcome:
    blocked: bool
    combined_action: str  # ALLOW | REDACT | BLOCK | ERROR (LLM call itself failed)
    input_verdict: GuardrailResponse | None
    output_verdict: GuardrailResponse | None
    llm_text: str | None
    llm_error: str | None
    latency_ms: float


def run_prompt(text: str) -> ScanOutcome:
    """Run `text` through the real middleware pipeline (blocking call)."""
    assert _middleware is not None, "call init_middleware() first"
    t0 = time.monotonic()

    try:
        resp = _middleware.messages.create(
            model=_middleware.config.llm_model,
            max_tokens=200,
            messages=[{"role": "user", "content": text}],
        )
    except GuardrailBlockedError as exc:
        return ScanOutcome(
            blocked=True,
            combined_action="BLOCK",
            input_verdict=exc.verdict,
            output_verdict=None,
            llm_text=None,
            llm_error=None,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
    except Exception as exc:
        # The guardrail scan itself didn't block — the LLM gateway call
        # failed (network/auth/reachability). Not a guardrail bug.
        logger.warning("Playground: LLM backend call failed: %s", exc)
        return ScanOutcome(
            blocked=False,
            combined_action="ERROR",
            input_verdict=None,
            output_verdict=None,
            llm_text=None,
            llm_error=str(exc),
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    llm_text: str | None
    try:
        llm_text = resp.choices[0].message.content
    except (AttributeError, IndexError):
        llm_text = None

    return ScanOutcome(
        blocked=False,
        combined_action=resp.guardrail_verdict.sanitization_result.action.value,
        input_verdict=resp.guardrail_input_verdict,
        output_verdict=resp.guardrail_output_verdict,
        llm_text=llm_text,
        llm_error=None,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _category_chips(filter_results: dict[str, Any]) -> str:
    chips = []
    for cat_name, cr in filter_results.items():
        is_output = cat_name.startswith("output_")
        base_name = cat_name.removeprefix("output_")
        label = CATEGORY_LABELS.get(base_name, base_name)
        if is_output:
            label = f"{label} (output)"

        if cr.match_state.value == "MATCH_FOUND":
            top = cr.detections[0] if cr.detections else None
            detail = f" {top.confidence:.2f}" if top else ""
            reason = f" &mdash; {escape(top.rule)}" if top and cr.tier == 2 and top.rule else ""
            chips.append(
                f'<span class="cat-chip cat-chip-hit">{escape(label)}<b>{detail}</b>{reason}</span>'
            )
        elif cr.execution_state.value == "NOT_EVALUATED":
            chips.append(f'<span class="cat-chip cat-chip-skip">{escape(label)} (skipped)</span>')
        else:
            chips.append(f'<span class="cat-chip cat-chip-clean">{escape(label)}</span>')
    return "".join(chips)


def _verdict_card(title: str, verdict: GuardrailResponse) -> str:
    sr = verdict.sanitization_result
    action = sr.action.value
    color = ACTION_COLORS.get(action, "#898781")
    chips = _category_chips(sr.filter_results)
    sanitized = ""
    if sr.sanitized_text:
        sanitized = (
            '<div class="sanitized-block"><div class="stat-label">Sanitized text</div>'
            f'<pre>{escape(sr.sanitized_text)}</pre></div>'
        )
    meta = sr.sanitization_metadata
    return (
        '<div class="verdict-card">'
        '<div class="verdict-header">'
        f'<span class="action-badge" style="background:{color}">{escape(action)}</span>'
        f'<span class="verdict-title">{escape(title)}</span>'
        f'<span class="verdict-meta">latency {sr.latency_ms}ms &middot; '
        f'{"cache HIT" if meta.cache_hit else "cache MISS"} &middot; '
        f'policy v{meta.policy_version}</span>'
        "</div>"
        f'<div class="chip-row">{chips}</div>'
        f"{sanitized}"
        "</div>"
    )


def render_approval_body(submitted_text: str, hits: list[DetectionResult]) -> str:
    keywords = sorted({h.category.removeprefix("DENY_LIST:") for h in hits})
    keyword_chips = "".join(
        f'<span class="cat-chip cat-chip-hit">{escape(k)}</span>' for k in keywords
    )
    return f"""
    <div class="banner banner-warn">
      <strong>Human review required.</strong> This prompt matches known prompt-injection
      phrasing (checked before the guardrail model even runs). Review it below and decide
      whether it should go to the LLM.
    </div>
    <div class="card">
      <h2>Matched keywords</h2>
      <div class="chip-row">{keyword_chips}</div>
    </div>
    <div class="card">
      <h2>Submitted prompt</h2>
      <pre>{escape(submitted_text)}</pre>
      <form method="post" style="margin-top:14px;">
        <input type="hidden" name="text" value="{escape(submitted_text)}">
        <div style="display:flex; gap:10px;">
          <button type="submit" formaction="/try/approve">Approve &amp; send to LLM</button>
          <button type="submit" formaction="/try/reject" style="background:var(--critical)">Reject</button>
        </div>
      </form>
    </div>
    """


def render_rejected_body(submitted_text: str) -> str:
    return f"""
    <div class="banner banner-error">
      <strong>Rejected.</strong> This prompt was not sent to the LLM.
    </div>
    <div class="card">
      <h2>Submitted prompt</h2>
      <pre>{escape(submitted_text)}</pre>
    </div>
    <div class="card">
      <h2>Try another prompt</h2>
      <form method="post" action="/try">
        <textarea name="text" rows="4" placeholder="Type a prompt to test&hellip;" required></textarea>
        <button type="submit">Run through guardrail</button>
      </form>
    </div>
    """


def render_try_body(
    submitted_text: str = "",
    outcome: ScanOutcome | None = None,
    approved_keywords: list[str] | None = None,
) -> str:
    banner = ""
    if _init_error:
        banner = (
            '<div class="banner banner-error">Guardrail middleware failed to start: '
            f'<code>{escape(_init_error)}</code>. Check your <code>.env</code> '
            "(<code>LLM_GATEWAY_API_KEY</code>, <code>GUARDRAIL_TOKEN</code>) and restart "
            "the server.</div>"
        )
    if approved_keywords:
        banner += (
            '<div class="banner banner-warn">Human-approved despite matching keyword(s): '
            f'<code>{escape(", ".join(approved_keywords))}</code> — sent on to the guardrail '
            "pipeline and LLM below.</div>"
        )

    result_html = ""
    if outcome is not None:
        if outcome.combined_action == "ERROR":
            result_html = (
                '<div class="banner banner-warn">The guardrail scan completed and did not '
                "block this request, but the LLM gateway call itself failed: "
                f'<code>{escape(outcome.llm_error or "")}</code>. That\'s usually a network/'
                "gateway reachability issue, not a guardrail bug.</div>"
            )
        else:
            cards = []
            if outcome.input_verdict is not None:
                cards.append(_verdict_card(
                    "Blocked on input" if outcome.blocked else "Input scan",
                    outcome.input_verdict,
                ))
            if outcome.output_verdict is not None:
                cards.append(_verdict_card("Output scan", outcome.output_verdict))

            response_html = ""
            if not outcome.blocked:
                if outcome.llm_text:
                    response_html = (
                        '<div class="card"><h2>LLM response</h2>'
                        f'<pre class="llm-response">{escape(outcome.llm_text)}</pre></div>'
                    )
                else:
                    response_html = (
                        '<div class="banner banner-warn">Request was not blocked, but no '
                        "response text could be read from the LLM backend.</div>"
                    )
            result_html = "".join(cards) + response_html

    return f"""
    <div class="card">
      <h2>Try a prompt</h2>
      <p class="subtitle">Runs the real guardrail pipeline (Tier 1 + Tier 2, verdict cache,
      policy.yaml hot-reload) against your text, then forwards it to the LLM gateway if it
      isn't blocked. The response is scanned too before it comes back.</p>
      <form method="post" action="/try">
        <textarea name="text" rows="4" placeholder="Type a prompt to test&hellip;" required>{escape(submitted_text)}</textarea>
        <button type="submit">Run through guardrail</button>
      </form>
    </div>
    {banner}
    {result_html}
    """
