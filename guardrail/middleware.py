"""
SDK middleware: wraps an Anthropic client and intercepts every
messages.create() call to run the guardrail pipeline.

Day 4 pipeline (both tiers, concurrent):
  1. Normalise text extracted from messages array.
  2. Check verdict cache (key = hash(norm_text + policy_version)).
  3. Submit Tier 1 (PII + secrets) AND Tier 2 (harmful_content + prompt_injection)
     concurrently to the thread pool executor.
  4. Wait for both futures within the latency budget; timeout → fallback verdict.
  5. Assemble combined CategoryResult objects via assemble_verdict().
  6. If REDACT: build sanitized_text via build_redacted_text().
  7. Cache verdict + sanitized_text.
  8. If BLOCK: raise GuardrailBlockedError (no LLM call).
  9. Forward original (ALLOW) or sanitized (REDACT) text to Anthropic.

CRITICAL: Tier 1 and Tier 2 check UNRELATED categories.
Tier 2 is NEVER skipped because Tier 1 was clean.
Both are always submitted. The only valid skip is category disabled in policy.yaml.

Streaming: rejected with NotImplementedError (v2 roadmap).
"""


from __future__ import annotations

import atexit
import time
import concurrent.futures
from typing import Any

import anthropic

from guardrail.cache import VerdictCache
from guardrail.config import GuardrailConfig
from guardrail.detectors.tier1 import Tier1Detector
from guardrail.detectors.tier2 import Tier2Detector
from guardrail.logger import log_decision
from guardrail.normalizer import normalise
from guardrail.policy import load_policy, PolicyConfig, PolicyWatcher
from guardrail.redactor import build_redacted_text
from guardrail.schema import Direction, GuardrailResponse
from guardrail.verdict import (
    assemble_verdict,
    assemble_tier1_verdict,
    build_stub_verdict,
    build_timeout_verdict,
    combine_verdicts,
    new_request_id,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GuardrailBlockedError(Exception):
    """Raised when the guardrail decision is BLOCK."""
    def __init__(self, verdict: GuardrailResponse) -> None:
        self.verdict = verdict
        super().__init__(
            f"[Guardrail] Request blocked. "
            f"request_id={verdict.request_id} "
            f"filter_match_state={verdict.sanitization_result.filter_match_state}"
        )


# ---------------------------------------------------------------------------
# Wrapped response
# ---------------------------------------------------------------------------

class GuardrailWrappedResponse:
    """
    Thin wrapper around the real Anthropic response that adds a
    .guardrail_verdict attribute so callers can inspect the full verdict.
    """
    def __init__(
        self,
        anthropic_response: anthropic.types.Message,
        verdict: GuardrailResponse,
    ) -> None:
        self._response = anthropic_response
        self.guardrail_verdict = verdict

    def __getattr__(self, name: str) -> Any:
        """Proxy every other attribute to the real Anthropic response."""
        return getattr(self._response, name)

    def __repr__(self) -> str:
        return (
            f"GuardrailWrappedResponse("
            f"id={self._response.id!r}, "
            f"action={self.guardrail_verdict.sanitization_result.action})"
        )


# ---------------------------------------------------------------------------
# Messages namespace (mirrors anthropic.resources.Messages)
# ---------------------------------------------------------------------------

class _GuardrailMessages:
    """Drop-in replacement for `client.messages` that intercepts .create()."""
    def __init__(self, middleware: "GuardrailMiddleware") -> None:
        self._mw = middleware

    def create(self, **kwargs: Any) -> GuardrailWrappedResponse:
        if kwargs.get("stream"):
            raise NotImplementedError(
                "[Guardrail] Streaming is a v2 feature. "
                "Please set stream=False for now."
            )
        return self._mw._run(kwargs)


# ---------------------------------------------------------------------------
# Main middleware class
# ---------------------------------------------------------------------------

class GuardrailMiddleware:
    """
    Wraps an Anthropic client with a two-tier guardrail pipeline.

    Args:
        config: GuardrailConfig loaded from environment / explicit params.
        policy: Optional pre-loaded PolicyConfig.  If omitted the policy
                is loaded from config.policy_path at construction time.
        cache_ttl_seconds: TTL for in-memory verdict cache entries.
        tier1: Optional pre-initialised Tier1Detector.  If omitted a new
               instance is created (expensive — Presidio loads spaCy on
               first instantiation, ~1-2s).

    Example::

        cfg = GuardrailConfig.from_env()
        client = GuardrailMiddleware(cfg)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": "My email is alice@example.com"}],
        )
        print(resp.guardrail_verdict.sanitization_result.action)     # REDACT
        print(resp.guardrail_verdict.sanitization_result.sanitized_text)
    """

    def __init__(
        self,
        config: GuardrailConfig,
        policy: PolicyConfig | None = None,
        cache_ttl_seconds: float = 300.0,
        tier1: Tier1Detector | None = None,
    ) -> None:
        self.config = config
        self.policy: PolicyConfig = policy or load_policy(config.policy_path)
        self._cache = VerdictCache(ttl_seconds=cache_ttl_seconds)
        self._anthropic = anthropic.Anthropic(
            api_key=config.llm_api_key.get_secret_value()
        )
        self._tier1: Tier1Detector = tier1 or Tier1Detector()
        self._tier2: Tier2Detector = Tier2Detector()  # lazy-loaded on first call
        self.messages = _GuardrailMessages(self)

        # Thread pool for latency budget
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="guardrail-detector"
        )
        
        # Policy watcher for hot reloading
        self._watcher = PolicyWatcher(self.config.policy_path, self.reload_policy)
        self._watcher.start()
        
        # Ensure clean shutdown
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Cleanly stop background threads."""
        self._watcher.stop()
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Policy reload
    # ------------------------------------------------------------------

    def reload_policy(self) -> None:
        """
        Reload policy.yaml from disk.
        Bumps policy_version so the cache key changes and stale verdicts
        are never served.
        """
        self.policy = load_policy(self.config.policy_path)

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text(self, messages: list[dict[str, Any]]) -> str:
        """
        Extract evaluable text from a messages array.

        Concatenates all user-role content in turn order, covering the
        full conversation surface for detection.  Handles both plain-string
        and content-block (vision API) shapes.
        """
        parts: list[str] = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block["text"])
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _run(self, call_kwargs: dict[str, Any]) -> GuardrailWrappedResponse:
        """
        Full guardrail pipeline for a single messages.create() call.

        Day 2 pipeline:
          1. Extract + normalise text.
          2. Cache lookup.
          3. Run Tier 1 (PII + secrets, always both).
          4. Assemble verdict + determine action.
          5. Build sanitized_text if action=REDACT.
          6. Cache verdict.
          7. BLOCK → raise. ALLOW/REDACT → forward to LLM.
        """
        t0 = time.monotonic()
        request_id = new_request_id()
        messages: list[dict[str, Any]] = call_kwargs.get("messages", [])

        # 1. Normalise
        raw_text = self._extract_text(messages)
        norm_text = normalise(raw_text)

        # 2. Cache lookup
        cache_key = self._cache.make_key(norm_text, self.policy.policy_version)
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached.sanitization_result.sanitization_metadata.cache_hit = True
            if cached.sanitization_result.action.value == "BLOCK":
                raise GuardrailBlockedError(cached)
            return self._forward_to_llm(call_kwargs, cached, messages)

        # 3. Submit Tier 1 + Tier 2 concurrently — BOTH always run
        #    (only valid skip is category disabled in policy.yaml)
        enabled_tier2_cats = [
            cat for cat in ("harmful_content", "prompt_injection")
            if (cp := self.policy.categories.get(cat)) and cp.enabled
        ]
        timeout_sec = self.policy.latency_budget_ms / 1000.0
        try:
            f1 = self._executor.submit(self._tier1.run, norm_text)
            f2 = self._executor.submit(self._tier2.run, norm_text, enabled_tier2_cats)
            # Wait for both — wall-clock cost = max(tier1, tier2) not sum
            tier1_results = f1.result(timeout=timeout_sec)
            # Remaining budget for tier2
            elapsed = time.monotonic() - t0
            remaining = max(0.0, timeout_sec - elapsed)
            tier2_results = f2.result(timeout=remaining + 0.5)  # small grace
        except concurrent.futures.TimeoutError:
            latency_ms = int((time.monotonic() - t0) * 1000)
            verdict = build_timeout_verdict(
                request_id=request_id,
                direction=Direction.INPUT,
                policy=self.policy,
                latency_ms=latency_ms,
            )
            log_decision(verdict)
            if verdict.sanitization_result.action.value == "BLOCK":
                raise GuardrailBlockedError(verdict)
            return self._forward_to_llm(call_kwargs, verdict, messages)

        # 4. Assemble combined verdict from both tiers
        latency_ms = int((time.monotonic() - t0) * 1000)
        verdict, redactable = assemble_verdict(
            request_id=request_id,
            direction=Direction.INPUT,
            policy=self.policy,
            tier1_results=tier1_results,
            tier2_results=tier2_results,
            normalised_text=norm_text,
            latency_ms=latency_ms,
            cache_hit=False,
        )

        # 5. Build sanitized_text for REDACT action
        if verdict.sanitization_result.action.value == "REDACT" and redactable:
            verdict.sanitization_result.sanitized_text = build_redacted_text(
                norm_text, redactable
            )

        # 6. Cache the verdict (includes sanitized_text if set)
        self._cache.set(cache_key, verdict)

        # Log decision asynchronously
        log_decision(verdict)

        # 7. Block?
        if verdict.sanitization_result.action.value == "BLOCK":
            raise GuardrailBlockedError(verdict)

        # 8. Forward to LLM
        return self._forward_to_llm(call_kwargs, verdict, messages)

    def _forward_to_llm(
        self,
        call_kwargs: dict[str, Any],
        verdict: GuardrailResponse,
        original_messages: list[dict[str, Any]],
    ) -> GuardrailWrappedResponse:
        """
        Forward the (possibly sanitized) request to Anthropic.

        If action=REDACT: replaces the last user message content with
        verdict.sanitization_result.sanitized_text so the LLM never sees
        the raw PII / secret.
        """
        forward_kwargs = dict(call_kwargs)

        sanitized_text = verdict.sanitization_result.sanitized_text
        if sanitized_text is not None:
            msgs = list(original_messages)
            for i in reversed(range(len(msgs))):
                if msgs[i].get("role") == "user":
                    msgs[i] = {**msgs[i], "content": sanitized_text}
                    break
            forward_kwargs["messages"] = msgs

        raw_response = self._anthropic.messages.create(**forward_kwargs)

        # TODO (Day 5): run output guardrail on raw_response.content here

        return GuardrailWrappedResponse(
            anthropic_response=raw_response,
            verdict=verdict,
        )
