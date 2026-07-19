"""
SDK middleware: wraps an LLM backend (OpenAI-compatible gateway, LangChain
BaseChatModel, or any custom LLMBackend — see guardrail/backends.py) and
intercepts every messages.create() call to run the guardrail pipeline.

Pipeline (both tiers submitted concurrently, Tier 2 conditionally awaited):
  1. Normalise text extracted from messages array. If the last user message
     is entirely valid base64, decode it FIRST and replace the message
     content outright — otherwise Tier 1/Tier 2 only ever see ciphertext,
     and any PII/secret/prompt injection wrapped in base64 sails straight
     through. The decoded text is what gets scanned, cached, and forwarded
     to the LLM, on ALLOW as well as REDACT (an LLM can't use raw base64).
  2. Check verdict cache (key = hash(norm_text + policy_version)).
  3. Submit Tier 1 (PII + secrets) AND Tier 2 (harmful_content + prompt_injection)
     concurrently to the thread pool executor.
  4. Wait for Tier 1 within latency_budget_ms; timeout → fallback verdict
     (fail-open/closed per policy.on_timeout).
  5. If Tier 1 secrets already resolves to BLOCK, skip waiting on Tier 2
     entirely — nothing it finds could make the outcome stricter.
     Otherwise wait for Tier 2, capped at tier2_timeout_ms; exceeding that
     always fails CLOSED (BLOCK), since letting semantically-unscanned text
     through is worse than added latency.
  6. Assemble combined CategoryResult objects via assemble_verdict()
     (or assemble_tier1_verdict() when Tier 2 was skipped).
  7. If REDACT: build sanitized_text via build_redacted_text().
  8. Cache verdict + sanitized_text.
  9. If BLOCK: raise GuardrailBlockedError (no LLM call).
  10. Forward original (ALLOW) or sanitized (REDACT) text to the LLM gateway.

CRITICAL: Tier 1 and Tier 2 check UNRELATED categories.
The ONLY cascade skip is Tier 1 secrets forcing BLOCK — a Tier 1 REDACT
(pii) alone still waits for Tier 2. The only other valid skip is a
category disabled in policy.yaml.

Streaming: rejected with NotImplementedError (v2 roadmap).
"""


from __future__ import annotations

import atexit
import logging
import time
import concurrent.futures
from typing import Any

logger = logging.getLogger(__name__)

from guardrail.backends import LLMBackend, OpenAIBackend
from guardrail.cache import VerdictCache
from guardrail.config import GuardrailConfig
from guardrail.detectors.tier1 import Tier1Detector
from guardrail.detectors.tier2 import Tier2Detector
from guardrail.logger import log_decision
from guardrail.normalizer import normalise, try_decode_base64
from guardrail.policy import load_policy, PolicyAction, PolicyConfig, PolicyWatcher
from guardrail.redactor import build_redacted_text
from guardrail.schema import Action, Direction, GuardrailResponse
from guardrail.verdict import (
    assemble_verdict,
    assemble_tier1_verdict,
    build_stub_verdict,
    build_timeout_verdict,
    build_tier2_timeout_verdict,
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
    Thin wrapper around the real LLM gateway response that adds guardrail
    verdict attributes.

    .guardrail_verdict       — combined (input + output) verdict
    .guardrail_input_verdict — input-side verdict only
    .guardrail_output_verdict— output-side verdict (None if output guardrail skipped)
    """
    def __init__(
        self,
        llm_response: Any,
        verdict: GuardrailResponse,
        output_verdict: GuardrailResponse | None = None,
    ) -> None:
        self._response = llm_response
        self.guardrail_input_verdict = verdict
        self.guardrail_output_verdict = output_verdict
        # guardrail_verdict = combined view (strictest action wins)
        self.guardrail_verdict = verdict  # will be replaced by combine_verdicts below

    def __getattr__(self, name: str) -> Any:
        """Proxy every other attribute to the real LLM gateway response."""
        return getattr(self._response, name)

    def __repr__(self) -> str:
        return (
            f"GuardrailWrappedResponse("
            f"id={self._response.id!r}, "
            f"action={self.guardrail_verdict.sanitization_result.action})"
        )


# ---------------------------------------------------------------------------
# Messages namespace (Anthropic-Messages-shaped facade over the gateway)
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
    Wraps an LLM backend with a two-tier guardrail pipeline. Defaults to the
    OpenAI-compatible gateway from config; pass `backend=` to wrap LangChain
    or any other framework instead (see guardrail/backends.py).

    Args:
        config: GuardrailConfig loaded from environment / explicit params.
        policy: Optional pre-loaded PolicyConfig.  If omitted the policy
                is loaded from config.policy_path at construction time.
        cache_ttl_seconds: TTL for in-memory verdict cache entries.
        tier1: Optional pre-initialised Tier1Detector.  If omitted a new
               instance is created (expensive — Presidio loads spaCy on
               first instantiation, ~1-2s).
        tier2: Optional pre-initialised Tier2Detector.  If omitted a new
               instance is created and eagerly loaded (expensive — pulls the
               model into memory, ~18s on first-ever run before it's cached).
        backend: Optional LLMBackend. Defaults to OpenAIBackend built from
                 `config` (the OpenAI-compatible gateway). Pass
                 LangChainBackend(chat_model) to wrap a LangChain
                 BaseChatModel, or any custom LLMBackend implementation to
                 wrap another framework/SDK.

    Example (default OpenAI-compatible gateway)::

        cfg = GuardrailConfig.from_env()
        client = GuardrailMiddleware(cfg)
        resp = client.messages.create(
            model="Bedrock-ant-haiku-4-5-20251001-v1-0",
            max_tokens=512,
            messages=[{"role": "user", "content": "My email is alice@example.com"}],
        )
        print(resp.guardrail_verdict.sanitization_result.action)     # REDACT
        print(resp.guardrail_verdict.sanitization_result.sanitized_text)

    Example (LangChain)::

        from langchain_openai import ChatOpenAI
        from guardrail.backends import LangChainBackend

        cfg = GuardrailConfig.from_env()
        client = GuardrailMiddleware(cfg, backend=LangChainBackend(ChatOpenAI(model="gpt-4o-mini")))
        resp = client.messages.create(
            messages=[{"role": "user", "content": "My email is alice@example.com"}],
        )
    """

    def __init__(
        self,
        config: GuardrailConfig,
        policy: PolicyConfig | None = None,
        cache_ttl_seconds: float = 300.0,
        tier1: Tier1Detector | None = None,
        tier2: Tier2Detector | None = None,
        backend: LLMBackend | None = None,
    ) -> None:
        self.config = config
        self.policy: PolicyConfig = policy or load_policy(config.policy_path)
        self._cache = VerdictCache(ttl_seconds=cache_ttl_seconds)
        # Defaults to the OpenAI-compatible gateway from config (today's
        # behaviour). Pass backend=LangChainBackend(chat_model) or any other
        # LLMBackend implementation to wrap a different framework — see
        # guardrail/backends.py.
        self._backend: LLMBackend = backend or OpenAIBackend.from_config(config)
        logger.info("Initialising Tier 1 detector (Presidio)…")
        self._tier1: Tier1Detector = tier1 or Tier1Detector()
        # eager=True: pay the ~18s model load cost once at startup instead of
        # on whichever request happens to arrive first.
        if tier2 is None:
            logger.info(
                "Initialising Tier 2 detector eagerly — this loads the model "
                "now (first-ever run can take up to ~20s or more if it also "
                "has to download weights)…"
            )
        self._tier2: Tier2Detector = tier2 or Tier2Detector(eager=True)
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

    def _last_user_message(self, messages: list[dict[str, Any]]) -> tuple[int, str] | None:
        """
        Return (index, text) of the LAST user-role message, or None if there
        isn't one. Mirrors _extract_text's string/content-block handling but
        scoped to a single message (used for base64 detection, which should
        act on one message's content, not the whole joined conversation).
        """
        for i in reversed(range(len(messages))):
            if messages[i].get("role") != "user":
                continue
            content = messages[i].get("content", "")
            if isinstance(content, str):
                return i, content
            if isinstance(content, list):
                parts = [
                    block["text"] for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return i, "\n".join(parts)
            return i, ""
        return None

    @staticmethod
    def _replace_last_user_message(
        messages: list[dict[str, Any]], new_content: str
    ) -> list[dict[str, Any]]:
        """Return a copy of `messages` with the last user-role message's content replaced."""
        msgs = list(messages)
        for i in reversed(range(len(msgs))):
            if msgs[i].get("role") == "user":
                msgs[i] = {**msgs[i], "content": new_content}
                break
        return msgs

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _run(self, call_kwargs: dict[str, Any]) -> GuardrailWrappedResponse:
        """
        Full guardrail pipeline for a single messages.create() call.

        See module docstring for the full pipeline. Summary:
          1. Extract text; decode base64 if the last user message is
             entirely base64; normalise.
          2. Cache lookup.
          3. Run Tier 1 + Tier 2 (Tier 2 conditionally skipped/awaited).
          4. Assemble verdict + determine action.
          5. Build sanitized_text if action=REDACT.
          6. Cache verdict.
          7. BLOCK → raise. ALLOW/REDACT → forward to LLM.
        """
        t0 = time.monotonic()
        request_id = new_request_id()
        messages: list[dict[str, Any]] = call_kwargs.get("messages", [])
        logger.info("[%s] input guardrail: starting", request_id)

        # 1. Base64 evasion check + normalise.
        # If the LAST user message is entirely valid base64, decode it BEFORE
        # anything else: Tier 1/Tier 2 must scan what the text actually says,
        # not opaque ciphertext, or base64-wrapping any PII/secret/prompt
        # injection would sail straight through. The decoded text replaces
        # the message content outright — an LLM can't use raw base64 anyway,
        # so it becomes what's cached, scanned, redacted, and forwarded,
        # regardless of the final action (ALLOW included, not just REDACT).
        last_user = self._last_user_message(messages)
        if last_user is not None:
            _, last_user_text = last_user
            decoded = try_decode_base64(last_user_text)
            if decoded is not None:
                logger.info(
                    "[%s] last user message is base64 (%d chars) — decoded to "
                    "[%s], scanning decoded text instead",
                    request_id, len(decoded),decoded
                )
                messages = self._replace_last_user_message(messages, decoded)
                call_kwargs = {**call_kwargs, "messages": messages}

        raw_text = self._extract_text(messages)
        logger.info("[%s] extracted text is: %r", request_id, raw_text)
        norm_text = normalise(raw_text)
        logger.info("[%s] norm text is: %r", request_id, norm_text)


        # 2. Cache lookup
        cache_key = self._cache.make_key(norm_text, self.policy.policy_version)
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached.sanitization_result.sanitization_metadata.cache_hit = True
            logger.info(
                "[%s] cache HIT — action=%s",
                request_id, cached.sanitization_result.action.value,
            )
            if cached.sanitization_result.action.value == "BLOCK":
                raise GuardrailBlockedError(cached)
            return self._forward_to_llm(call_kwargs, cached, messages)
        logger.info("[%s] cache MISS", request_id)

        # 3. Submit Tier 1 + Tier 2 concurrently.
        enabled_tier2_cats = [
            cat for cat in ("harmful_content", "prompt_injection")
            if (cp := self.policy.categories.get(cat)) and cp.enabled
        ]
        timeout_sec = self.policy.latency_budget_ms / 1000.0
        logger.info(
            "[%s] submitting tier1 + tier2 (tier1_budget=%.0fms, tier2_categories=%s)",
            request_id, timeout_sec * 1000, enabled_tier2_cats,
        )
        f1 = self._executor.submit(self._tier1.run, norm_text, self.policy)
        f2 = self._executor.submit(self._tier2.run, norm_text, enabled_tier2_cats)

        # Tier 1 has its own latency budget — fail-open/closed per
        # policy.on_timeout, same as before.
        try:
            tier1_results = f1.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.warning(
                "[%s] tier1 TIMED OUT after %dms — falling back per on_timeout=%s",
                request_id, latency_ms, self.policy.on_timeout.value,
            )
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
        logger.info(
            "[%s] tier1 done: pii=%d secrets=%d",
            request_id, len(tier1_results.pii), len(tier1_results.secrets),
        )

        # 4. Cascade: if Tier 1 secrets already forces BLOCK, that's already
        #    the strictest possible action — skip waiting on Tier 2 entirely.
        #    (f2 keeps running in the background; its result is simply unused.)
        sec_policy = self.policy.categories.get("secrets")
        secrets_already_blocking = bool(
            sec_policy
            and sec_policy.enabled
            and sec_policy.action == PolicyAction.BLOCK
            and any(d.confidence >= sec_policy.confidence_threshold for d in tier1_results.secrets)
        )

        if secrets_already_blocking:
            logger.info(
                "[%s] tier1 secrets already BLOCK — skipping tier2 wait", request_id,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            verdict, redactable = assemble_tier1_verdict(
                request_id=request_id,
                direction=Direction.INPUT,
                policy=self.policy,
                tier1_results=tier1_results,
                normalised_text=norm_text,
                latency_ms=latency_ms,
                cache_hit=False,
                tier2_skip_reason="skipped: tier1 secrets already BLOCK",
            )
        else:
            # Not already blocking — wait for Tier 2's real verdict, capped
            # at tier2_timeout_ms. Exceeding it fails CLOSED (BLOCK).
            tier2_timeout_sec = self.policy.tier2_timeout_ms / 1000.0
            logger.info(
                "[%s] waiting for tier2 (timeout=%.1fs)…", request_id, tier2_timeout_sec,
            )
            try:
                tier2_results = f2.result(timeout=tier2_timeout_sec)
            except concurrent.futures.TimeoutError:
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.warning(
                    "[%s] tier2 TIMED OUT after %dms (budget=%dms) — failing CLOSED",
                    request_id, latency_ms, self.policy.tier2_timeout_ms,
                )
                verdict = build_tier2_timeout_verdict(
                    request_id=request_id,
                    direction=Direction.INPUT,
                    policy=self.policy,
                    tier1_results=tier1_results,
                    normalised_text=norm_text,
                    latency_ms=latency_ms,
                )
                log_decision(verdict)
                raise GuardrailBlockedError(verdict)
            logger.info(
                "[%s] tier2 done: harmful=%d injection=%d",
                request_id, len(tier2_results.harmful), len(tier2_results.injection),
            )

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
        logger.info(
            "[%s] input verdict: action=%s latency=%dms",
            request_id, verdict.sanitization_result.action.value, latency_ms,
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

    def _run_output_guardrail(
        self,
        response_text: str,
        input_verdict: GuardrailResponse,
    ) -> GuardrailResponse:
        """
        Run the full parallel T1+T2 guardrail pipeline on the LLM response.

        Uses the same policy, executor, and detectors as the input guardrail.
        The output verdict direction is set to OUTPUT.
        """
        t0 = time.monotonic()
        request_id = input_verdict.request_id  # same request, output direction
        norm_text = normalise(response_text)
        logger.info("[%s] output guardrail: starting", request_id)

        enabled_tier2_cats = [
            cat for cat in ("harmful_content", "prompt_injection")
            if (cp := self.policy.categories.get(cat)) and cp.enabled
        ]
        timeout_sec = self.policy.latency_budget_ms / 1000.0

        f1 = self._executor.submit(self._tier1.run, norm_text, self.policy)
        f2 = self._executor.submit(self._tier2.run, norm_text, enabled_tier2_cats)

        # Tier 1 has its own latency budget — fail-open/closed per
        # policy.on_timeout, same as the input path.
        try:
            tier1_results = f1.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.warning(
                "[%s] output tier1 TIMED OUT after %dms — falling back per on_timeout=%s",
                request_id, latency_ms, self.policy.on_timeout.value,
            )
            return build_timeout_verdict(
                request_id=request_id,
                direction=Direction.OUTPUT,
                policy=self.policy,
                latency_ms=latency_ms,
            )
        logger.info(
            "[%s] output tier1 done: pii=%d secrets=%d",
            request_id, len(tier1_results.pii), len(tier1_results.secrets),
        )

        # Wait for Tier 2's real verdict, capped at tier2_timeout_ms (same
        # generous budget as the input path) instead of whatever's left of
        # the small shared latency_budget_ms — the response must actually be
        # scanned before it's returned, not waved through on a fast timeout.
        tier2_timeout_sec = self.policy.tier2_timeout_ms / 1000.0
        logger.info(
            "[%s] waiting for output tier2 (timeout=%.1fs)…", request_id, tier2_timeout_sec,
        )
        try:
            tier2_results = f2.result(timeout=tier2_timeout_sec)
        except concurrent.futures.TimeoutError:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.warning(
                "[%s] output tier2 TIMED OUT after %dms (budget=%dms) — failing CLOSED",
                request_id, latency_ms, self.policy.tier2_timeout_ms,
            )
            return build_tier2_timeout_verdict(
                request_id=request_id,
                direction=Direction.OUTPUT,
                policy=self.policy,
                tier1_results=tier1_results,
                normalised_text=norm_text,
                latency_ms=latency_ms,
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "[%s] output tier2 done: harmful=%d injection=%d",
            request_id, len(tier2_results.harmful), len(tier2_results.injection),
        )
        output_verdict, redactable = assemble_verdict(
            request_id=request_id,
            direction=Direction.OUTPUT,
            policy=self.policy,
            tier1_results=tier1_results,
            tier2_results=tier2_results,
            normalised_text=norm_text,
            latency_ms=latency_ms,
        )

        # Apply redaction on output text if needed
        if output_verdict.sanitization_result.action == Action.REDACT and redactable:
            output_verdict.sanitization_result.sanitized_text = build_redacted_text(
                norm_text, redactable
            )

        logger.info(
            "[%s] output verdict: action=%s latency=%dms",
            request_id, output_verdict.sanitization_result.action.value, latency_ms,
        )
        log_decision(output_verdict)
        return output_verdict

    def _forward_to_llm(
        self,
        call_kwargs: dict[str, Any],
        verdict: GuardrailResponse,
        original_messages: list[dict[str, Any]],
    ) -> GuardrailWrappedResponse:
        """
        Forward the (possibly sanitized) request to the LLM backend, then run
        the output guardrail on the LLM response.

        Input side:
          - action=REDACT: replaces the last user message with sanitized_text.
          - action=ALLOW:  forwards unchanged.

        Output side (Day 5):
          - Runs T1+T2 on the LLM response text.
          - action=BLOCK:  raises GuardrailBlockedError (LLM response suppressed).
          - action=REDACT: replaces response content with sanitized_text.
          - action=ALLOW:  returns response unchanged.
        """
        forward_kwargs = dict(call_kwargs)

        sanitized_text = verdict.sanitization_result.sanitized_text
        if sanitized_text is not None:
            forward_kwargs["messages"] = self._replace_last_user_message(
                original_messages, sanitized_text
            )

        logger.info(
            "[%s] forwarding to LLM backend (model=%s)…",
            verdict.request_id, forward_kwargs.get("model"),
        )
        raw_response = self._backend.send(forward_kwargs)
        logger.info("[%s] LLM backend responded", verdict.request_id)

        # Output guardrail
        response_text = self._backend.extract_text(raw_response)
        output_verdict = self._run_output_guardrail(response_text, verdict)
        combined_verdict = combine_verdicts(verdict, output_verdict)

        if combined_verdict.sanitization_result.action.value == "BLOCK":
            raise GuardrailBlockedError(combined_verdict)

        # If output was REDACT, swap the response text
        out_sanitized = output_verdict.sanitization_result.sanitized_text
        if out_sanitized is not None:
            raw_response = self._backend.set_text(raw_response, out_sanitized)

        wrapped = GuardrailWrappedResponse(
            llm_response=raw_response,
            verdict=combined_verdict,
            output_verdict=output_verdict,
        )
        wrapped.guardrail_verdict = combined_verdict
        return wrapped
