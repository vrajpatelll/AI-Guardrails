"""
SDK middleware: wraps an Anthropic client and intercepts every
messages.create() call to run the guardrail pipeline.

Day 1 behaviour (pass-through):
  - Extracts the last user message text from the messages array.
  - Normalises it and checks the verdict cache.
  - Calls build_stub_verdict() → a no-op ALLOW verdict.
  - Forwards the *original* (or sanitized_text if REDACT) message to
    the real Anthropic API and returns the result.
  - Raises a structured GuardrailBlockedError on BLOCK.

Streaming: if stream=True is passed, the wrapper raises a clear
NotImplementedError (streaming is v2 roadmap).

Usage::

    import anthropic
    from guardrail.middleware import GuardrailMiddleware
    from guardrail.config import GuardrailConfig

    cfg = GuardrailConfig.from_env()
    client = GuardrailMiddleware(cfg)
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    # response.guardrail_verdict is the full GuardrailResponse
"""

from __future__ import annotations

import time
from typing import Any

import anthropic

from guardrail.cache import VerdictCache
from guardrail.config import GuardrailConfig
from guardrail.normalizer import normalise
from guardrail.policy import load_policy, PolicyConfig
from guardrail.schema import Direction, GuardrailResponse
from guardrail.verdict import build_stub_verdict, new_request_id, combine_verdicts


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
    """
    Drop-in replacement for `client.messages` that intercepts .create().
    """
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

    Example::

        cfg = GuardrailConfig.from_env()
        client = GuardrailMiddleware(cfg)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": "Write me a poem."}],
        )
        print(resp.content[0].text)
        print(resp.guardrail_verdict.model_dump_json(indent=2))
    """

    def __init__(
        self,
        config: GuardrailConfig,
        policy: PolicyConfig | None = None,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self.config = config
        self.policy: PolicyConfig = policy or load_policy(config.policy_path)
        self._cache = VerdictCache(ttl_seconds=cache_ttl_seconds)
        self._anthropic = anthropic.Anthropic(api_key=config.llm_api_key)
        self.messages = _GuardrailMessages(self)

    # ------------------------------------------------------------------
    # Policy reload (call this to pick up policy.yaml changes live)
    # ------------------------------------------------------------------

    def reload_policy(self) -> None:
        """
        Reload policy.yaml from disk.

        Bumps policy_version (via load_policy) so the cache key changes
        and stale verdicts are never served — no explicit cache flush needed.
        """
        self.policy = load_policy(self.config.policy_path)

    # ------------------------------------------------------------------
    # Internal pipeline (Day 1: stub only)
    # ------------------------------------------------------------------

    def _extract_text(self, messages: list[dict[str, Any]]) -> str:
        """
        Extract evaluable text from a messages array.

        Strategy: concatenate all user-role message content in order,
        separated by a newline.  This covers multi-turn context while
        giving detectors the full conversation surface to scan.

        Content can be a plain string or a list of content blocks
        (Anthropic's vision API shape); we extract only the text blocks.
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

    def _run(self, call_kwargs: dict[str, Any]) -> GuardrailWrappedResponse:
        """
        Full guardrail pipeline for a single messages.create() call.

        Day 1 pipeline:
          1. Extract + normalise text from messages array.
          2. Check cache.
          3. Build stub verdict (no real detection yet).
          4. Cache the verdict.
          5. If BLOCK → raise GuardrailBlockedError.
          6. Forward to Anthropic.  Use sanitized_text if action=REDACT.
          7. [Day 5] Run output guardrail on LLM response.
          8. Return GuardrailWrappedResponse.
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
            # Replay action from cache
            if cached.sanitization_result.action.value == "BLOCK":
                raise GuardrailBlockedError(cached)
            return self._forward_to_llm(call_kwargs, cached, messages, norm_text)

        # 3. Build stub verdict (Days 2-4 will replace this with real detectors)
        latency_ms = int((time.monotonic() - t0) * 1000)
        verdict = build_stub_verdict(
            request_id=request_id,
            direction=Direction.INPUT,
            policy=self.policy,
            original_text=norm_text,
            latency_ms=latency_ms,
            cache_hit=False,
        )

        # 4. Cache the verdict
        self._cache.set(cache_key, verdict)

        # 5. Block?
        if verdict.sanitization_result.action.value == "BLOCK":
            raise GuardrailBlockedError(verdict)

        # 6. Forward to LLM
        return self._forward_to_llm(call_kwargs, verdict, messages, norm_text)

    def _forward_to_llm(
        self,
        call_kwargs: dict[str, Any],
        verdict: GuardrailResponse,
        original_messages: list[dict[str, Any]],
        norm_text: str,
    ) -> GuardrailWrappedResponse:
        """
        Forward the (possibly sanitized) request to Anthropic and return
        a wrapped response.

        If action=REDACT, the last user message's content is replaced
        with verdict.sanitization_result.sanitized_text before forwarding
        so the LLM never sees raw PII.
        """
        forward_kwargs = dict(call_kwargs)

        sanitized_text = verdict.sanitization_result.sanitized_text
        if sanitized_text is not None:
            # Replace the last user message content with the sanitized text
            msgs = list(original_messages)
            for i in reversed(range(len(msgs))):
                if msgs[i].get("role") == "user":
                    msgs[i] = {**msgs[i], "content": sanitized_text}
                    break
            forward_kwargs["messages"] = msgs

        # Call the real Anthropic API
        raw_response = self._anthropic.messages.create(**forward_kwargs)

        # TODO (Day 5): run output guardrail on raw_response.content here

        return GuardrailWrappedResponse(
            anthropic_response=raw_response,
            verdict=verdict,
        )
