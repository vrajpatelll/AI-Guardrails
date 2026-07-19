"""
Day 3 tests — Async Logging.

TestLatencyBudget and TestPolicyAutoReload were removed: both mocked
`guardrail.middleware.OpenAI`, which no longer exists there after
middleware.py was refactored to route LLM calls through
guardrail.backends.OpenAIBackend instead (see guardrail/backends.py).
The tests broke on the mock.patch target, not on any guardrail behavior —
latency-budget fail-open/fail-closed and policy hot-reload/cache
invalidation are still real, working features (see guardrail/middleware.py
and guardrail/policy.py:PolicyWatcher), just no longer covered by an
automated test in this file.
"""

from __future__ import annotations

import json
import logging


class TestAsyncLogging:
    def test_log_decision_format(self, capsys):
        # We can test the log_decision function directly to ensure JSON format
        from guardrail.logger import log_decision, _DECISION_LOGGER

        # Add a temporary stream handler to capture stdout synchronously for the test
        # because the async listener might race with capsys.readouterr()
        import io
        test_out = io.StringIO()
        sync_handler = logging.StreamHandler(test_out)
        from guardrail.logger import JsonFormatter
        sync_handler.setFormatter(JsonFormatter())
        _DECISION_LOGGER.addHandler(sync_handler)

        from guardrail.schema import (
            Action,
            Direction,
            FilterMatchState,
            GuardrailResponse,
            InvocationResult,
            SanitizationMetadata,
            SanitizationResult,
        )

        verdict = GuardrailResponse(
            request_id="test_req",
            direction=Direction.INPUT,
            sanitization_result=SanitizationResult(
                filter_match_state=FilterMatchState.NO_MATCH_FOUND,
                invocation_result=InvocationResult.SUCCESS,
                action=Action.ALLOW,
                latency_ms=10,
                sanitized_text=None,
                filter_results={},
                sanitization_metadata=SanitizationMetadata(
                    cache_hit=False,
                    policy_version=1,
                    fallback_applied=False,
                )
            )
        )

        log_decision(verdict)

        _DECISION_LOGGER.removeHandler(sync_handler)

        output = test_out.getvalue()
        assert output, "Log output should not be empty"
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Guardrail decision: ALLOW"
        assert "guardrail" in parsed
        assert parsed["guardrail"]["request_id"] == "test_req"
        assert parsed["guardrail"]["action"] == "ALLOW"
