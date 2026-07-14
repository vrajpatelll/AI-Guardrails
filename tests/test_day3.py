"""
Day 3 tests — Caching, Auto-Reload, Async Logging, and Latency Budgets.
"""

from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from guardrail.config import GuardrailConfig
from guardrail.detectors.tier1 import Tier1Detector, Tier1Results
from guardrail.middleware import GuardrailMiddleware, GuardrailBlockedError
from guardrail.policy import OnTimeout, PolicyConfig, PolicyAction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic() -> MagicMock:
    with patch("guardrail.middleware.anthropic.Anthropic") as mock_class:
        mock_instance = MagicMock()
        mock_instance.messages.create.return_value = MagicMock(id="msg_123", content=[])
        mock_class.return_value = mock_instance
        yield mock_class


@pytest.fixture
def mock_tier1() -> MagicMock:
    """A Tier1Detector mock that returns clean by default."""
    mock = MagicMock(spec=Tier1Detector)
    mock.run.return_value = Tier1Results(pii=[], secrets=[])
    return mock


@pytest.fixture
def temp_policy(tmp_path: Path) -> Path:
    yaml_text = """
name: default-test
categories:
  pii:
    enabled: true
    action: redact
  secrets:
    enabled: true
    action: block
latency_budget_ms: 50
on_timeout: fail_open
"""
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    return p


@pytest.fixture
def config(temp_policy: Path) -> GuardrailConfig:
    return GuardrailConfig(
        llm_api_key=SecretStr("sk-test"),
        guardrail_token=SecretStr("token-test"),
        policy_path=str(temp_policy)
    )


# ---------------------------------------------------------------------------
# 1. Latency Budget Tests
# ---------------------------------------------------------------------------

class TestLatencyBudget:
    def test_fail_open_on_timeout(self, config: GuardrailConfig, mock_anthropic: MagicMock, mock_tier1: MagicMock):
        # Configure tier1 to sleep longer than the 50ms budget
        def slow_run(*args, **kwargs):
            time.sleep(0.1)
            return Tier1Results(pii=[], secrets=[])
        mock_tier1.run.side_effect = slow_run
        
        middleware = GuardrailMiddleware(config, tier1=mock_tier1)
        
        # Should not raise exception (fail_open = ALLOW)
        response = middleware.messages.create(
            model="test",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        verdict = response.guardrail_verdict
        sr = verdict.sanitization_result
        assert sr.action.value == "ALLOW"
        assert sr.sanitization_metadata.fallback_applied is True
        
        middleware.shutdown()

    def test_fail_closed_on_timeout(self, config: GuardrailConfig, mock_anthropic: MagicMock, mock_tier1: MagicMock, temp_policy: Path):
        # Update policy to fail_closed
        yaml_text = temp_policy.read_text().replace("fail_open", "fail_closed")
        temp_policy.write_text(yaml_text)
        
        def slow_run(*args, **kwargs):
            time.sleep(0.1)
            return Tier1Results(pii=[], secrets=[])
        mock_tier1.run.side_effect = slow_run
        
        middleware = GuardrailMiddleware(config, tier1=mock_tier1)
        
        # Should raise GuardrailBlockedError (fail_closed = BLOCK)
        with pytest.raises(GuardrailBlockedError) as exc:
            middleware.messages.create(
                model="test",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
        verdict = exc.value.verdict
        sr = verdict.sanitization_result
        assert sr.action.value == "BLOCK"
        assert sr.sanitization_metadata.fallback_applied is True
        
        middleware.shutdown()


# ---------------------------------------------------------------------------
# 2. Policy Auto-Reload & Caching Tests
# ---------------------------------------------------------------------------

class TestPolicyAutoReload:
    def test_policy_reload_invalidates_cache(self, config: GuardrailConfig, mock_anthropic: MagicMock, mock_tier1: MagicMock, temp_policy: Path):
        middleware = GuardrailMiddleware(config, tier1=mock_tier1)
        
        # Request 1: Cache Miss
        resp1 = middleware.messages.create(
            model="test",
            messages=[{"role": "user", "content": "Cache me"}]
        )
        v1 = resp1.guardrail_verdict
        assert v1.sanitization_result.sanitization_metadata.cache_hit is False
        v1_policy_ver = v1.sanitization_result.sanitization_metadata.policy_version
        
        # Request 2: Cache Hit
        resp2 = middleware.messages.create(
            model="test",
            messages=[{"role": "user", "content": "Cache me"}]
        )
        v2 = resp2.guardrail_verdict
        assert v2.sanitization_result.sanitization_metadata.cache_hit is True
        assert v2.sanitization_result.sanitization_metadata.policy_version == v1_policy_ver
        
        # Modify policy to trigger reload
        old_yaml = temp_policy.read_text()
        new_yaml = old_yaml.replace("latency_budget_ms: 50", "latency_budget_ms: 100")
        temp_policy.write_text(new_yaml)
        
        # Wait for watchdog to pick up the change
        time.sleep(0.5)
        
        # Request 3: Cache Miss (because policy_version changed)
        resp3 = middleware.messages.create(
            model="test",
            messages=[{"role": "user", "content": "Cache me"}]
        )
        v3 = resp3.guardrail_verdict
        assert v3.sanitization_result.sanitization_metadata.cache_hit is False
        v3_policy_ver = v3.sanitization_result.sanitization_metadata.policy_version
        assert v3_policy_ver > v1_policy_ver
        
        middleware.shutdown()


# ---------------------------------------------------------------------------
# 3. Async Logging Test
# ---------------------------------------------------------------------------

class TestAsyncLogging:
    def test_log_decision_format(self, capsys):
        # We can test the log_decision function directly to ensure JSON format
        from guardrail.logger import log_decision, _DECISION_LOGGER
        from guardrail.schema import GuardrailResponse, SanitizationResult, Action, FilterMatchState, InvocationResult, SanitizationMetadata
        
        # Add a temporary stream handler to capture stdout synchronously for the test
        # because the async listener might race with capsys.readouterr()
        import io
        test_out = io.StringIO()
        sync_handler = logging.StreamHandler(test_out)
        from guardrail.logger import JsonFormatter
        sync_handler.setFormatter(JsonFormatter())
        _DECISION_LOGGER.addHandler(sync_handler)
        
        verdict = GuardrailResponse(
            request_id="test_req",
            direction=__import__("guardrail.schema", fromlist=["Direction"]).Direction.INPUT,
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
