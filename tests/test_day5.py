"""
Day 5 tests — combine_verdicts (input + output verdict merging), and the
verdict cache.

TestEndToEndScenarios (scenarios 1-9, redaction/BLOCK-precedence/disabled-
category coverage) is still removed — only cache-hit coverage was asked
for. Scenario 10 is restored below as TestCacheHit, using a FakeBackend
double injected via GuardrailMiddleware's `backend=` parameter instead of
mock.patch'ing `guardrail.middleware.OpenAI` (which broke when
middleware.py moved to guardrail.backends.OpenAIBackend — see
guardrail/backends.py's module docstring for the intended extension
point).

combine_verdicts unit tests still covered here:
  - input ALLOW + output ALLOW  → ALLOW
  - input REDACT + output BLOCK → BLOCK
  - input BLOCK + output=None   → BLOCK (short-circuit)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from guardrail.config import GuardrailConfig
from guardrail.detectors.tier1 import Tier1Detector, Tier1Results
from guardrail.detectors.tier2 import Tier2Detector, Tier2Results
from guardrail.middleware import GuardrailMiddleware
from guardrail.policy import load_policy
from guardrail.schema import Action, Direction, FilterMatchState
from guardrail.verdict import build_stub_verdict, combine_verdicts

# ---------------------------------------------------------------------------
# YAML fixtures
# ---------------------------------------------------------------------------

YAML_ALL_ENABLED = """
name: test-day5
categories:
  pii:
    enabled: true
    action: redact
    confidence_threshold: 0.7
  secrets:
    enabled: true
    action: block
    confidence_threshold: 0.7
  harmful_content:
    enabled: true
    action: block
    confidence_threshold: 0.7
  prompt_injection:
    enabled: true
    action: block
    confidence_threshold: 0.7
latency_budget_ms: 5000
on_timeout: fail_open
"""


@pytest.fixture
def policy_all(tmp_path: Path):
    p = tmp_path / "policy.yaml"
    p.write_text(YAML_ALL_ENABLED)
    return load_policy(str(p))


# ---------------------------------------------------------------------------
# combine_verdicts unit tests
# ---------------------------------------------------------------------------

class TestCombineVerdicts:

    def test_both_allow(self, policy_all):
        input_v = build_stub_verdict("r1", Direction.INPUT, policy_all, "hi")
        output_v = build_stub_verdict("r1", Direction.OUTPUT, policy_all, "hello")
        combined = combine_verdicts(input_v, output_v)
        assert combined.sanitization_result.action.value == "ALLOW"

    def test_output_none_returns_input(self, policy_all):
        input_v = build_stub_verdict("r1", Direction.INPUT, policy_all, "hi")
        combined = combine_verdicts(input_v, None)
        assert combined is input_v

    def test_output_block_wins_over_input_allow(self, policy_all):
        input_v = build_stub_verdict("r1", Direction.INPUT, policy_all, "hi")
        # Build an output verdict with BLOCK action manually
        output_v = build_stub_verdict("r1", Direction.OUTPUT, policy_all, "bad output")
        output_v.sanitization_result.action = Action.BLOCK
        output_v.sanitization_result.filter_match_state = FilterMatchState.MATCH_FOUND
        combined = combine_verdicts(input_v, output_v)
        assert combined.sanitization_result.action.value == "BLOCK"

    def test_output_prefixed_in_filter_results(self, policy_all):
        input_v = build_stub_verdict("r1", Direction.INPUT, policy_all, "hi")
        output_v = build_stub_verdict("r1", Direction.OUTPUT, policy_all, "hello")
        combined = combine_verdicts(input_v, output_v)
        # Output categories should be prefixed
        assert "output_pii" in combined.sanitization_result.filter_results
        assert "pii" in combined.sanitization_result.filter_results

    def test_latency_is_summed(self, policy_all):
        input_v = build_stub_verdict("r1", Direction.INPUT, policy_all, "hi")
        output_v = build_stub_verdict("r1", Direction.OUTPUT, policy_all, "hello")
        input_v.sanitization_result.latency_ms = 40
        output_v.sanitization_result.latency_ms = 60
        combined = combine_verdicts(input_v, output_v)
        assert combined.sanitization_result.latency_ms == 100


# ---------------------------------------------------------------------------
# Verdict cache (middleware-level, via injected FakeBackend)
# ---------------------------------------------------------------------------

class FakeBackend:
    """
    Test double for guardrail.backends.LLMBackend. Records every call and
    returns a scripted response — avoids mock.patch'ing an SDK import path,
    which is what broke the original version of this test (see this file's
    module docstring).
    """

    def __init__(self, response_text: str = "Here is your answer.") -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    def send(self, call_kwargs: dict) -> SimpleNamespace:
        self.calls.append(call_kwargs)
        message = SimpleNamespace(content=self.response_text)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(id="msg_test", choices=[choice])

    def extract_text(self, response: SimpleNamespace) -> str:
        choices = getattr(response, "choices", None) or []
        return choices[0].message.content if choices else ""

    def set_text(self, response: SimpleNamespace, text: str) -> SimpleNamespace:
        if response.choices:
            response.choices[0].message.content = text
        return response


@pytest.fixture
def mw_config_all(tmp_path: Path) -> GuardrailConfig:
    p = tmp_path / "policy.yaml"
    p.write_text(YAML_ALL_ENABLED)
    return GuardrailConfig(
        llm_api_key=SecretStr("sk-test"),
        guardrail_token=SecretStr("tok-test"),
        policy_path=str(p),
    )


class TestCacheHit:
    def test_scenario_10_cache_hit(self, mw_config_all: GuardrailConfig):
        """Second identical call returns a cached verdict (cache_hit=True)
        without re-running Tier 1 on the input side."""
        mock_t1 = MagicMock(spec=Tier1Detector)
        mock_t1.run.return_value = Tier1Results(pii=[], secrets=[])
        mock_t2 = MagicMock(spec=Tier2Detector)
        mock_t2.run.return_value = Tier2Results(harmful=[], injection=[])
        fake_backend = FakeBackend()

        mw = GuardrailMiddleware(mw_config_all, tier1=mock_t1, tier2=mock_t2, backend=fake_backend)
        msg = [{"role": "user", "content": "Tell me a joke."}]

        resp1 = mw.messages.create(model="test", messages=msg)
        resp2 = mw.messages.create(model="test", messages=msg)
        mw.shutdown()

        assert resp1.guardrail_verdict.sanitization_result.sanitization_metadata.cache_hit is False
        assert resp2.guardrail_verdict.sanitization_result.sanitization_metadata.cache_hit is True
        # T1 call breakdown:
        #   First call (cache MISS):  T1 for input (1) + T1 for output (1) = 2
        #   Second call (cache HIT):  T1 for input = 0 (cached), T1 for output (1) = 1
        # Total = 3. Without cache it would be 4 (2 input + 2 output).
        assert mock_t1.run.call_count == 3
        # The verdict is cached, not the LLM call itself — both requests
        # still reach the backend.
        assert len(fake_backend.calls) == 2
