"""
Day 1 unit tests — no API calls, no external dependencies.

Covers:
  - Schema instantiation and field defaults
  - Policy loader (from a temp file)
  - Normalizer correctness
  - Cache key determinism and TTL expiry
  - Stub verdict shape and content
  - Config validation
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from guardrail.cache import VerdictCache
from guardrail.config import GuardrailConfig
from guardrail.normalizer import normalise
from guardrail.policy import load_policy, PolicyConfig, CategoryPolicy
from guardrail.schema import (
    Action,
    CategoryResult,
    Direction,
    ExecutionState,
    FilterMatchState,
    GuardrailRequest,
    GuardrailResponse,
    InvocationResult,
    MatchState,
    SanitizationMetadata,
    SanitizationResult,
)
from guardrail.verdict import build_stub_verdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def policy_yaml(tmp_path: Path) -> Path:
    """Write a minimal policy.yaml to a temp directory."""
    content = """
name: test-policy
categories:
  pii:
    enabled: true
    action: redact
    confidence_threshold: 0.8
  secrets:
    enabled: true
    action: block
  harmful_content:
    enabled: true
    action: block
    confidence_threshold: 0.7
  prompt_injection:
    enabled: false
    action: block
latency_budget_ms: 200
on_timeout: fail_open
"""
    p = tmp_path / "policy.yaml"
    p.write_text(content)
    return p


@pytest.fixture()
def policy(policy_yaml: Path) -> PolicyConfig:
    return load_policy(policy_yaml)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_sanitization_result_defaults(self):
        sr = SanitizationResult(
            filter_match_state=FilterMatchState.NO_MATCH_FOUND,
            invocation_result=InvocationResult.SUCCESS,
            action=Action.ALLOW,
        )
        assert sr.latency_ms == 0
        assert sr.sanitized_text is None
        assert sr.filter_results == {}

    def test_guardrail_response_round_trip(self, policy: PolicyConfig):
        resp = build_stub_verdict(
            request_id="req_test",
            direction=Direction.INPUT,
            policy=policy,
            original_text="hello world",
        )
        dumped = resp.model_dump_json()
        restored = GuardrailResponse.model_validate_json(dumped)
        assert restored.request_id == "req_test"
        assert restored.sanitization_result.action == Action.ALLOW

    def test_sanitized_text_is_none_on_allow(self, policy: PolicyConfig):
        verdict = build_stub_verdict(
            request_id="req_1",
            direction=Direction.INPUT,
            policy=policy,
            original_text="clean text",
        )
        assert verdict.sanitization_result.sanitized_text is None

    def test_category_result_fields(self):
        cr = CategoryResult(
            execution_state=ExecutionState.EXECUTION_SUCCESS,
            match_state=MatchState.NO_MATCH_FOUND,
            tier=1,
        )
        assert cr.detections == []
        assert cr.model is None


# ---------------------------------------------------------------------------
# Policy loader tests
# ---------------------------------------------------------------------------

class TestPolicyLoader:
    def test_loads_all_categories(self, policy: PolicyConfig):
        assert "pii" in policy.categories
        assert "secrets" in policy.categories
        assert "harmful_content" in policy.categories
        assert "prompt_injection" in policy.categories

    def test_disabled_category(self, policy: PolicyConfig):
        pi_cat = policy.categories["prompt_injection"]
        assert pi_cat.enabled is False

    def test_policy_version_increments_on_reload(self, policy_yaml: Path):
        p1 = load_policy(policy_yaml)
        p2 = load_policy(policy_yaml)
        assert p2.policy_version > p1.policy_version

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_policy(tmp_path / "nonexistent.yaml")

    def test_latency_budget_loaded(self, policy: PolicyConfig):
        assert policy.latency_budget_ms == 200

    def test_on_timeout_loaded(self, policy: PolicyConfig):
        assert policy.on_timeout.value == "fail_open"


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_strips_zero_width_space(self):
        assert "\u200b" not in normalise("hel\u200blo")

    def test_strips_bom(self):
        assert "\ufeff" not in normalise("\ufeffhello")

    def test_nfkc_collapses_fullwidth(self):
        # Fullwidth Latin letters → ASCII equivalents under NFKC
        result = normalise("\uff48\uff45\uff4c\uff4c\uff4f")  # ｈｅｌｌｏ
        assert result == "hello"

    def test_plain_text_unchanged(self):
        text = "The quick brown fox."
        assert normalise(text) == text

    def test_empty_string(self):
        assert normalise("") == ""

    def test_strips_multiple_zero_width_chars(self):
        text = "a\u200bb\u200cc\u200dd"
        result = normalise(text)
        assert result == "abcd"


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestVerdictCache:
    def test_miss_returns_none(self):
        cache = VerdictCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self, policy: PolicyConfig):
        cache = VerdictCache()
        verdict = build_stub_verdict("req_1", Direction.INPUT, policy, "text")
        key = VerdictCache.make_key("text", policy.policy_version)
        cache.set(key, verdict)
        hit = cache.get(key)
        assert hit is not None
        assert hit.request_id == "req_1"

    def test_ttl_expiry(self, policy: PolicyConfig):
        cache = VerdictCache(ttl_seconds=0.01)  # 10ms TTL
        verdict = build_stub_verdict("req_x", Direction.INPUT, policy, "text")
        key = VerdictCache.make_key("text", policy.policy_version)
        cache.set(key, verdict)
        time.sleep(0.05)  # wait past TTL
        assert cache.get(key) is None

    def test_key_changes_with_policy_version(self, policy_yaml: Path):
        p1 = load_policy(policy_yaml)
        p2 = load_policy(policy_yaml)
        k1 = VerdictCache.make_key("same text", p1.policy_version)
        k2 = VerdictCache.make_key("same text", p2.policy_version)
        assert k1 != k2, "Policy reload must produce a different cache key"

    def test_clear_empties_cache(self, policy: PolicyConfig):
        cache = VerdictCache()
        verdict = build_stub_verdict("req_2", Direction.INPUT, policy, "t")
        key = VerdictCache.make_key("t", policy.policy_version)
        cache.set(key, verdict)
        cache.clear()
        assert cache.size() == 0

    def test_same_text_same_policy_same_key(self, policy: PolicyConfig):
        k1 = VerdictCache.make_key("hello", policy.policy_version)
        k2 = VerdictCache.make_key("hello", policy.policy_version)
        assert k1 == k2


# ---------------------------------------------------------------------------
# Stub verdict tests
# ---------------------------------------------------------------------------

class TestStubVerdict:
    def test_action_is_allow(self, policy: PolicyConfig):
        v = build_stub_verdict("r1", Direction.INPUT, policy, "text")
        assert v.sanitization_result.action == Action.ALLOW

    def test_no_match_found(self, policy: PolicyConfig):
        v = build_stub_verdict("r1", Direction.INPUT, policy, "text")
        assert v.sanitization_result.filter_match_state == FilterMatchState.NO_MATCH_FOUND

    def test_enabled_categories_are_execution_success(self, policy: PolicyConfig):
        v = build_stub_verdict("r1", Direction.INPUT, policy, "text")
        for name, result in v.sanitization_result.filter_results.items():
            cat = policy.categories[name]
            if cat.enabled:
                assert result.execution_state == ExecutionState.EXECUTION_SUCCESS
                assert result.detections == []

    def test_disabled_categories_are_not_evaluated(self, policy: PolicyConfig):
        v = build_stub_verdict("r1", Direction.INPUT, policy, "text")
        pi_result = v.sanitization_result.filter_results.get("prompt_injection")
        assert pi_result is not None
        assert pi_result.execution_state == ExecutionState.NOT_EVALUATED

    def test_tier_assignments(self, policy: PolicyConfig):
        v = build_stub_verdict("r1", Direction.INPUT, policy, "text")
        assert v.sanitization_result.filter_results["pii"].tier == 1
        assert v.sanitization_result.filter_results["secrets"].tier == 1
        assert v.sanitization_result.filter_results["harmful_content"].tier == 2

    def test_policy_version_in_metadata(self, policy: PolicyConfig):
        v = build_stub_verdict("r1", Direction.INPUT, policy, "text")
        assert v.sanitization_result.sanitization_metadata.policy_version == policy.policy_version

    def test_cache_hit_flag_propagates(self, policy: PolicyConfig):
        v = build_stub_verdict("r1", Direction.INPUT, policy, "text", cache_hit=True)
        assert v.sanitization_result.sanitization_metadata.cache_hit is True


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestGuardrailConfig:
    def test_from_env_raises_on_missing_keys(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GUARDRAIL_TOKEN", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            GuardrailConfig.from_env()

    def test_safe_dict_redacts_secrets(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real-key")
        monkeypatch.setenv("GUARDRAIL_TOKEN", "real-token")
        cfg = GuardrailConfig.from_env()
        safe = cfg.safe_dict()
        assert safe["llm_api_key"] == "***"
        assert safe["guardrail_token"] == "***"

    def test_fail_mode_validation(self):
        with pytest.raises(ValueError):
            GuardrailConfig(
                llm_api_key="k",
                guardrail_token="t",
                fail_mode="invalid_mode",
            )

    def test_log_level_uppercased(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("GUARDRAIL_TOKEN", "t")
        monkeypatch.setenv("GUARDRAIL_LOG_LEVEL", "debug")
        cfg = GuardrailConfig.from_env()
        assert cfg.log_level == "DEBUG"
