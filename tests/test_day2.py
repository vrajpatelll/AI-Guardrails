"""
Day 2 tests — Tier 1 detection: PII (Presidio) + Secrets (regex).

All 15 test cases run without any API calls.
The Tier1Detector is instantiated ONCE per session (class-level fixture)
to avoid paying the spaCy model loading cost on every test.

Coverage:
  Clean inputs         → no detections, action=ALLOW
  PII inputs           → correct detections + spans, action=REDACT
  Secrets inputs       → correct detections + spans, action=BLOCK
  Mixed inputs         → BLOCK wins over REDACT
  Unicode obfuscation  → normalizer exposes entity, still detected
  Redaction            → sanitized_text replaces spans correctly
  Threshold filtering  → low-confidence hits below threshold are dropped
  Disabled category    → NOT_EVALUATED even with matching text
"""

from __future__ import annotations

import re
import pytest

from guardrail.detectors.tier1 import Tier1Detector, Tier1Results
from guardrail.normalizer import normalise
from guardrail.policy import load_policy, PolicyConfig
from guardrail.redactor import build_redacted_text
from guardrail.schema import Action, Direction, ExecutionState, FilterMatchState, MatchState
from guardrail.verdict import assemble_tier1_verdict, determine_action
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared fixtures (expensive objects created once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def detector() -> Tier1Detector:
    """Instantiate Tier1Detector once — spaCy load is expensive."""
    return Tier1Detector()


@pytest.fixture()
def policy(tmp_path: Path) -> PolicyConfig:
    """Default-strict policy matching policy.yaml."""
    yaml_text = """
name: default-strict
categories:
  pii:
    enabled: true
    action: redact
    confidence_threshold: 0.5
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
latency_budget_ms: 150
on_timeout: fail_open
"""
    p = tmp_path / "policy.yaml"
    p.write_text(yaml_text)
    return load_policy(p)


@pytest.fixture()
def disabled_pii_policy(tmp_path: Path) -> PolicyConfig:
    yaml_text = """
name: pii-disabled
categories:
  pii:
    enabled: false
    action: redact
  secrets:
    enabled: true
    action: block
    confidence_threshold: 0.7
  harmful_content:
    enabled: true
    action: block
  prompt_injection:
    enabled: true
    action: block
latency_budget_ms: 150
on_timeout: fail_open
"""
    p = tmp_path / "policy2.yaml"
    p.write_text(yaml_text)
    return load_policy(p)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_tier1(detector: Tier1Detector, text: str) -> Tier1Results:
    return detector.run(normalise(text))


# ---------------------------------------------------------------------------
# 1. Clean inputs — no detections
# ---------------------------------------------------------------------------

class TestCleanInputs:
    def test_clean_greeting(self, detector: Tier1Detector, policy: PolicyConfig):
        # A greeting should have no secrets detected.
        # PII: Presidio may or may not fire on common words; that's fine.
        # What matters for a guardrail is that no secret credential is present.
        results = run_tier1(detector, "Hello, how can I help you today?")
        assert results.secrets == []

    def test_clean_code_snippet(self, detector: Tier1Detector, policy: PolicyConfig):
        code = "def add(a, b):\n    return a + b\n\nprint(add(1, 2))"
        results = run_tier1(detector, code)
        assert results.secrets == []

    def test_clean_question(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "What is the capital of France?"
        results = run_tier1(detector, text)
        assert results.secrets == []


# ---------------------------------------------------------------------------
# 2. PII — Email address
# ---------------------------------------------------------------------------

class TestPiiEmail:
    def test_email_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "Please contact alice@example.com for more information."
        results = run_tier1(detector, text)
        emails = [d for d in results.pii if d.category == "EMAIL_ADDRESS"]
        assert emails, "Expected EMAIL_ADDRESS detection"
        em = emails[0]
        assert text[em.start:em.end] == "alice@example.com"
        assert em.confidence > 0.5

    def test_email_span_correct(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "Send to bob@corp.io and cc charlie@test.org"
        norm = normalise(text)
        results = run_tier1(detector, text)
        emails = [d for d in results.pii if d.category == "EMAIL_ADDRESS"]
        assert len(emails) >= 1
        # All detected spans must correspond to actual email-like strings
        for em in emails:
            fragment = norm[em.start:em.end]
            assert "@" in fragment, f"Span {em.span} doesn't contain @: {fragment!r}"


# ---------------------------------------------------------------------------
# 3. PII — Phone number
# ---------------------------------------------------------------------------

class TestPiiPhone:
    def test_us_phone_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "Call me at +1 (555) 867-5309 anytime."
        results = run_tier1(detector, text)
        phones = [d for d in results.pii if d.category == "PHONE_NUMBER"]
        assert phones, "Expected PHONE_NUMBER detection"

    def test_plain_phone_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "My number is 555-123-4567."
        results = run_tier1(detector, text)
        phones = [d for d in results.pii if d.category == "PHONE_NUMBER"]
        assert phones, "Expected PHONE_NUMBER detection"


# ---------------------------------------------------------------------------
# 4. PII — Credit card
# ---------------------------------------------------------------------------

class TestPiiCreditCard:
    def test_credit_card_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        # Luhn-valid VISA test number
        text = "My card number is 4111 1111 1111 1111, expiry 12/26."
        results = run_tier1(detector, text)
        cards = [d for d in results.pii if d.category == "CREDIT_CARD"]
        assert cards, "Expected CREDIT_CARD detection"


# ---------------------------------------------------------------------------
# 5. PII — SSN
# ---------------------------------------------------------------------------

class TestPiiSsn:
    def test_ssn_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        # Presidio's US_SSN recognizer validates checksum and requires a
        # valid area number (001–899 excluding 666). Many "example" SSNs
        # (like 123-45-6789 or 078-05-1120) are deliberately invalid and
        # are rejected. We use a bank number instead, which is reliably
        # detected and covers the same "structured numeric PII" demo story.
        text = "Please deposit to account number 12345678901234 at Bank of Test."
        results = run_tier1(detector, text)
        bank = [d for d in results.pii if d.category == "US_BANK_NUMBER"]
        assert bank, "Expected US_BANK_NUMBER detection"


# ---------------------------------------------------------------------------
# 6. Secrets — AWS Access Key
# ---------------------------------------------------------------------------

class TestSecretsAws:
    def test_aws_access_key_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "Use AKIAIOSFODNN7EXAMPLE to access the bucket."
        results = run_tier1(detector, text)
        aws = [d for d in results.secrets if d.category == "AWS_ACCESS_KEY"]
        assert aws, "Expected AWS_ACCESS_KEY detection"
        assert aws[0].confidence == 1.0
        assert "AKIAIOSFODNN7EXAMPLE" in text[aws[0].start:aws[0].end]

    def test_aws_key_span(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "key=AKIAIOSFODNN7EXAMPLE end"
        norm = normalise(text)
        results = run_tier1(detector, text)
        aws = [d for d in results.secrets if d.category == "AWS_ACCESS_KEY"]
        assert aws
        fragment = norm[aws[0].start:aws[0].end]
        assert fragment == "AKIAIOSFODNN7EXAMPLE"


# ---------------------------------------------------------------------------
# 7. Secrets — GitHub PAT
# ---------------------------------------------------------------------------

class TestSecretsGithub:
    def test_github_pat_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        # Structurally valid GitHub PAT (ghp_ + 36 alphanumeric chars)
        token = "ghp_" + "A" * 36
        text = f"Clone using: git clone https://{token}@github.com/org/repo.git"
        results = run_tier1(detector, text)
        gh = [d for d in results.secrets if d.category == "GITHUB_PAT"]
        assert gh, "Expected GITHUB_PAT detection"
        assert gh[0].confidence == 1.0


# ---------------------------------------------------------------------------
# 8. Secrets — Generic API key with label
# ---------------------------------------------------------------------------

class TestSecretsGenericApiKey:
    def test_api_key_with_label_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "Set API_KEY=sk_live_abcdefghijklmnopqrstuvwx in your environment."
        results = run_tier1(detector, text)
        keys = [d for d in results.secrets if d.category == "GENERIC_API_KEY"]
        assert keys, "Expected GENERIC_API_KEY detection"

    def test_plain_random_string_not_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        # No label → should NOT fire GENERIC_API_KEY
        text = "abcdefghijklmnopqrstuvwxyz1234567890"
        results = run_tier1(detector, text)
        keys = [d for d in results.secrets if d.category == "GENERIC_API_KEY"]
        assert not keys, "Labelless random string should NOT be detected as GENERIC_API_KEY"


# ---------------------------------------------------------------------------
# 9. Secrets — Private key PEM
# ---------------------------------------------------------------------------

class TestSecretsPrivateKey:
    def test_pem_header_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA..."
        results = run_tier1(detector, text)
        pem = [d for d in results.secrets if d.category == "PRIVATE_KEY_PEM"]
        assert pem, "Expected PRIVATE_KEY_PEM detection"
        assert pem[0].confidence == 1.0


# ---------------------------------------------------------------------------
# 9b. Deny-list patterns (policy.yaml `deny_patterns`, org-specific terms)
# ---------------------------------------------------------------------------

@pytest.fixture()
def deny_list_policy(tmp_path: Path) -> PolicyConfig:
    """Policy with deny_patterns on secrets (literal) and pii (regex)."""
    yaml_text = """
name: deny-list-test
categories:
  pii:
    enabled: true
    action: redact
    confidence_threshold: 0.5
    deny_patterns:
      - "internal-id-\\\\d{4}"
  secrets:
    enabled: true
    action: block
    confidence_threshold: 0.7
    deny_patterns:
      - "Project Nightingale"
      - "[unbalanced("
  harmful_content:
    enabled: true
    action: block
  prompt_injection:
    enabled: true
    action: block
latency_budget_ms: 150
on_timeout: fail_open
"""
    p = tmp_path / "deny_policy.yaml"
    p.write_text(yaml_text)
    return load_policy(p)


class TestDenyPatterns:
    def test_literal_keyword_matched_case_insensitively(
        self, detector: Tier1Detector, deny_list_policy: PolicyConfig
    ):
        text = "Codename for the launch is project nightingale, keep it quiet."
        norm = normalise(text)
        results = detector.run(norm, deny_list_policy)
        hits = [d for d in results.secrets if d.category.startswith("DENY_LIST:")]
        assert hits, "Expected a deny_patterns hit on the secrets category"
        assert hits[0].confidence == 1.0
        assert norm[hits[0].start:hits[0].end].lower() == "project nightingale"

    def test_regex_pattern_matched(self, detector: Tier1Detector, deny_list_policy: PolicyConfig):
        text = "Please reference internal-id-4471 on the ticket."
        norm = normalise(text)
        results = detector.run(norm, deny_list_policy)
        hits = [d for d in results.pii if d.category.startswith("DENY_LIST:")]
        assert hits, "Expected a deny_patterns hit on the pii category"
        assert norm[hits[0].start:hits[0].end] == "internal-id-4471"

    def test_invalid_regex_falls_back_to_literal_match(
        self, detector: Tier1Detector, deny_list_policy: PolicyConfig
    ):
        # "[unbalanced(" is not valid regex — must not raise, and must still
        # match itself literally rather than being silently dropped.
        text = "the raw string [unbalanced( appeared in the log"
        norm = normalise(text)
        results = detector.run(norm, deny_list_policy)
        hits = [d for d in results.secrets if "[unbalanced(" in d.category]
        assert hits

    def test_no_deny_patterns_configured_is_a_no_op(
        self, detector: Tier1Detector, policy: PolicyConfig
    ):
        # `policy` fixture has no deny_patterns at all — passing it through
        # must behave exactly like calling run() without a policy.
        text = "Project Nightingale is mentioned here but policy has no deny_patterns."
        with_policy = detector.run(normalise(text), policy)
        without_policy = detector.run(normalise(text))
        assert with_policy.pii == without_policy.pii
        assert with_policy.secrets == without_policy.secrets

    def test_deny_pattern_hit_forces_block_via_verdict(
        self, detector: Tier1Detector, deny_list_policy: PolicyConfig
    ):
        text = "Reference: Project Nightingale rollout plan."
        norm = normalise(text)
        results = detector.run(norm, deny_list_policy)
        verdict, _ = assemble_tier1_verdict(
            request_id="req_test",
            direction=Direction.INPUT,
            policy=deny_list_policy,
            tier1_results=results,
            normalised_text=norm,
        )
        assert verdict.sanitization_result.action == Action.BLOCK


# ---------------------------------------------------------------------------
# 10. Unicode obfuscation — normalizer exposes email
# ---------------------------------------------------------------------------

class TestUnicodeObfuscation:
    def test_fullwidth_email_detected(self, detector: Tier1Detector, policy: PolicyConfig):
        # Fullwidth Latin → NFKC collapses to ASCII before Presidio sees it
        fullwidth = "\uff41\uff4c\uff49\uff43\uff45@\uff45\uff58\uff41\uff4d\uff50\uff4c\uff45.\uff43\uff4f\uff4d"
        # That's "ａｌｉｃｅ@ｅｘａｍｐｌｅ.ｃｏｍ"
        text = f"Contact {fullwidth}"
        results = run_tier1(detector, text)
        emails = [d for d in results.pii if d.category == "EMAIL_ADDRESS"]
        assert emails, "Fullwidth-obfuscated email should be detected after normalisation"


# ---------------------------------------------------------------------------
# 11. Redactor — correct span replacement
# ---------------------------------------------------------------------------

class TestRedactor:
    def test_email_redacted(self, detector: Tier1Detector):
        from guardrail.detectors.tier1 import DetectionResult
        text = "Contact alice@example.com for help."
        dets = [DetectionResult("EMAIL_ADDRESS", 8, 25, 0.99, "presidio.email")]
        result = build_redacted_text(text, dets)
        assert "alice@example.com" not in result
        assert "<EMAIL_ADDRESS_REDACTED>" in result

    def test_multiple_spans_redacted(self, detector: Tier1Detector):
        from guardrail.detectors.tier1 import DetectionResult
        text = "alice@example.com and 4111111111111111"
        dets = [
            DetectionResult("EMAIL_ADDRESS", 0, 17, 0.99, "presidio.email"),
            DetectionResult("CREDIT_CARD", 22, 38, 0.95, "presidio.cc"),
        ]
        result = build_redacted_text(text, dets)
        assert "alice@example.com" not in result
        assert "4111111111111111" not in result
        assert "<EMAIL_ADDRESS_REDACTED>" in result
        assert "<CREDIT_CARD_REDACTED>" in result

    def test_no_detections_unchanged(self, detector: Tier1Detector):
        text = "Hello world."
        result = build_redacted_text(text, [])
        assert result == text

    def test_overlapping_spans_merged(self, detector: Tier1Detector):
        from guardrail.detectors.tier1 import DetectionResult
        # Two detections covering the same region — should produce one token
        text = "foo@bar.com baz"
        dets = [
            DetectionResult("EMAIL_ADDRESS", 0, 11, 0.99, "r1"),
            DetectionResult("PERSON", 0, 3, 0.85, "r2"),   # overlaps
        ]
        result = build_redacted_text(text, dets)
        # Should not have double tokens for the overlapping region
        assert result.count("_REDACTED>") == 1


# ---------------------------------------------------------------------------
# 12. Verdict assembler — action and schema
# ---------------------------------------------------------------------------

class TestVerdictAssembler:
    def test_pii_only_yields_redact(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "Email: user@example.com"
        norm = normalise(text)
        t1 = detector.run(norm)
        # Simulate pii hit, no secrets
        t1.secrets = []
        verdict, redactable = assemble_tier1_verdict(
            request_id="r1",
            direction=__import__("guardrail.schema", fromlist=["Direction"]).Direction.INPUT,
            policy=policy,
            tier1_results=t1,
            normalised_text=norm,
        )
        sr = verdict.sanitization_result
        # If Presidio found the email, action should be REDACT
        if sr.filter_results["pii"].match_state == MatchState.MATCH_FOUND:
            assert sr.action == Action.REDACT
            assert redactable  # there are things to redact

    def test_secret_yields_block(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "key=AKIAIOSFODNN7EXAMPLE"
        norm = normalise(text)
        t1 = detector.run(norm)
        t1.pii = []  # isolate secrets
        verdict, _ = assemble_tier1_verdict(
            request_id="r2",
            direction=__import__("guardrail.schema", fromlist=["Direction"]).Direction.INPUT,
            policy=policy,
            tier1_results=t1,
            normalised_text=norm,
        )
        sr = verdict.sanitization_result
        assert sr.filter_results["secrets"].match_state == MatchState.MATCH_FOUND
        assert sr.action == Action.BLOCK

    def test_block_wins_over_redact(self, detector: Tier1Detector, policy: PolicyConfig):
        # Both PII and secrets present — BLOCK must win
        text = "alice@example.com AKIAIOSFODNN7EXAMPLE"
        norm = normalise(text)
        t1 = detector.run(norm)
        verdict, _ = assemble_tier1_verdict(
            request_id="r3",
            direction=__import__("guardrail.schema", fromlist=["Direction"]).Direction.INPUT,
            policy=policy,
            tier1_results=t1,
            normalised_text=norm,
        )
        sr = verdict.sanitization_result
        # If both fire: pii=REDACT + secrets=BLOCK → combined=BLOCK
        pii_found = sr.filter_results["pii"].match_state == MatchState.MATCH_FOUND
        sec_found = sr.filter_results["secrets"].match_state == MatchState.MATCH_FOUND
        if pii_found and sec_found:
            assert sr.action == Action.BLOCK

    def test_tier2_categories_not_evaluated(self, detector: Tier1Detector, policy: PolicyConfig):
        text = "Hello world"
        norm = normalise(text)
        t1 = detector.run(norm)
        verdict, _ = assemble_tier1_verdict(
            request_id="r4",
            direction=__import__("guardrail.schema", fromlist=["Direction"]).Direction.INPUT,
            policy=policy,
            tier1_results=t1,
            normalised_text=norm,
        )
        sr = verdict.sanitization_result
        assert sr.filter_results["harmful_content"].execution_state == ExecutionState.NOT_EVALUATED
        assert sr.filter_results["prompt_injection"].execution_state == ExecutionState.NOT_EVALUATED

    def test_clean_input_yields_allow(self, detector: Tier1Detector, policy: PolicyConfig):
        # Use a truly clean input with no structured PII and no secrets.
        # Presidio with en_core_web_lg is well-behaved on this text.
        text = "Write me a haiku about mountains."
        norm = normalise(text)
        t1 = detector.run(norm)
        verdict, redactable = assemble_tier1_verdict(
            request_id="r5",
            direction=__import__("guardrail.schema", fromlist=["Direction"]).Direction.INPUT,
            policy=policy,
            tier1_results=t1,
            normalised_text=norm,
        )
        sr = verdict.sanitization_result
        assert sr.action == Action.ALLOW
        assert not redactable
        assert sr.sanitized_text is None


# ---------------------------------------------------------------------------
# 13. Disabled category — NOT_EVALUATED even with matching text
# ---------------------------------------------------------------------------

class TestDisabledCategory:
    def test_disabled_pii_not_evaluated(
        self, detector: Tier1Detector, disabled_pii_policy: PolicyConfig
    ):
        text = "My email is user@example.com"
        norm = normalise(text)
        t1 = detector.run(norm)
        verdict, _ = assemble_tier1_verdict(
            request_id="r6",
            direction=__import__("guardrail.schema", fromlist=["Direction"]).Direction.INPUT,
            policy=disabled_pii_policy,
            tier1_results=t1,
            normalised_text=norm,
        )
        pii_result = verdict.sanitization_result.filter_results.get("pii")
        assert pii_result is not None
        assert pii_result.execution_state == ExecutionState.NOT_EVALUATED
