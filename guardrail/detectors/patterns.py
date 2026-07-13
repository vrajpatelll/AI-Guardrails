"""
Secrets regex pattern catalog for Tier 1 detection.

Design goals:
- High precision over recall: every pattern here should have a very low
  false-positive rate so we don't block legitimate traffic.
- Each entry has a name (used as Detection.category), a compiled regex,
  and a base confidence score.
- Extending: add a new SecretPattern entry here — no code changes needed
  elsewhere; SecretsDetector reads the CATALOG list at startup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretPattern:
    name: str          # used as Detection.category, e.g. "AWS_ACCESS_KEY"
    pattern: re.Pattern[str]
    confidence: float  # base confidence (0.0-1.0)
    rule: str          # human-readable rule ID


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

CATALOG: list[SecretPattern] = [

    # ── AWS ─────────────────────────────────────────────────────────────────
    SecretPattern(
        name="AWS_ACCESS_KEY",
        # AWS access keys always start with AKIA and are exactly 20 chars
        pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        confidence=1.0,
        rule="secrets.aws_access_key",
    ),
    SecretPattern(
        name="AWS_SECRET_KEY",
        # 40-char base64url string preceded by a known label
        pattern=re.compile(
            r"(?i)(?:aws[_\s]?secret[_\s]?(?:access[_\s]?)?key|aws[_\s]?secret)"
            r"[^A-Za-z0-9]{1,10}"
            r"([A-Za-z0-9/+]{40})"
        ),
        confidence=0.9,
        rule="secrets.aws_secret_key",
    ),

    # ── GitHub ───────────────────────────────────────────────────────────────
    SecretPattern(
        name="GITHUB_PAT",
        # Classic PAT: ghp_, gho_, ghu_, ghs_, ghr_ followed by 36 base36 chars
        pattern=re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),
        confidence=1.0,
        rule="secrets.github_pat",
    ),
    SecretPattern(
        name="GITHUB_FINE_GRAINED_PAT",
        # Fine-grained PAT: github_pat_ followed by 82 chars
        pattern=re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b"),
        confidence=1.0,
        rule="secrets.github_fine_grained_pat",
    ),

    # ── Slack ────────────────────────────────────────────────────────────────
    SecretPattern(
        name="SLACK_TOKEN",
        # xoxb- bot, xoxp- user, xoxa- workspace app, xoxr- refresh, xoxs- socket
        pattern=re.compile(r"\bxox[bpars]-[0-9A-Za-z\-]{10,72}\b"),
        confidence=0.98,
        rule="secrets.slack_token",
    ),

    # ── Generic Bearer / API key ─────────────────────────────────────────────
    SecretPattern(
        name="BEARER_TOKEN",
        # Authorization: Bearer <token>  - requires the "Bearer" keyword
        pattern=re.compile(
            r"(?i)\bbearer\s+([A-Za-z0-9\-_]{20,512})"
        ),
        confidence=0.85,
        rule="secrets.bearer_token",
    ),
    SecretPattern(
        name="GENERIC_API_KEY",
        # API_KEY=<value>, api-key: <value>, apiKey: "<value>"
        # Only fires when a label is present
        pattern=re.compile(
            r"(?i)(?:api[_\-\s]?key|apikey|api[_\-\s]?token|access[_\-\s]?token)"
            r"[\s:='\"`]{1,5}"
            r"([A-Za-z0-9\-_.]{16,256})"
        ),
        confidence=0.8,
        rule="secrets.generic_api_key",
    ),

    # ── Private key material ─────────────────────────────────────────────────
    SecretPattern(
        name="PRIVATE_KEY_PEM",
        # PEM-encoded private key header - RSA, EC, generic
        pattern=re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
        ),
        confidence=1.0,
        rule="secrets.private_key_pem",
    ),

    # ── Anthropic ────────────────────────────────────────────────────────────
    SecretPattern(
        name="ANTHROPIC_API_KEY",
        # sk-ant-api03-<base64url>
        pattern=re.compile(r"\bsk-ant-api\d{2}-[A-Za-z0-9\-_]{93}AA\b"),
        confidence=1.0,
        rule="secrets.anthropic_api_key",
    ),

    # ── OpenAI ───────────────────────────────────────────────────────────────
    SecretPattern(
        name="OPENAI_API_KEY",
        # sk-proj-<48 chars> or legacy sk-<48 chars>
        pattern=re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}\b"),
        confidence=1.0,
        rule="secrets.openai_api_key",
    ),

    # ── Generic hex secret (high-entropy with label) ─────────────────────────
    SecretPattern(
        name="GENERIC_SECRET",
        # SOME_SECRET=<hex string >= 32 chars>
        pattern=re.compile(
            r"(?i)(?:secret|passwd|password|private[_\-]?key|secret[_\-]?key)"
            r"[\s:='\"`]{1,5}"
            r"([0-9A-Fa-f]{32,})"
        ),
        confidence=0.75,
        rule="secrets.generic_secret_hex",
    ),
]
