"""
Individual guardrail checks for tool-call parameters.

Each check is a pure function: given a parameter value, return a
list[Finding] of violations (empty = clean). Kept separate from
middleware.py/registry.py so each check is independently testable — the
same "one job per function" shape as guardrail/detectors/tier1.py's
sub-detectors (PiiDetector, SecretsDetector).

A "block" severity finding always wins outright (see middleware.py); a
"review" finding routes to HUMAN_APPROVAL instead of auto-allowing.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

from guardrail.detectors.patterns import CATALOG
from guardrail.tools.schema import Finding

# ---------------------------------------------------------------------------
# PII / secrets inspection
#
# Reuses the same high-precision secrets catalog the text guardrail's Tier 1
# already ships (guardrail/detectors/patterns.py) — API keys, tokens, PEM
# private keys — rather than re-deriving those regexes. Tool-call parameters
# are short structured strings, not free-form prose, so a lightweight regex
# pass is enough here; there's no need to pull in Presidio/spaCy just to
# scan a single "location" or "query" argument.
#
# Two additions beyond the catalog: SSN (not in CATALOG, which is
# secrets-only) and a permissive password-literal pattern. CATALOG's own
# GENERIC_SECRET pattern intentionally requires a 32+ char hex value after
# a "secret"/"password" label (tuned for high precision on real key
# material) — too strict to catch an ordinary demo password like
# "hunter2!", so a separate, looser pattern covers that case explicitly.
# ---------------------------------------------------------------------------

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Requires an actual assignment separator (':' or '=') between "password"
# and the value — bare whitespace isn't enough, or "SELECT password FROM
# admins" (a column reference, not a credential) would false-positive.
_PASSWORD_PATTERN = re.compile(r"(?i)\bpassword\s*[:=]\s*[\"']?\S{4,}")


def check_pii(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for secret_pattern in CATALOG:
        if secret_pattern.pattern.search(text):
            label = secret_pattern.name.replace("_", " ").title()
            findings.append(Finding(
                check="pii", severity="block",
                message=f"Parameter contains what looks like a {label} ({secret_pattern.rule}).",
            ))
    if _SSN_PATTERN.search(text):
        findings.append(Finding(
            check="pii", severity="block",
            message="Parameter contains what looks like a US Social Security Number.",
        ))
    if _PASSWORD_PATTERN.search(text):
        findings.append(Finding(
            check="pii", severity="block",
            message="Parameter contains what looks like a plaintext password.",
        ))
    return findings


# ---------------------------------------------------------------------------
# SQL safety — strictly read-only SELECT
# ---------------------------------------------------------------------------

_SQL_DANGEROUS_KEYWORDS = (
    "DROP", "DELETE", "UPDATE", "ALTER", "INSERT", "TRUNCATE",
    "GRANT", "REVOKE", "CREATE", "EXEC", "EXECUTE", "MERGE", "REPLACE",
)


def check_sql_safety(query: str) -> list[Finding]:
    findings: list[Finding] = []
    stripped = query.strip().rstrip(";")
    upper = stripped.upper()

    if not upper.startswith("SELECT"):
        findings.append(Finding(
            check="sql_safety", severity="block",
            message="Only read-only SELECT statements are permitted.",
        ))

    for kw in _SQL_DANGEROUS_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            findings.append(Finding(
                check="sql_safety", severity="block",
                message=f"Query contains the disallowed keyword '{kw}'.",
            ))

    if ";" in stripped:
        findings.append(Finding(
            check="sql_safety", severity="block",
            message="Stacked queries (multiple statements separated by ';') are not allowed.",
        ))

    if re.search(r"(--|#|/\*)", query):
        findings.append(Finding(
            check="sql_safety", severity="block",
            message="Query contains a comment sequence, often used to truncate/bypass filtering.",
        ))

    if re.search(r"\bOR\b\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?", upper):
        findings.append(Finding(
            check="sql_safety", severity="block",
            message="Query contains a tautology pattern (e.g. OR 1=1), typical of SQL injection.",
        ))

    if "UNION" in upper:
        findings.append(Finding(
            check="sql_safety", severity="review",
            message="Query uses UNION — flagged for human review (can exfiltrate cross-table data).",
        ))

    return findings


# ---------------------------------------------------------------------------
# SSRF prevention — URL scheme + domain allowlist + private/link-local IPs
# ---------------------------------------------------------------------------

DOMAIN_ALLOWLIST = {
    "api.github.com",
    "api.weather.gov",
    "jsonplaceholder.typicode.com",
}

_KNOWN_METADATA_HOSTS = {"169.254.169.254", "metadata.google.internal", "localhost"}


def check_ssrf(url: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        parsed = urlparse(url)
    except ValueError:
        return [Finding(check="ssrf", severity="block", message="URL could not be parsed.")]

    if parsed.scheme not in ("http", "https"):
        findings.append(Finding(
            check="ssrf", severity="block",
            message=f"Unsupported URL scheme '{parsed.scheme}' — only http/https are allowed.",
        ))

    host = parsed.hostname or ""
    if not host:
        findings.append(Finding(check="ssrf", severity="block", message="URL has no host."))
        return findings

    # A literal IP is checked directly against private/loopback/link-local
    # ranges. A hostname is checked against the allowlist. Note: a real
    # deployment should also resolve the hostname's DNS and re-check the
    # resulting IP to prevent DNS-rebinding SSRF — out of scope for this
    # demo, which never makes a real network call either way.
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            findings.append(Finding(
                check="ssrf", severity="block",
                message=f"URL host {host} is a private/internal IP address — blocked to prevent SSRF.",
            ))
    except ValueError:
        if host.lower() in _KNOWN_METADATA_HOSTS:
            findings.append(Finding(
                check="ssrf", severity="block",
                message=f"URL host '{host}' is a known cloud metadata/internal service — blocked.",
            ))
        elif host.lower() not in DOMAIN_ALLOWLIST:
            findings.append(Finding(
                check="ssrf", severity="block",
                message=(
                    f"URL host '{host}' is not in the domain allowlist "
                    f"({', '.join(sorted(DOMAIN_ALLOWLIST))})."
                ),
            ))

    return findings


# ---------------------------------------------------------------------------
# Command execution safety
# ---------------------------------------------------------------------------

_SHELL_CHAIN_TOKENS = (";", "&&", "|", "`", "$(", ">", "<")


def check_command_injection(command: str) -> list[Finding]:
    findings: list[Finding] = []
    for token in _SHELL_CHAIN_TOKENS:
        if token in command:
            findings.append(Finding(
                check="command_injection", severity="block",
                message=f"Command contains the shell chaining/redirection token '{token}'.",
            ))
    if "../" in command:
        findings.append(Finding(
            check="command_injection", severity="block",
            message="Command contains a path-traversal pattern ('../').",
        ))
    return findings
