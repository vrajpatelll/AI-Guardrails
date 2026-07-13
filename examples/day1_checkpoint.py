"""
Day 1 checkpoint script.

Run this after setting up your .env to verify:
  ✓ Policy loads correctly from policy.yaml
  ✓ Config loads from environment
  ✓ Normalizer works on obfuscated text
  ✓ Cache key changes when policy_version changes
  ✓ A real call through the wrapped Anthropic client returns a response
    wrapped in a schema-valid no-op guardrail verdict

Usage:
    cp .env.example .env  # fill in your API key + guardrail token
    python -m dotenv run -- python examples/day1_checkpoint.py
    # or with python-dotenv installed:
    python examples/day1_checkpoint.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # manually export env vars if dotenv isn't installed

# Add repo root to path so we can import guardrail directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrail.cache import VerdictCache
from guardrail.config import GuardrailConfig
from guardrail.middleware import GuardrailMiddleware
from guardrail.normalizer import normalise
from guardrail.policy import load_policy


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(label: str, passed: bool) -> None:
    icon = "✓" if passed else "✗"
    print(f"  {icon}  {label}")
    if not passed:
        sys.exit(1)


def main() -> None:
    print("\n🛡  AI Guardrail — Day 1 Checkpoint")

    # ── 1. Policy loader ───────────────────────────────────────────────
    section("1. Policy loader")
    policy = load_policy("policy.yaml")
    check("policy.name is set", bool(policy.name))
    check("pii category loaded", "pii" in policy.categories)
    check("secrets category loaded", "secrets" in policy.categories)
    check("harmful_content category loaded", "harmful_content" in policy.categories)
    check("prompt_injection category loaded", "prompt_injection" in policy.categories)
    check("policy_version > 0", policy.policy_version > 0)
    print(f"     policy_version = {policy.policy_version}")
    print(f"     latency_budget_ms = {policy.latency_budget_ms}")

    # ── 2. Normalizer ──────────────────────────────────────────────────
    section("2. Normalizer")
    obfuscated = "Héllo\u200b wörld\ufeff"  # zero-width + homoglyphs
    norm = normalise(obfuscated)
    check("zero-width characters stripped", "\u200b" not in norm and "\ufeff" not in norm)
    check("NFKC applied (non-empty result)", len(norm) > 0)
    print(f"     before: {repr(obfuscated)}")
    print(f"     after:  {repr(norm)}")

    # ── 3. Cache key changes on policy reload ─────────────────────────
    section("3. Cache key includes policy_version")
    text = "hello world"
    policy2 = load_policy("policy.yaml")  # reload → new version
    key1 = VerdictCache.make_key(text, policy.policy_version)
    key2 = VerdictCache.make_key(text, policy2.policy_version)
    check("same text, different policy_version → different keys", key1 != key2)
    print(f"     key v{policy.policy_version}: {key1[:16]}...")
    print(f"     key v{policy2.policy_version}: {key2[:16]}...")

    # ── 4. Config from environment ────────────────────────────────────
    section("4. GuardrailConfig from environment")
    try:
        cfg = GuardrailConfig.from_env()
        check("config loads without error", True)
        check("llm_model is set", bool(cfg.llm_model))
        print(f"     model: {cfg.llm_model}")
        print(f"     fail_mode: {cfg.fail_mode}")
    except ValueError as e:
        print(f"\n  ⚠  Skipping live LLM test: {e}")
        print("     Set ANTHROPIC_API_KEY and GUARDRAIL_TOKEN in .env to run the full checkpoint.")
        print("\n  ✓  Day 1 structural checkpoint PASSED (no live LLM call made)\n")
        return

    # ── 5. Live LLM pass-through ──────────────────────────────────────
    section("5. Live LLM pass-through (real Anthropic call)")
    policy_fresh = load_policy("policy.yaml")
    client = GuardrailMiddleware(cfg, policy=policy_fresh)

    response = client.messages.create(
        model=cfg.llm_model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Say 'guardrail checkpoint passed' and nothing else."}],
    )

    verdict = response.guardrail_verdict
    sr = verdict.sanitization_result

    check("response has content", len(response.content) > 0)
    check("guardrail_verdict is attached", verdict is not None)
    check("action is ALLOW", sr.action.value == "ALLOW")
    check("filter_match_state is NO_MATCH_FOUND", sr.filter_match_state.value == "NO_MATCH_FOUND")
    check("cache_hit is False (first call)", not sr.sanitization_metadata.cache_hit)
    check("all 4 categories in filter_results", len(sr.filter_results) >= 4)
    check("sanitized_text is None (no redactions)", sr.sanitized_text is None)

    print(f"\n  LLM reply: {response.content[0].text!r}")
    print("\n  Full guardrail verdict:")
    print(json.dumps(verdict.model_dump(), indent=4, default=str))

    # ── 6. Cache hit on repeated call ────────────────────────────────
    section("6. Cache hit on repeated identical call")
    response2 = client.messages.create(
        model=cfg.llm_model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Say 'guardrail checkpoint passed' and nothing else."}],
    )
    verdict2 = response2.guardrail_verdict
    check(
        "second call is a cache hit",
        verdict2.sanitization_result.sanitization_metadata.cache_hit,
    )
    print("     Cache working — repeated call served from in-memory store.")

    print("\n  🎉  Day 1 checkpoint PASSED — all systems go.\n")


if __name__ == "__main__":
    main()
