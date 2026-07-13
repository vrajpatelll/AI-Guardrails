# guardrail — two-tier LLM guardrail system
# Tier 1: Presidio (PII + secrets) — deterministic, always runs
# Tier 2: Qwen3Guard (harmful content + jailbreak) — LLM-based, always runs
# Both tiers run concurrently; they check unrelated categories.
