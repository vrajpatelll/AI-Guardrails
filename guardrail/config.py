"""
Integrator-facing connection config (Section 4 of development-plan.md).

Mandatory fields (can't run without them):
  - llm_api_key:     API key for the wrapped LLM gateway
  - guardrail_token: auth token protecting the guardrail endpoint

Optional fields (sensible defaults ship, real users will override):
  - llm_model:       which gateway model to use
  - llm_base_url:    base URL of the LLM gateway (OpenAI-compatible)
  - llm_cert_path:   path to a CA cert file for verifying the gateway's TLS cert
  - policy_path:     path to policy.yaml
  - cache_ttl:       verdict cache TTL in seconds
  - log_level:       async logger verbosity

All fields are loaded from environment variables by from_env() so no
secrets are hardcoded or logged anywhere.

Environment variables:
  LLM_GATEWAY_API_KEY     → llm_api_key      (required)
  GUARDRAIL_TOKEN         → guardrail_token  (required)
  GUARDRAIL_MODEL         → llm_model        (default: Bedrock-ant-haiku-4-5-20251001-v1-0)
  GUARDRAIL_LLM_BASE_URL  → llm_base_url     (default: https://apiproxy.cdsys.local)
  GUARDRAIL_LLM_CERT_PATH → llm_cert_path    (default: None)
  GUARDRAIL_POLICY_PATH   → policy_path      (default: policy.yaml)
  GUARDRAIL_CACHE_TTL     → cache_ttl        (default: 300)
  GUARDRAIL_LOG_LEVEL     → log_level        (default: INFO)
  GUARDRAIL_FAIL_MODE     → fail_mode        (default: fail_open)
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, SecretStr


class GuardrailConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Mandatory
    # ------------------------------------------------------------------
    llm_api_key: SecretStr = Field(
        ...,
        description="LLM gateway API key.  Never logged.",
    )
    guardrail_token: SecretStr = Field(
        ...,
        description=(
            "Auth token that protects the guardrail endpoint. "
            "Pass as Authorization: Bearer <token> on every call."
        ),
    )

    # ------------------------------------------------------------------
    # Optional — LLM
    # ------------------------------------------------------------------
    llm_model: str = Field(
        default="Bedrock-ant-haiku-4-5-20251001-v1-0",
        description="LLM gateway model identifier to use for wrapped calls.",
    )
    llm_base_url: str = Field(
        default="https://apiproxy.cdsys.local/v1",
        description="Base URL of the OpenAI-compatible LLM gateway.",
    )
    llm_cert_path: Path | None = Field(
        default=None,
        description=(
            "Path to a CA cert file for verifying the gateway's TLS "
            "certificate (required when on the Crest network/VPN)."
        ),
    )

    # ------------------------------------------------------------------
    # Optional — policy
    # ------------------------------------------------------------------
    policy_path: Path = Field(
        default=Path("policy.yaml"),
        description="Path to the policy-as-code YAML file.",
    )

    # ------------------------------------------------------------------
    # Optional — cache
    # ------------------------------------------------------------------
    cache_ttl: float = Field(
        default=300.0,
        ge=0.0,
        description="Verdict cache TTL in seconds.",
    )

    # ------------------------------------------------------------------
    # Optional — latency / reliability
    # ------------------------------------------------------------------
    fail_mode: str = Field(
        default="fail_open",
        description=(
            "Behaviour on detector timeout or failure. "
            "'fail_open' → allow, 'fail_closed' → block."
        ),
    )

    # ------------------------------------------------------------------
    # Optional — observability
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Log level for the async structured logger.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("fail_mode")
    @classmethod
    def _validate_fail_mode(cls, v: str) -> str:
        allowed = {"fail_open", "fail_closed"}
        if v not in allowed:
            raise ValueError(f"fail_mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "GuardrailConfig":
        """
        Load config from environment variables.

        Required env vars:
          LLM_GATEWAY_API_KEY, GUARDRAIL_TOKEN

        Raises:
          ValueError if any mandatory variable is missing.
        """
        missing: list[str] = []

        llm_api_key = os.environ.get("LLM_GATEWAY_API_KEY", "")
        if not llm_api_key:
            missing.append("LLM_GATEWAY_API_KEY")

        guardrail_token = os.environ.get("GUARDRAIL_TOKEN", "")
        if not guardrail_token:
            missing.append("GUARDRAIL_TOKEN")

        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}\n"
                "Set them in your .env file or export them before running."
            )

        cert_path_str = os.environ.get("GUARDRAIL_LLM_CERT_PATH", "")

        return cls(
            llm_api_key=SecretStr(llm_api_key),
            guardrail_token=SecretStr(guardrail_token),
            llm_model=os.environ.get(
                "GUARDRAIL_MODEL", "Bedrock-ant-haiku-4-5-20251001-v1-0"
            ),
            llm_base_url=os.environ.get(
                "GUARDRAIL_LLM_BASE_URL", "https://apiproxy.cdsys.local"
            ),
            llm_cert_path=Path(cert_path_str) if cert_path_str else None,
            policy_path=Path(
                os.environ.get("GUARDRAIL_POLICY_PATH", "policy.yaml")
            ),
            cache_ttl=float(os.environ.get("GUARDRAIL_CACHE_TTL", "300")),
            fail_mode=os.environ.get("GUARDRAIL_FAIL_MODE", "fail_open"),
            log_level=os.environ.get("GUARDRAIL_LOG_LEVEL", "INFO"),
        )

    def safe_dict(self) -> dict:
        """Return config as a dict with secrets redacted — safe to log."""
        d = self.model_dump()
        d["llm_api_key"] = "***"
        d["guardrail_token"] = "***"
        return d
