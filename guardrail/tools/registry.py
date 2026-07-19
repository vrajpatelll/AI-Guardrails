"""
Tool registry: maps a tool name to its risk tier, Pydantic parameter model,
and mock executor. guardrail/tools/middleware.py looks tools up here to
know which parameter schema applies and what a clean call defaults to.

To register a new tool: add a Pydantic parameter model to schema.py, an
executor function to mock_tools.py, wire any tool-specific check into
middleware.py's _param_specific_checks(), and add an entry here — no other
code changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from guardrail.tools import mock_tools
from guardrail.tools.schema import (
    ExecuteDbQueryParams,
    FetchExternalUrlParams,
    GetWeatherParams,
    RiskTier,
    RunSystemCommandParams,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    param_model: type[BaseModel]
    risk_tier: RiskTier
    executor: Callable[[BaseModel], str]
    # Tools whose risk tier alone must always get human review before
    # executing, regardless of whether the parameters look clean.
    always_requires_approval: bool = False


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "get_weather": ToolSpec(
        name="get_weather",
        description="Get the current weather for a location. Read-only, no side effects.",
        param_model=GetWeatherParams,
        risk_tier=RiskTier.LOW,
        executor=mock_tools.get_weather,
    ),
    "execute_db_query": ToolSpec(
        name="execute_db_query",
        description="Run a read-only SQL SELECT query against the reporting database.",
        param_model=ExecuteDbQueryParams,
        risk_tier=RiskTier.MEDIUM,
        executor=mock_tools.execute_db_query,
    ),
    "fetch_external_url": ToolSpec(
        name="fetch_external_url",
        description="Fetch content from an external URL. Allowlisted domains only.",
        param_model=FetchExternalUrlParams,
        risk_tier=RiskTier.MEDIUM,
        executor=mock_tools.fetch_external_url,
    ),
    "run_system_command": ToolSpec(
        name="run_system_command",
        description="Run a shell command on the host. High-risk — always requires human approval.",
        param_model=RunSystemCommandParams,
        risk_tier=RiskTier.HIGH,
        executor=mock_tools.run_system_command,
        always_requires_approval=True,
    ),
}


# ---------------------------------------------------------------------------
# OpenAI function-calling schema — for wiring TOOL_REGISTRY into a real
# chat.completions(..., tools=[...]) call (see examples/agentic_tools_demo.py)
# ---------------------------------------------------------------------------

def openai_tool_schemas() -> list[dict]:
    """Build the `tools=[...]` payload OpenAI-compatible chat APIs expect."""
    schemas = []
    for spec in TOOL_REGISTRY.values():
        schema = spec.param_model.model_json_schema()
        schema.pop("title", None)
        schemas.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": schema,
            },
        })
    return schemas
