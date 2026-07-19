"""
Mock tool executors for the agentic guardrail demo.

These simulate what a real integration would return WITHOUT touching a
real database, the network, or a shell. This package demonstrates the
guardrail that sits in FRONT of tool execution — wiring a real DB driver,
HTTP client, or subprocess call is a deliberate non-goal: a "guardrail
demo" that actually shells out to an approved command would turn a
security demo into a real attack surface. Swap these for real
integrations once the guardrail layer itself is trusted.
"""

from __future__ import annotations

from guardrail.tools.schema import (
    ExecuteDbQueryParams,
    FetchExternalUrlParams,
    GetWeatherParams,
    RunSystemCommandParams,
)

_MOCK_WEATHER = {
    "san francisco": "62°F, foggy",
    "new york": "71°F, partly cloudy",
    "london": "58°F, light rain",
    "tokyo": "75°F, clear",
}


def get_weather(params: GetWeatherParams) -> str:
    key = params.location.strip().lower()
    condition = _MOCK_WEATHER.get(key, "68°F, clear (simulated default)")
    return f"Weather in {params.location}: {condition}"


def execute_db_query(params: ExecuteDbQueryParams) -> str:
    return (
        f"[MOCK DB] Executed: {params.query}\n"
        "3 rows returned (simulated) — columns: id, name, created_at"
    )


def fetch_external_url(params: FetchExternalUrlParams) -> str:
    return f"[MOCK HTTP {params.method}] {params.url} -> 200 OK, 1.2KB (simulated response body)"


def run_system_command(params: RunSystemCommandParams) -> str:
    return f"[MOCK SHELL] $ {params.command}\n(simulated — no real command was executed)"
