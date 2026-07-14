"""
Policy loader: reads policy.yaml → PolicyConfig Pydantic model.

Design notes:
- policy_version is an auto-incrementing integer kept in memory.
  It increments every time policy.yaml is (re)loaded so the cache
  key hash(normalized_text + str(policy_version)) is automatically
  invalidated on any policy change.
- CategoryPolicy.action = "block" takes precedence over "redact" when
  the verdict combiner merges results from both tiers (enforced by the
  verdict combiner, not here — the loader just surfaces the config).
- load_policy() is intentionally synchronous; it is only called at
  startup and on explicit reload, never in the hot request path.
"""

from __future__ import annotations

import threading
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


# ---------------------------------------------------------------------------
# Enums / value objects
# ---------------------------------------------------------------------------

class PolicyAction(str, Enum):
    BLOCK = "block"
    REDACT = "redact"
    ALLOW = "allow"


class OnTimeout(str, Enum):
    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"


# ---------------------------------------------------------------------------
# Per-category config
# ---------------------------------------------------------------------------

class CategoryPolicy(BaseModel):
    enabled: bool = True
    action: PolicyAction = PolicyAction.BLOCK
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Optional deny-list: org-specific keywords / regex strings
    deny_patterns: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level policy config
# ---------------------------------------------------------------------------

class PolicyConfig(BaseModel):
    name: str
    # dict keyed by category name: pii, secrets, harmful_content, prompt_injection, ...
    categories: dict[str, CategoryPolicy] = Field(default_factory=dict)
    latency_budget_ms: int = 150
    on_timeout: OnTimeout = OnTimeout.FAIL_OPEN
    # Populated by the loader after parsing — never comes from the YAML file
    policy_version: int = 0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_version_lock = threading.Lock()
_current_version: int = 0


def _next_version() -> int:
    global _current_version
    with _version_lock:
        _current_version += 1
        return _current_version


def load_policy(path: str | Path) -> PolicyConfig:
    """
    Parse `path` (YAML) into a PolicyConfig.

    Each call increments the in-memory policy_version counter so the
    cache key changes and stale verdicts are never served.

    Raises:
        FileNotFoundError: if the policy file does not exist.
        ValueError: if required fields are missing or invalid.
    """
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    with policy_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if not raw:
        raise ValueError(f"Policy file is empty: {policy_path}")

    # Normalise category entries from raw YAML into CategoryPolicy objects
    raw_categories: dict[str, Any] = raw.pop("categories", {})
    parsed_categories: dict[str, CategoryPolicy] = {
        name: CategoryPolicy(**cat_data)
        for name, cat_data in raw_categories.items()
    }

    config = PolicyConfig(
        categories=parsed_categories,
        **raw,
    )
    config.policy_version = _next_version()
    return config


def get_category(config: PolicyConfig, name: str) -> CategoryPolicy | None:
    """Return the CategoryPolicy for `name`, or None if not defined / disabled."""
    cat = config.categories.get(name)
    if cat is None or not cat.enabled:
        return None
    return cat


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

class PolicyWatcher(FileSystemEventHandler):
    """
    Watches the policy file and triggers a reload on modification.
    """

    def __init__(self, policy_path: str | Path, reload_callback: Any) -> None:
        self.policy_path = Path(policy_path).resolve()
        self.reload_callback = reload_callback
        self._observer = Observer()

    def start(self) -> None:
        """Start the background observer thread."""
        watch_dir = self.policy_path.parent
        # watchdog requires a string path
        self._observer.schedule(self, str(watch_dir), recursive=False)
        self._observer.start()

    def stop(self) -> None:
        """Stop the background observer thread."""
        self._observer.stop()
        self._observer.join()

    def on_modified(self, event: FileSystemEvent) -> None:
        # Check if the modified file is exactly our policy file
        # watchdog paths can vary slightly based on OS, so resolve both
        try:
            modified_path = Path(event.src_path).resolve()
            if modified_path == self.policy_path:
                self.reload_callback()
        except Exception:
            pass  # File might have been deleted mid-event, ignore
