"""
Verdict cache: hash(normalized_text + str(policy_version)) → GuardrailResponse.

Design notes:
- Cache key is a SHA-256 hex digest of (normalized_text + policy_version)
  so any policy reload (which bumps policy_version) automatically
  invalidates all cached verdicts — no explicit flush needed.
- Backend is an in-memory dict behind a threading.Lock for now.
  Swapping to Redis in Day 3 only requires replacing _store with a
  Redis client and updating get/set — the interface stays identical.
- TTL is enforced at read time (check timestamp on hit).
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

from guardrail.schema import GuardrailResponse


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    response: GuardrailResponse
    stored_at: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

class VerdictCache:
    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(normalized_text: str, policy_version: int) -> str:
        """
        Deterministic cache key.

        Including policy_version means any policy.yaml reload
        (which bumps the version counter) produces a new key and the
        old entry is simply never hit again — no explicit invalidation.
        """
        raw = f"{normalized_text}||{policy_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> GuardrailResponse | None:
        """Return a cached response if it exists and hasn't expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() - entry.stored_at > self.ttl_seconds:
                del self._store[key]
                return None
            return entry.response

    def set(self, key: str, response: GuardrailResponse) -> None:
        """Store a verdict response under `key`."""
        with self._lock:
            self._store[key] = _CacheEntry(response=response)

    def invalidate(self, key: str) -> None:
        """Explicitly remove a single entry (rarely needed given key design)."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Flush the entire cache (useful in tests)."""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)
