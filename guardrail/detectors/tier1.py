"""
Tier 1 deterministic detector: PII (Presidio) + Secrets (regex).

Both sub-detectors operate on the already-normalised text.
Span offsets in DetectionResult.start/end refer to positions in that
normalised string - consistent with how the cache key is built.

Public API
----------
    detector = Tier1Detector()                 # loads Presidio once at startup
    results = detector.run(normalised_text)    # returns Tier1Results
    results.pii      -> list[DetectionResult]
    results.secrets  -> list[DetectionResult]

Thread safety: Tier1Detector instances are safe to share across threads.
Presidio's AnalyzerEngine is thread-safe after initialisation.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

logger = logging.getLogger(__name__)

# Presidio - imported so the module can be imported even if presidio
# isn't installed (ImportError only when Tier1Detector() is instantiated).
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    _PRESIDIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PRESIDIO_AVAILABLE = False

from guardrail.detectors.patterns import CATALOG, SecretPattern

if TYPE_CHECKING:
    from guardrail.policy import PolicyConfig


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """A single matched entity - normalised output from both sub-detectors."""
    category: str           # e.g. "EMAIL_ADDRESS", "AWS_ACCESS_KEY"
    start: int              # char offset in normalised text (inclusive)
    end: int                # char offset in normalised text (exclusive)
    confidence: float       # 0.0-1.0
    rule: str               # e.g. "presidio.email_recognizer"

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclass
class Tier1Results:
    """Aggregated output from one Tier1Detector.run() call."""
    pii: list[DetectionResult] = field(default_factory=list)
    secrets: list[DetectionResult] = field(default_factory=list)

    def any_found(self) -> bool:
        return bool(self.pii or self.secrets)


# ---------------------------------------------------------------------------
# PII sub-detector (Presidio)
# ---------------------------------------------------------------------------

# The set of PII entity types we recognise.
# IMPORTANT: Only include recognizers with structural anchors (regex/
# checksum-backed). Exclude pure-NLP/spaCy recognizers (DATE_TIME,
# LOCATION, PERSON, NRP) - they fire on normal text like "today" or
# common first names and produce false positives in a guardrail context.
_PII_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "MEDICAL_LICENSE",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
    "US_BANK_NUMBER",
    "CRYPTO",           # crypto wallet addresses
)


class PiiDetector:
    """
    Wraps Presidio AnalyzerEngine to detect PII entities.

    Initialisation is expensive (~1-2s on first call due to spaCy model
    loading). Create one instance per process and reuse it.
    """

    def __init__(self) -> None:
        if not _PRESIDIO_AVAILABLE:
            raise ImportError(
                "presidio-analyzer is not installed. "
                "Run: pip install presidio-analyzer presidio-anonymizer spacy "
                "&& python -m spacy download en_core_web_lg"
            )
        t0 = time.monotonic()
        logger.info("Tier 1: loading Presidio AnalyzerEngine (spaCy model)…")
        self._engine = AnalyzerEngine()
        logger.info("Tier 1: AnalyzerEngine ready (%.0fms)", (time.monotonic() - t0) * 1000)

    def run(
        self,
        text: str,
        language: str = "en",
        entities: Sequence[str] | None = None,
        score_threshold: float = 0.0,
    ) -> list[DetectionResult]:
        """
        Analyse `text` and return all detected PII entities.

        Args:
            text: Normalised input text.
            language: Language code for Presidio (default "en").
            entities: Override the default entity list.
            score_threshold: Minimum Presidio score (we apply policy threshold
                             separately so keep this low).

        Returns:
            List of DetectionResult sorted by start offset.
        """
        requested_entities = list(entities) if entities else list(_PII_ENTITIES)

        try:
            raw: list[RecognizerResult] = self._engine.analyze(
                text=text,
                language=language,
                entities=requested_entities,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            raise RuntimeError(f"Presidio analysis failed: {exc}") from exc

        results: list[DetectionResult] = []
        for r in raw:
            rule = (
                f"presidio.{r.recognition_metadata.get('recognizer_name', 'unknown').lower()}"
                if r.recognition_metadata
                else f"presidio.{r.entity_type.lower()}"
            )
            results.append(DetectionResult(
                category=r.entity_type,
                start=r.start,
                end=r.end,
                confidence=round(r.score, 4),
                rule=rule,
            ))

        results.sort(key=lambda d: d.start)
        return results


# ---------------------------------------------------------------------------
# Secrets sub-detector (regex)
# ---------------------------------------------------------------------------

class SecretsDetector:
    """
    Pure-regex secrets detector. No ML, no external dependencies.
    Scans the CATALOG of SecretPattern entries from patterns.py.
    """

    def __init__(self, extra_patterns: list[SecretPattern] | None = None) -> None:
        self._patterns: list[SecretPattern] = list(CATALOG)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def run(self, text: str) -> list[DetectionResult]:
        """
        Scan `text` against all secret patterns in the catalog.

        Returns:
            List of DetectionResult sorted by start offset.
        """
        results: list[DetectionResult] = []
        for pattern in self._patterns:
            for match in pattern.pattern.finditer(text):
                # If the pattern has a capture group (e.g. label + value),
                # report the span of group 1 (the secret value itself).
                # If no groups, report the full match span.
                if match.lastindex and match.lastindex >= 1:
                    start, end = match.span(1)
                else:
                    start, end = match.span(0)

                results.append(DetectionResult(
                    category=pattern.name,
                    start=start,
                    end=end,
                    confidence=pattern.confidence,
                    rule=pattern.rule,
                ))

        results.sort(key=lambda d: d.start)
        return results


# ---------------------------------------------------------------------------
# Deny-list sub-detector (policy-driven keywords/regex)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _compile_deny_pattern(raw: str) -> re.Pattern[str]:
    """
    Compile one policy.yaml `deny_patterns` entry.

    Most org-specific entries are plain words/phrases (internal codenames,
    custom ID prefixes) rather than intentional regex, so a string that
    isn't valid regex (unbalanced brackets, etc.) is treated as a literal
    via re.escape() instead of raising - a typo in policy.yaml shouldn't
    take down the detector. Case-insensitive either way.
    """
    try:
        return re.compile(raw, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(raw), re.IGNORECASE)


class DenyListDetector:
    """
    Deterministic keyword/regex matcher for `CategoryPolicy.deny_patterns` -
    org-specific terms (internal codenames, custom ID formats) that Presidio
    and the built-in secrets catalog don't know about and can't be taught
    without a code change. Every match is confidence=1.0: there's no model
    uncertainty in a literal/regex match, unlike Presidio's PII scores.
    """

    def run(self, text: str, category: str, patterns: Sequence[str]) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        for raw in patterns:
            compiled = _compile_deny_pattern(raw)
            for match in compiled.finditer(text):
                results.append(DetectionResult(
                    category=f"DENY_LIST:{raw}",
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                    rule=f"policy.deny_pattern[{category}]",
                ))
        results.sort(key=lambda d: d.start)
        return results


# ---------------------------------------------------------------------------
# Tier 1 orchestrator
# ---------------------------------------------------------------------------

class Tier1Detector:
    """
    Runs PII detection (Presidio) and secrets detection (regex) on the
    same normalised text and returns combined Tier1Results.

    Both detectors always run - they check independent categories and
    neither result gates the other.
    """

    def __init__(
        self,
        extra_secret_patterns: list[SecretPattern] | None = None,
    ) -> None:
        self._pii = PiiDetector()
        self._secrets = SecretsDetector(extra_patterns=extra_secret_patterns)
        self._deny_list = DenyListDetector()

    def run(
        self,
        normalised_text: str,
        policy: "PolicyConfig | None" = None,
    ) -> Tier1Results:
        """
        Run both sub-detectors on `normalised_text`.

        Args:
            normalised_text: Text that has already been through normalise().
            policy: Optional PolicyConfig. When given, each category's
                    deny_patterns (pii, secrets) are also matched and folded
                    into that category's result list alongside the built-in
                    detections. Passed per-call (not baked in at construction)
                    so policy hot-reloads take effect immediately, same as
                    every other threshold/action in policy.yaml.

        Returns:
            Tier1Results with pii and secrets lists.
        """
        t0 = time.monotonic()
        pii_hits = self._pii.run(normalised_text)
        secret_hits = self._secrets.run(normalised_text)

        if policy is not None:
            pii_policy = policy.categories.get("pii")
            if pii_policy and pii_policy.deny_patterns:
                pii_hits = pii_hits + self._deny_list.run(
                    normalised_text, "pii", pii_policy.deny_patterns
                )
                pii_hits.sort(key=lambda d: d.start)

            secrets_policy = policy.categories.get("secrets")
            if secrets_policy and secrets_policy.deny_patterns:
                secret_hits = secret_hits + self._deny_list.run(
                    normalised_text, "secrets", secrets_policy.deny_patterns
                )
                secret_hits.sort(key=lambda d: d.start)

        logger.info(
            "Tier 1: result pii=%d secrets=%d (%.0fms)",
            len(pii_hits), len(secret_hits), (time.monotonic() - t0) * 1000,
        )

        return Tier1Results(pii=pii_hits, secrets=secret_hits)
