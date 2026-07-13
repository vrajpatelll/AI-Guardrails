"""
Redaction engine: replaces detected spans with typed redaction tokens.

Design decisions:
- Replacements are applied in REVERSE span order (highest start offset
  first) so earlier span offsets remain valid after each substitution.
- Overlapping spans from the same category are merged before replacement
  to avoid double-substitution artifacts.
- Redaction tokens follow the pattern <ENTITY_TYPE_REDACTED>, e.g.:
    EMAIL_ADDRESS   -> <EMAIL_ADDRESS_REDACTED>
    AWS_ACCESS_KEY  -> <AWS_ACCESS_KEY_REDACTED>
- The function is pure (no side effects) and returns the sanitized string.

Public API
----------
    sanitized = build_redacted_text(text, detections)
    # detections: list[DetectionResult] from tier1.py
"""

from __future__ import annotations

from guardrail.detectors.tier1 import DetectionResult


def _merge_overlapping(
    detections: list[DetectionResult],
) -> list[tuple[int, int, str]]:
    """
    Merge overlapping spans and return (start, end, redaction_token) triples.

    When two detections overlap, we union their spans and use the token from
    the first (highest-confidence) detection.

    Detections are expected to be pre-sorted by start offset.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: (d.start, -d.confidence))

    merged: list[tuple[int, int, str]] = []
    cur_start = sorted_dets[0].start
    cur_end = sorted_dets[0].end
    cur_token = f"<{sorted_dets[0].category}_REDACTED>"

    for det in sorted_dets[1:]:
        token = f"<{det.category}_REDACTED>"
        if det.start < cur_end:
            # Overlapping - extend the current span
            cur_end = max(cur_end, det.end)
        else:
            merged.append((cur_start, cur_end, cur_token))
            cur_start = det.start
            cur_end = det.end
            cur_token = token

    merged.append((cur_start, cur_end, cur_token))
    return merged


def build_redacted_text(
    text: str,
    detections: list[DetectionResult],
) -> str:
    """
    Replace each detected span in `text` with a typed redaction token.

    Args:
        text: The normalised text (same string the spans refer to).
        detections: All detections to redact, from any category.
                    Mixed PII + secrets is fine - each gets its own token.

    Returns:
        A new string with all detected spans replaced.
        Returns `text` unchanged if `detections` is empty.

    Example::
        text = "Contact alice@example.com or call 555-123-4567"
        dets = [
            DetectionResult("EMAIL_ADDRESS", 8, 25, 0.99, "..."),
            DetectionResult("PHONE_NUMBER", 34, 46, 0.95, "..."),
        ]
        result = build_redacted_text(text, dets)
        # -> "Contact <EMAIL_ADDRESS_REDACTED> or call <PHONE_NUMBER_REDACTED>"
    """
    if not detections:
        return text

    merged = _merge_overlapping(detections)

    # Apply replacements in REVERSE order so start/end offsets stay valid
    result = text
    for start, end, token in reversed(merged):
        result = result[:start] + token + result[end:]

    return result
