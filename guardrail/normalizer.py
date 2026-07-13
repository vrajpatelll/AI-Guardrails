"""
Normalizer: text preprocessing that runs before hashing and detection.

Day 1: two transforms only —
  1. Zero-width character stripping (U+200B, U+FEFF, etc.)
  2. Homoglyph normalisation via Unicode NFKC decomposition

Days 2+ will extend this with more aggressive homoglyph tables,
confusable character mapping, and whitespace canonicalisation.

Design: normalise() is a pure function (str → str) with no side effects
so it's safe to call from both the cache key builder and the detectors.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and invisible characters to strip
_ZERO_WIDTH_PATTERN = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f"   # zero-width space/non-joiner/joiner/LRM/RLM
    r"\u00ad"                             # soft hyphen
    r"\ufeff"                             # BOM / zero-width no-break space
    r"\u2060\u2061\u2062\u2063\u2064"    # word joiner + invisible math operators
    r"\u206a-\u206f]",                    # deprecated formatting characters
    re.UNICODE,
)


def normalise(text: str) -> str:
    """
    Return a normalised version of `text` suitable for hashing and detection.

    Steps:
    1. Strip zero-width / invisible characters.
    2. Apply Unicode NFKC normalisation (collapses many homoglyphs and
       compatibility equivalents — e.g. ① → 1, ﬁ → fi, ｅ → e).

    The original text is never mutated; a new string is returned.
    """
    # Step 1: remove zero-width / invisible chars
    stripped = _ZERO_WIDTH_PATTERN.sub("", text)

    # Step 2: NFKC normalisation (handles most homoglyph obfuscation)
    normalised = unicodedata.normalize("NFKC", stripped)

    return normalised
