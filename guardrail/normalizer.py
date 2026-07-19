"""
Normalizer: text preprocessing that runs before hashing and detection.

Day 1: two transforms only —
  1. Zero-width character stripping (U+200B, U+FEFF, etc.)
  2. Homoglyph normalisation via Unicode NFKC decomposition

Days 2+ will extend this with more aggressive homoglyph tables,
confusable character mapping, and whitespace canonicalisation.

Design: normalise() is a pure function (str → str) with no side effects
so it's safe to call from both the cache key builder and the detectors.

try_decode_base64() is a separate, opt-in helper (not part of normalise())
because base64 decoding changes what the LLM should actually receive —
unlike NFKC/zero-width stripping, which are cosmetic equivalences kept
internal to detection, decoded text must also replace what's forwarded
to the LLM. The caller (middleware) is responsible for that substitution.
"""

from __future__ import annotations

import base64
import binascii
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


# Strict base64 alphabet, correct padding (0 or 2 or 3 padding chars per
# RFC 4648 groups of 4).
_BASE64_PATTERN = re.compile(
    r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{4})$"
)

# Below this length, too many short alphanumeric strings are technically
# valid base64 by chance — not worth the false-positive risk.
_MIN_BASE64_LEN = 16

# Decoded bytes must be "mostly" printable text to count as a real base64
# evasion attempt rather than a binary blob that just happens to be valid
# base64 (images, keys, etc. — those aren't something we can usefully scan
# or forward as text anyway).
_MIN_PRINTABLE_RATIO = 0.95


def try_decode_base64(text: str) -> str | None:
    """
    If the ENTIRE (trimmed) `text` is valid base64 that decodes to
    printable UTF-8 text, return the decoded string. Otherwise return None.

    This exists to close a guardrail bypass: base64-encoded PII, secrets,
    or a prompt injection reads as opaque noise to both Tier 1 (regex/
    Presidio) and Tier 2 (semantic model) — decoding first means detection
    sees what the text actually says.
    """
    candidate = text.strip()
    if len(candidate) < _MIN_BASE64_LEN or len(candidate) % 4 != 0:
        return None
    if not _BASE64_PATTERN.match(candidate):
        return None

    try:
        decoded_bytes = base64.b64decode(candidate, validate=True)
    except (binascii.Error, ValueError):
        return None

    try:
        decoded_text = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None

    if not decoded_text.strip():
        return None

    printable = sum(1 for c in decoded_text if c.isprintable() or c in "\n\r\t")
    if printable / len(decoded_text) < _MIN_PRINTABLE_RATIO:
        return None

    return decoded_text
