"""Greek sentence segmentation.

Splits Greek text into sentences on sentence-ending punctuation while
respecting abbreviations, quoted text, and Greek-specific punctuation
conventions.
"""

import re

# Modern Greek abbreviations (lowercase, without trailing period)
_MG_ABBREVS = {
    "κ", "π.χ", "κλπ", "δηλ", "βλ", "σ", "σελ", "αρ", "τ",
    "κ.ά", "κ.τ.λ", "κ.λπ", "εκ", "αι",
}

# Ancient Greek / LSJ citation abbreviations
_AG_ABBREVS = {
    "cf", "v", "sq", "al", "sc", "l.c", "s.v", "q.v",
    "e.g", "i.e", "etc",
}

_ALL_ABBREVS = _MG_ABBREVS | _AG_ABBREVS

# Sentence-ending punctuation: period, Greek question mark (;),
# ano teleia (middle dot), exclamation mark
_SENTENCE_END = re.compile(r'[.;\u00B7!]')

# Quote characters (Greek and Latin)
_OPEN_QUOTES = set('"\u00AB\u201C\u2018')   # " << left-dq left-sq
_CLOSE_QUOTES = set('"\u00BB\u201D\u2019')  # " >> right-dq right-sq


def _is_single_uppercase(token: str) -> bool:
    """Check if token is a single uppercase letter (author abbreviation)."""
    return len(token) == 1 and token.isupper()


def segment(text: str) -> list[str]:
    """Split Greek text into sentences.

    Splits on sentence-ending punctuation (. ; middle-dot !) while
    preserving abbreviations, handling quoted text, and normalizing
    whitespace.

    Args:
        text: Input text, potentially containing multiple sentences.

    Returns:
        List of trimmed, non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace (collapse runs, strip leading/trailing)
    text = re.sub(r'\s+', ' ', text.strip())

    sentences = []
    current = []
    quote_depth = 0
    i = 0
    chars = list(text)
    n = len(chars)

    while i < n:
        ch = chars[i]

        # Track quote depth
        if ch in _OPEN_QUOTES:
            quote_depth += 1
            current.append(ch)
            i += 1
            continue
        if ch in _CLOSE_QUOTES:
            quote_depth = max(0, quote_depth - 1)
            current.append(ch)
            i += 1
            continue

        # Check for sentence-ending punctuation
        if _SENTENCE_END.match(ch) and quote_depth == 0:
            # For periods, check if this is an abbreviation
            if ch == '.':
                # Extract the token immediately before this period
                pre = ''.join(current).rstrip()
                # Find the last whitespace to isolate the preceding token
                last_space = pre.rfind(' ')
                token_before = pre[last_space + 1:] if last_space >= 0 else pre

                # Check for single uppercase letter (author abbreviation)
                if _is_single_uppercase(token_before):
                    current.append(ch)
                    i += 1
                    continue

                # Check for known multi-part abbreviations
                # Build the candidate by looking back through dots
                # e.g., for "κ.τ.λ" we need to match the full form
                candidate = token_before.lower()
                if candidate in _ALL_ABBREVS:
                    current.append(ch)
                    i += 1
                    continue

                # Check if this period is part of a multi-dot abbreviation
                # that is still being built, e.g., "κ." in "κ.τ.λ."
                # Look ahead to see if more abbreviation text follows
                if i + 1 < n and chars[i + 1] not in (' ', '\t', '\n') and chars[i + 1] not in _OPEN_QUOTES | _CLOSE_QUOTES:
                    # Period followed immediately by non-space - likely part
                    # of an abbreviation like "π.χ." or "κ.τ.λ."
                    current.append(ch)
                    i += 1
                    continue

                # Also check compound abbreviations where the token includes
                # internal dots, e.g., token_before might be "π.χ" or "κ.τ.λ"
                # (already handled above via _ALL_ABBREVS)

            # This is a sentence boundary
            current.append(ch)
            # Consume any additional sentence-ending punctuation
            i += 1
            while i < n and _SENTENCE_END.match(chars[i]):
                current.append(chars[i])
                i += 1

            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
            # Skip whitespace after sentence boundary
            while i < n and chars[i] == ' ':
                i += 1
            continue

        current.append(ch)
        i += 1

    # Handle remaining text (no trailing punctuation)
    remainder = ''.join(current).strip()
    if remainder:
        sentences.append(remainder)

    return sentences
