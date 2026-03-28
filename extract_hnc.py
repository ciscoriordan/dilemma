#!/usr/bin/env python3
"""Extract form->lemma pairs from the HNC Golden Corpus.

The Hellenic National Corpus (HNC) Gold Standard is a POS-tagged and
lemmatized Modern Greek corpus from CLARIN:EL. It contains 88K tokens
across 65 concatenated XML documents.

Source: https://inventory.clarin.gr/corpus/870
License: openUnder-PSI (Public Sector Information)

Outputs:
    data/hnc_pairs.json - {form: lemma} dict of clean MG form->lemma pairs

Usage:
    python extract_hnc.py
"""

import json
import re
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
HNC_PATH = DATA_DIR / "HNC_Golden_Corpus.xml"
OUTPUT_PATH = DATA_DIR / "hnc_pairs.json"

# Tags to skip (noise, not real word tokens)
SKIP_TAGS = {"ABBR", "DIG", "DATE", "INIT", "RgSyXx", "RgAbXx", "PuXx"}

# Regex for non-Greek noise: URLs, hashtags, email addresses, numbers
_NOISE_RE = re.compile(
    r"^(?:"
    r"https?://|www\.|"           # URLs
    r"#|@|"                       # hashtags, mentions
    r"\d|"                        # starts with digit
    r"[a-zA-Z]|"                  # Latin characters
    r"[^\w]"                      # non-word characters
    r")",
    re.UNICODE,
)


def _is_greek(s: str) -> bool:
    """Check if string contains Greek characters."""
    return any(
        "\u0370" <= c <= "\u03FF"       # Greek and Coptic
        or "\u1F00" <= c <= "\u1FFF"    # Greek Extended
        for c in s
    )


def _is_noise(word: str) -> bool:
    """Filter out URLs, hashtags, numbers, Latin-only tokens, punctuation."""
    if not word:
        return True
    # Single character (punctuation, etc.)
    if len(word) == 1 and not _is_greek(word):
        return True
    # Must contain at least one Greek character
    if not _is_greek(word):
        return True
    # URLs, hashtags, etc.
    if _NOISE_RE.match(word):
        # But allow Greek words that happen to start with a digit-like char
        if not _is_greek(word):
            return True
    return False


def extract_hnc_pairs(hnc_path: Path = HNC_PATH) -> dict[str, str]:
    """Parse HNC XML and extract form->lemma pairs.

    Returns:
        Dict mapping lowercase surface form to lemma.
    """
    content = hnc_path.read_text(encoding="utf-8")

    # Parse tokens with regex (file has 65 concatenated <document> elements)
    token_re = re.compile(
        r'<t\s+[^>]*?word="([^"]+)"\s+tag="([^"]+)"\s+lemma="([^"]+)"'
    )

    pairs = {}  # form -> lemma
    total = 0
    skipped_tag = 0
    skipped_noise = 0

    for match in token_re.finditer(content):
        word, tag, lemma = match.groups()
        total += 1

        # Skip non-word tags
        if tag in SKIP_TAGS:
            skipped_tag += 1
            continue

        # Skip noise
        if _is_noise(word) or _is_noise(lemma):
            skipped_noise += 1
            continue

        # Normalize: lowercase the surface form (HNC has mixed case)
        form = word.lower()
        lemma = lemma.strip()

        if not form or not lemma:
            continue

        # Skip self-maps (form == lemma, case-insensitive)
        # These don't add lookup value
        if form == lemma.lower():
            continue

        # First occurrence wins (gold standard, so all are correct,
        # but we keep the first in case of duplicates)
        if form not in pairs:
            pairs[form] = lemma

    print(f"HNC Golden Corpus:")
    print(f"  Total tokens: {total:,}")
    print(f"  Skipped (tag): {skipped_tag:,}")
    print(f"  Skipped (noise): {skipped_noise:,}")
    print(f"  Unique form->lemma pairs (non-self-map): {len(pairs):,}")

    return pairs


def main():
    if not HNC_PATH.exists():
        print(f"Error: {HNC_PATH} not found")
        print("Download from https://inventory.clarin.gr/corpus/870")
        return

    pairs = extract_hnc_pairs()

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  -> {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
