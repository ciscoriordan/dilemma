#!/usr/bin/env python3
"""Build lsj9_pos_lookup.json from lsj9 headword POS data.

Reads lsj9_headword_pos.json (from the lsj9 repo) which maps headwords to
UPOS tags, then cross-references with ag_lookup.json to find all inflected
forms for those headwords. Only includes forms that are genuinely useful for
POS disambiguation (i.e., NOUN and ADJ entries, since these are the
grammar-derived categories from lsj9).

Output: data/lsj9_pos_lookup.json
Format: {form: {UPOS: headword}, ...}

Usage:
    python build/build_lsj9_pos_lookup.py
"""

import json
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
AG_LOOKUP_PATH = DATA_DIR / "ag_lookup.json"
LSJ9_DIR = Path.home() / "Documents" / "lsj9"
HEADWORD_POS_PATH = LSJ9_DIR / "lsj9_headword_pos.json"
OUTPUT_PATH = DATA_DIR / "lsj9_pos_lookup.json"


def _strip_diacritics(s: str) -> str:
    """Strip all combining diacritics for accent-free comparison."""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def _strip_length_marks(s: str) -> str:
    """Strip combining breve (U+0306) and macron (U+0304)."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if ord(c) not in (0x0306, 0x0304)))


def main():
    print("Building lsj9_pos_lookup.json")

    # Load headword -> UPOS mapping from lsj9
    if not HEADWORD_POS_PATH.exists():
        print(f"  Error: {HEADWORD_POS_PATH} not found")
        print("  Run build_exports.py in the lsj9 repo first.")
        return

    with open(HEADWORD_POS_PATH, encoding="utf-8") as f:
        headword_pos = json.load(f)
    print(f"  Loaded {len(headword_pos):,} headword POS entries")

    # Only include NOUN and ADJ - these are the grammar-derived categories
    # that provide genuine disambiguation value. VERB entries from lsj9 are
    # heuristic-based (ending detection) and less reliable.
    target_pos = {"NOUN", "ADJ"}
    hw_pos_filtered = {hw: upos for hw, upos in headword_pos.items()
                       if upos in target_pos}
    print(f"  Filtered to {len(hw_pos_filtered):,} NOUN/ADJ headwords")

    # Load ag_lookup (form -> headword)
    if not AG_LOOKUP_PATH.exists():
        print(f"  Error: {AG_LOOKUP_PATH} not found")
        return

    with open(AG_LOOKUP_PATH, encoding="utf-8") as f:
        ag_lookup = json.load(f)
    print(f"  Loaded {len(ag_lookup):,} ag_lookup entries")

    # Build reverse index: headword -> set of forms
    # Then for headwords with known POS, create POS lookup entries
    pos_lookup = {}
    for form, lemma in ag_lookup.items():
        lemma_clean = _strip_length_marks(lemma)
        upos = hw_pos_filtered.get(lemma_clean)
        if not upos:
            continue

        # Add accented form
        if form not in pos_lookup:
            pos_lookup[form] = {}
        pos_lookup[form][upos] = lemma_clean

        # Add accent-stripped form
        plain = _strip_diacritics(form)
        if plain != form:
            if plain not in pos_lookup:
                pos_lookup[plain] = {}
            pos_lookup[plain][upos] = lemma_clean

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pos_lookup, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  Output: {len(pos_lookup):,} form entries ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
