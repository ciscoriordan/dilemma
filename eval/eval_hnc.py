#!/usr/bin/env python3
"""Evaluate Dilemma against HNC Golden Corpus (Modern Greek).

The Hellenic National Corpus Gold Standard contains 88K tokens of
POS-tagged and lemmatized Modern Greek text from CLARIN:EL.

Reports lemmatization accuracy: how often Dilemma's MG lookup table
returns the same lemma as the HNC gold annotation.

Source: https://inventory.clarin.gr/corpus/870
License: openUnder-PSI (Public Sector Information)

Usage:
    python eval/eval_hnc.py
    python eval/eval_hnc.py --breakdown    # show error breakdown
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
HNC_PATH = DATA_DIR / "HNC_Golden_Corpus.xml"
DB_PATH = DATA_DIR / "lookup.db"

# Tags to skip (same as extract_hnc.py)
SKIP_TAGS = {"ABBR", "DIG", "DATE", "INIT", "RgSyXx", "RgAbXx", "PuXx"}


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def _is_greek(s: str) -> bool:
    return any(
        "\u0370" <= c <= "\u03FF" or "\u1F00" <= c <= "\u1FFF"
        for c in s
    )


def _is_noise(word: str) -> bool:
    if not word or len(word) == 1:
        return True
    if not _is_greek(word):
        return True
    return False


def load_dilemma_lookup():
    """Load Dilemma's lookup table from SQLite."""
    import sqlite3
    if not DB_PATH.exists():
        print(f"Error: {DB_PATH} not found. Run build_lookup_db.py first.")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA mmap_size=268435456")

    # Load lemma table
    lemmas = {}
    for lid, text in conn.execute("SELECT id, text FROM lemmas"):
        lemmas[lid] = text

    # Load MG-priority lookups: lang='el' first, then lang='all'
    lookup = {}
    # el-specific entries (override combined)
    for form, lemma_id in conn.execute(
            "SELECT form, lemma_id FROM lookup WHERE lang='el'"):
        lookup[form] = lemmas[lemma_id]
    # Combined entries (fill gaps)
    for form, lemma_id in conn.execute(
            "SELECT form, lemma_id FROM lookup WHERE lang='all'"):
        if form not in lookup:
            lookup[form] = lemmas[lemma_id]

    conn.close()
    return lookup


def parse_hnc_tokens(hnc_path: Path):
    """Parse HNC XML and yield (word, tag, lemma) tuples."""
    content = hnc_path.read_text(encoding="utf-8")
    token_re = re.compile(
        r'<t\s+[^>]*?word="([^"]+)"\s+tag="([^"]+)"\s+lemma="([^"]+)"'
    )
    for match in token_re.finditer(content):
        yield match.groups()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Dilemma against HNC Golden Corpus")
    parser.add_argument("--breakdown", action="store_true",
                        help="Show error category breakdown")
    args = parser.parse_args()

    if not HNC_PATH.exists():
        print(f"Error: {HNC_PATH} not found")
        return

    print("Loading Dilemma lookup table...")
    lookup = load_dilemma_lookup()
    print(f"  {len(lookup):,} entries loaded")

    correct = 0
    wrong = 0
    missing = 0
    skipped = 0
    total = 0

    wrong_examples = []
    missing_examples = []
    accent_mismatch = 0  # polytonic vs monotonic

    for word, tag, gold_lemma in parse_hnc_tokens(HNC_PATH):
        if tag in SKIP_TAGS:
            skipped += 1
            continue
        if _is_noise(word) or _is_noise(gold_lemma):
            skipped += 1
            continue

        total += 1
        form = word.lower()

        # Try lookup: exact, then lowercase
        dilemma_lemma = lookup.get(form) or lookup.get(word)

        if dilemma_lemma is None:
            missing += 1
            missing_examples.append((form, gold_lemma))
            continue

        # Check match: exact, case-insensitive, or accent-stripped
        if (dilemma_lemma == gold_lemma
                or dilemma_lemma.lower() == gold_lemma.lower()):
            correct += 1
        elif strip_accents(dilemma_lemma.lower()) == strip_accents(gold_lemma.lower()):
            # Accent/breathing difference (e.g. polytonic vs monotonic)
            correct += 1
            accent_mismatch += 1
        else:
            wrong += 1
            wrong_examples.append((form, gold_lemma, dilemma_lemma))

    print(f"\nHNC Golden Corpus Evaluation")
    print(f"{'=' * 40}")
    print(f"Evaluated tokens:  {total:,}")
    print(f"Skipped (noise):   {skipped:,}")
    print(f"Correct:           {correct:,} ({100*correct/total:.1f}%)")
    print(f"  (accent-equiv):  {accent_mismatch:,}")
    print(f"Wrong:             {wrong:,} ({100*wrong/total:.1f}%)")
    print(f"Missing:           {missing:,} ({100*missing/total:.1f}%)")
    print(f"Coverage:          {100*(correct+wrong)/total:.1f}%")
    print(f"Accuracy (found):  {100*correct/(correct+wrong):.1f}%")

    if args.breakdown:
        print(f"\nTop 20 wrong lemmatizations:")
        for form, gold, got in sorted(wrong_examples,
                                        key=lambda x: x[0])[:20]:
            print(f"  {form:20s} gold={gold:20s} got={got}")

        print(f"\nTop 20 missing forms:")
        missing_counter = Counter(missing_examples)
        for (form, lemma), count in missing_counter.most_common(20):
            print(f"  {form:20s} -> {lemma}")


if __name__ == "__main__":
    main()
