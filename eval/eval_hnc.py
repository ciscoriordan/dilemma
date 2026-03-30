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
sys.path.insert(0, str(SCRIPT_DIR))

from dilemma import Dilemma
DATA_DIR = SCRIPT_DIR / "data"
HNC_PATH = DATA_DIR / "HNC_Golden_Corpus.xml"
EQUIV_PATH = DATA_DIR / "lemma_equivalences.json"

# Tags to skip (same as extract_hnc.py)
SKIP_TAGS = {"ABBR", "DIG", "DATE", "INIT", "RgSyXx", "RgAbXx", "PuXx"}

# Load lemma equivalences (same pattern as bench_fast.py)
with open(EQUIV_PATH) as f:
    _eq_data = json.load(f)
equiv = {}
for group in _eq_data["groups"]:
    group_set = set(group)
    for lemma in group:
        equiv[lemma] = equiv.get(lemma, set()) | group_set


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def are_equivalent(pred: str, gold: str) -> bool:
    """Check if two lemmas are equivalent (exact, accent-stripped, or via equivalence groups)."""
    pa, ga = strip_accents(pred).lower(), strip_accents(gold).lower()
    if pa == ga:
        return True
    for e in equiv.get(gold, set()):
        if strip_accents(e).lower() == pa:
            return True
    for e in equiv.get(pred, set()):
        if strip_accents(e).lower() == ga:
            return True
    return False


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


def safe_lemmatize(d: Dilemma, form: str) -> str:
    try:
        return d.lemmatize(form)
    except Exception:
        return form


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

    print("Loading Dilemma (lang='el', convention='triantafyllidis')...")
    d = Dilemma(lang='el', convention='triantafyllidis')
    d.preload()
    print(f"  Dilemma loaded")

    correct = 0
    wrong = 0
    missing = 0
    skipped = 0
    total = 0

    wrong_examples = []
    missing_examples = []

    for word, tag, gold_lemma in parse_hnc_tokens(HNC_PATH):
        if tag in SKIP_TAGS:
            skipped += 1
            continue
        if _is_noise(word) or _is_noise(gold_lemma):
            skipped += 1
            continue

        total += 1
        form = word.lower()

        dilemma_lemma = safe_lemmatize(d, form)

        # If Dilemma just returns the input unchanged, count as missing
        if dilemma_lemma == form and form != gold_lemma.lower():
            # Check if it's truly missing (lemmatize returns input for unknowns)
            # but some words are already in citation form
            if not are_equivalent(dilemma_lemma, gold_lemma):
                missing += 1
                missing_examples.append((form, gold_lemma))
                continue

        # Check match using equivalence groups
        if are_equivalent(dilemma_lemma, gold_lemma):
            correct += 1
        else:
            wrong += 1
            wrong_examples.append((form, gold_lemma, dilemma_lemma))

    print(f"\nHNC Golden Corpus Evaluation")
    print(f"{'=' * 40}")
    print(f"Evaluated tokens:  {total:,}")
    print(f"Skipped (noise):   {skipped:,}")
    print(f"Correct:           {correct:,} ({100*correct/total:.1f}%)")
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
