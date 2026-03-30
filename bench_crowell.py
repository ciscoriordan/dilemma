#!/usr/bin/env python3
"""Crowell-style benchmarks for Cyropaedia (Attic) and Astronautilia (epic).

Replicates the methodology from Crowell's test_lemmatizers:
  - Exclude top 3000 common forms
  - Exclude capitalized words (proper nouns)
  - Check whether Dilemma's output lemma is a valid LSJ headword
  - Also check gold-standard accuracy for Cyropaedia (which has Gorman treebank gold)

Crowell's published results (2026-03):
                       Morpheus  Lemming  Stanza  OdyCy  Dilemma
  Astronautilia (epic)   74       78       74      54     81
  Cyropaedia (Attic)     99.5     99.5     84      72     84
  Herodotus (Ionic)      99.5     95.2     88       x     79

References:
  https://bitbucket.org/ben-crowell/test_lemmatizers/src/master/
"""

import json
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

DILEMMA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DILEMMA_DIR))

DATA_DIR = DILEMMA_DIR / "data"
CORPUS_FREQ_PATH = DATA_DIR / "corpus_freq.json"
EQUIV_PATH = DATA_DIR / "lemma_equivalences.json"
LOOKUP_DB = DATA_DIR / "lookup.db"
LSJ_HEADWORDS_PATH = DATA_DIR / "lsj_headwords.json"
AG_HEADWORDS_PATH = DATA_DIR / "ag_headwords.json"

# Crowell's test data (cloned from bitbucket)
CROWELL_DIR = Path("/tmp/test_lemmatizers/data")
CYROPAEDIA_CSV = CROWELL_DIR / "cyropaedia.csv"
CYROPAEDIA_TXT = CROWELL_DIR / "cyropaedia.txt"
ASTRONAUTILIA_TXT = CROWELL_DIR / "astronautilia_13.txt"

SKIP_UPOS = {"PUNCT", "NUM", "X", "SYM"}


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()


def grave_to_acute(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    out = nfd.replace("\u0300", "\u0301")
    return unicodedata.normalize("NFC", out)


def load_top_n_forms(n=3000) -> set[str]:
    """Load top N forms from corpus_freq.json (accent-stripped keys)."""
    with open(CORPUS_FREQ_PATH) as f:
        data = json.load(f)
    forms = data["forms"]
    sorted_forms = sorted(forms.items(), key=lambda x: x[1][0], reverse=True)
    return {f for f, _ in sorted_forms[:n]}


def load_lsj_headwords() -> set[str]:
    """Load LSJ headwords and normalize for matching."""
    with open(LSJ_HEADWORDS_PATH) as f:
        raw = json.load(f)
    # Also load ag_headwords which includes Wiktionary headwords
    with open(AG_HEADWORDS_PATH) as f:
        ag_raw = json.load(f)

    headwords = set()
    for hw in raw:
        hw = hw.strip()
        if hw:
            headwords.add(hw)
            headwords.add(strip_accents(hw))
    for hw in ag_raw:
        hw = hw.strip()
        if hw:
            headwords.add(hw)
            headwords.add(strip_accents(hw))
    return headwords


def load_equivalences() -> dict[str, set[str]]:
    with open(EQUIV_PATH) as f:
        data = json.load(f)
    equiv = {}
    for group in data["groups"]:
        group_set = set(group)
        for lemma in group:
            if lemma in equiv:
                equiv[lemma] = equiv[lemma] | group_set
            else:
                equiv[lemma] = set(group_set)
    return equiv


def lemma_match(predicted: str, gold: str, equiv: dict[str, set[str]]) -> bool:
    if predicted == gold:
        return True
    if strip_accents(predicted) == strip_accents(gold):
        return True
    if gold in equiv and predicted in equiv[gold]:
        return True
    if predicted in equiv and gold in equiv[predicted]:
        return True
    return False


def is_valid_lemma(lemma: str, headwords: set[str]) -> bool:
    """Check if a lemma is a valid LSJ/Wiktionary headword."""
    if lemma in headwords:
        return True
    if strip_accents(lemma) in headwords:
        return True
    # Try grave_to_acute
    acute = grave_to_acute(lemma)
    if acute in headwords:
        return True
    return False


def tokenize_greek(text: str) -> list[str]:
    """Simple Greek tokenizer: split on whitespace and punctuation."""
    import re
    # Split on non-Greek characters (keep elision marks)
    tokens = re.findall(r"[ʼ᾽''ʹ\u0370-\u03FF\u1F00-\u1FFF\u0300-\u036F]+", text)
    return tokens


def is_capitalized(word: str) -> bool:
    """Check if first letter is uppercase Greek."""
    for ch in word:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            return cat == "Lu"
    return False


def parse_cyropaedia_csv(path: Path) -> list[dict]:
    """Parse Crowell's Cyropaedia CSV with Gorman gold lemmas."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            fields = line.split(",")
            # Format: work,book,chapter,section,word_idx,form,lemma,?,pos_tag
            if len(fields) < 7:
                continue
            form = fields[5]
            lemma = fields[6]
            pos_tag = fields[8] if len(fields) > 8 else ""
            if not form or not lemma:
                continue
            pairs.append({"form": form, "lemma": lemma, "pos": pos_tag})
    return pairs


def run_crowell_test(name: str, words: list[str], top3000: set[str],
                     headwords: set[str], dilemma_cls, gold_pairs=None,
                     equiv=None):
    """Run Crowell-style test: exclude common words, check LSJ validity.

    If gold_pairs is provided (as a dict form->lemma), also report accuracy.
    """
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    # Filter: exclude common forms and capitalized words
    uncommon = []
    skipped_common = 0
    skipped_cap = 0
    for w in words:
        if is_capitalized(w):
            skipped_cap += 1
            continue
        if strip_accents(w) in top3000:
            skipped_common += 1
            continue
        uncommon.append(w)

    # Deduplicate
    seen = set()
    unique_words = []
    for w in uncommon:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)

    print(f"  Total tokens: {len(words):,}")
    print(f"  Skipped (common): {skipped_common:,}")
    print(f"  Skipped (capitalized): {skipped_cap:,}")
    print(f"  Uncommon tokens: {len(uncommon):,}")
    print(f"  Unique uncommon words: {len(unique_words):,}")

    # Run Dilemma
    m = dilemma_cls(lang="all", convention="lsj")
    valid = 0
    invalid = 0
    invalid_examples = []
    gold_correct = 0
    gold_wrong = 0
    gold_wrong_examples = []

    for w in unique_words:
        pred = m.lemmatize(w)

        # Crowell test: is the lemma in LSJ?
        if is_valid_lemma(pred, headwords):
            valid += 1
        else:
            invalid += 1
            if len(invalid_examples) < 30:
                invalid_examples.append((w, pred))

        # Gold accuracy test (if gold available)
        if gold_pairs and w in gold_pairs:
            gold = gold_pairs[w]
            if lemma_match(pred, gold, equiv or {}):
                gold_correct += 1
            else:
                gold_wrong += 1
                if len(gold_wrong_examples) < 30:
                    gold_wrong_examples.append((w, gold, pred))

    total = valid + invalid
    pct = valid / total * 100 if total else 0

    print(f"\n  Crowell-style results (is lemma in LSJ/Wiktionary?):")
    print(f"    Valid lemmas: {valid}/{total} = {pct:.1f}%")

    if invalid_examples:
        print(f"\n    Invalid lemma examples:")
        print(f"    {'Form':<25} {'Predicted lemma':<25}")
        print(f"    {'-'*50}")
        for form, pred in invalid_examples[:20]:
            print(f"    {form:<25} {pred:<25}")

    if gold_pairs:
        gold_total = gold_correct + gold_wrong
        gold_pct = gold_correct / gold_total * 100 if gold_total else 0
        print(f"\n  Gold accuracy (vs Gorman treebank):")
        print(f"    Correct: {gold_correct}/{gold_total} = {gold_pct:.1f}%")

        if gold_wrong_examples:
            print(f"\n    Wrong examples:")
            print(f"    {'Form':<25} {'Gold':<20} {'Predicted':<20}")
            print(f"    {'-'*65}")
            for form, gold, pred in gold_wrong_examples[:20]:
                print(f"    {form:<25} {gold:<20} {pred:<20}")

    return {"name": name, "valid": valid, "total": total, "pct": pct,
            "gold_correct": gold_correct if gold_pairs else None,
            "gold_total": (gold_correct + gold_wrong) if gold_pairs else None}


def main():
    print("=" * 70)
    print("Crowell-Style Benchmarks: Cyropaedia + Astronautilia")
    print("=" * 70)

    # Check for test data
    if not CROWELL_DIR.exists():
        print(f"\nError: Crowell test data not found at {CROWELL_DIR}")
        print("Clone from: https://bitbucket.org/ben-crowell/test_lemmatizers.git")
        print("  git clone https://bitbucket.org/ben-crowell/test_lemmatizers.git /tmp/test_lemmatizers")
        sys.exit(1)

    top3000 = load_top_n_forms(3000)
    headwords = load_lsj_headwords()
    equiv = load_equivalences()
    print(f"Top 3000 forms loaded")
    print(f"LSJ+Wiktionary headwords: {len(headwords):,}")
    print(f"Equivalence groups: {len(equiv):,}")

    from dilemma import Dilemma

    results = []

    # --- Cyropaedia ---
    if CYROPAEDIA_TXT.exists():
        text = CYROPAEDIA_TXT.read_text(encoding="utf-8")
        words = tokenize_greek(text)

        # Also load gold lemmas from CSV
        gold_pairs = None
        if CYROPAEDIA_CSV.exists():
            csv_data = parse_cyropaedia_csv(CYROPAEDIA_CSV)
            # Build form->lemma dict (keep most frequent lemma per form)
            form_lemma_counts: dict[str, Counter] = {}
            for p in csv_data:
                f = p["form"]
                if f not in form_lemma_counts:
                    form_lemma_counts[f] = Counter()
                form_lemma_counts[f][p["lemma"]] += 1
            gold_pairs = {f: c.most_common(1)[0][0] for f, c in form_lemma_counts.items()}
            print(f"\nCyropaedia gold pairs from Gorman CSV: {len(gold_pairs):,}")

        r = run_crowell_test(
            "Cyropaedia (Attic prose, Xenophon)",
            words, top3000, headwords, Dilemma,
            gold_pairs=gold_pairs, equiv=equiv
        )
        results.append(r)
    else:
        print(f"\n  Cyropaedia text not found: {CYROPAEDIA_TXT}")

    # --- Astronautilia ---
    if ASTRONAUTILIA_TXT.exists():
        text = ASTRONAUTILIA_TXT.read_text(encoding="utf-8")
        words = tokenize_greek(text)

        r = run_crowell_test(
            "Astronautilia Book 13 (epic, Kresadlo)",
            words, top3000, headwords, Dilemma
        )
        results.append(r)
    else:
        print(f"\n  Astronautilia text not found: {ASTRONAUTILIA_TXT}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY: Comparison with Crowell's results")
    print(f"{'=' * 70}")

    crowell_numbers = {
        "Cyropaedia": {"Morpheus": 99.5, "Lemming": 99.5, "Stanza": 84,
                       "OdyCy": 72, "Dilemma_old": 84},
        "Astronautilia Book 13": {"Morpheus": 74, "Lemming": 78, "Stanza": 74,
                                  "OdyCy": 54, "Dilemma_old": 81},
        "Herodotus": {"Morpheus": 99.5, "Lemming": 95.2, "Stanza": 88,
                      "OdyCy": None, "Dilemma_old": 79},
    }

    print(f"\n  {'Text':<30} {'Dilemma old':<14} {'Dilemma now':<14} {'Lemming':<10} {'Morpheus':<10}")
    print(f"  {'-'*78}")

    for r in results:
        name_short = r["name"].split("(")[0].strip()
        cn = crowell_numbers.get(name_short, {})
        old = cn.get("Dilemma_old", "?")
        lemming = cn.get("Lemming", "?")
        morpheus = cn.get("Morpheus", "?")
        print(f"  {name_short:<30} {old!s:<14} {r['pct']:.1f}%{'':<8} {lemming!s:<10} {morpheus!s:<10}")
        if r.get("gold_correct") is not None:
            gold_pct = r["gold_correct"] / r["gold_total"] * 100 if r["gold_total"] else 0
            print(f"    (gold accuracy: {gold_pct:.1f}%)")

    # Add Herodotus for context
    print(f"  {'Herodotus':<30} {'79':<14} {'92.9%':<14} {'95.2':<10} {'99.5':<10}")
    print(f"    (from bench_herodotus.py)")


if __name__ == "__main__":
    main()
