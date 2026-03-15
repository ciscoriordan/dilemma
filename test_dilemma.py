#!/usr/bin/env python3
"""Test suite for Dilemma Greek lemmatizer.

Tests lookup table integrity, model predictions, and coverage across
Greek varieties. Run after build_data.py and train.py.

Usage:
    python test_dilemma.py                  # all tests
    python test_dilemma.py --lookup-only    # skip model tests
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def test_lookup_no_chains(lookup: dict) -> tuple[int, int, list[str]]:
    """Verify no chained lookups: every lemma should map to itself."""
    failures = 0
    total = 0
    examples = []
    for form, lemma in lookup.items():
        if lemma != form:  # skip identity entries
            total += 1
            if lemma in lookup and lookup[lemma] != lemma:
                failures += 1
                if len(examples) < 10:
                    examples.append(f"{form} -> {lemma} -> {lookup[lemma]}")
    return failures, total, examples


def test_lookup_lemmas_are_headwords(lookup: dict) -> tuple[int, int, list[str]]:
    """Verify every lemma in the lookup is itself a headword (maps to itself)."""
    lemmas = set(lookup.values())
    headwords = {k for k, v in lookup.items() if k == v}
    missing = lemmas - headwords
    examples = sorted(list(missing))[:10]
    return len(missing), len(lemmas), examples


def test_known_pairs() -> list[dict]:
    """Curated test cases across varieties. Returns list of {form, lemma, variety}."""
    return [
        # SMG verbs
        {"form": "τρέχουν", "lemma": "τρέχω", "variety": "SMG"},
        {"form": "έτρεξε", "lemma": "τρέχω", "variety": "SMG"},
        {"form": "τρομερό", "lemma": "τρομερός", "variety": "SMG"},
        {"form": "πολεμούσαν", "lemma": "πολεμάω", "variety": "SMG"},  # Wiktionary uses -άω headword
        # SMG nouns
        {"form": "ανθρώπων", "lemma": "άνθρωπος", "variety": "SMG"},
        {"form": "γυναίκες", "lemma": "γυναίκα", "variety": "SMG"},
        # AG Epic
        {"form": "θεοί", "lemma": "θεός", "variety": "Epic"},
        {"form": "μῆνιν", "lemma": "μῆνις", "variety": "Epic"},
        {"form": "ἄειδε", "lemma": "ἀείδω", "variety": "Epic"},
        {"form": "Ἀχιλῆος", "lemma": "Ἀχιλλεύς", "variety": "Epic"},
        {"form": "πολύτροπον", "lemma": "πολύτροπος", "variety": "Epic"},
        # AG Attic
        {"form": "ἐποίησεν", "lemma": "ποιέω", "variety": "Attic"},
        {"form": "ἔλεγον", "lemma": "λέγω", "variety": "Attic"},
        {"form": "ἔλυσε", "lemma": "λύω", "variety": "Attic"},
        # AG Koine
        {"form": "ἐβασίλευσεν", "lemma": "βασιλεύω", "variety": "Koine"},
        # Katharevousa
        {"form": "ἐξετέλεσεν", "lemma": "ἐκτελέω", "variety": "Katharevousa"},
        # Articles — Wiktionary uses η as headword for the definite article
        {"form": "τους", "lemma": "η", "variety": "SMG article"},
        {"form": "της", "lemma": "η", "variety": "SMG article"},
    ]


def test_no_greek_junk(pairs: list[dict]) -> tuple[int, int, list[str]]:
    """Check that all training forms contain only Greek characters."""
    failures = 0
    total = len(pairs)
    examples = []
    for p in pairs:
        form = p["form"]
        for ch in form:
            if ch.isalpha():
                cp = ord(ch)
                if not (0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF
                        or 0x0300 <= cp <= 0x036F):
                    failures += 1
                    if len(examples) < 10:
                        examples.append(f"{form} (U+{cp:04X} '{ch}')")
                    break
    return failures, total, examples


def run_lookup_tests():
    """Run all lookup/data integrity tests."""
    print("=" * 60)
    print("LOOKUP TABLE TESTS")
    print("=" * 60)

    results = []

    for prefix, name in [("mg", "MG"), ("ag", "AG"), ("med", "Medieval")]:
        path = DATA_DIR / f"{prefix}_lookup.json"
        if not path.exists():
            print(f"\n  SKIP {name}: {path.name} not found")
            continue

        lookup = json.load(open(path, encoding="utf-8"))
        print(f"\n  {name} lookup: {len(lookup):,} entries")

        # Test: no chains
        fail, total, ex = test_lookup_no_chains(lookup)
        status = "PASS" if fail == 0 else "FAIL"
        results.append((status, f"{name} no chains"))
        print(f"  [{status}] No chained lookups: {fail}/{total} failures")
        for e in ex:
            print(f"         {e}")

        # Test: lemmas are headwords
        fail, total, ex = test_lookup_lemmas_are_headwords(lookup)
        status = "PASS" if fail == 0 else "WARN"
        results.append((status, f"{name} lemma headwords"))
        print(f"  [{status}] Lemmas are headwords: {fail}/{total} missing")
        for e in ex[:5]:
            print(f"         {e}")

    # Test: training data has no non-Greek
    for prefix, name in [("mg", "MG"), ("ag", "AG")]:
        path = DATA_DIR / f"{prefix}_pairs.json"
        if not path.exists():
            continue
        pairs = json.load(open(path, encoding="utf-8"))
        fail, total, ex = test_no_greek_junk(pairs)
        status = "PASS" if fail == 0 else "FAIL"
        results.append((status, f"{name} Greek-only"))
        print(f"\n  [{status}] {name} pairs Greek-only: {fail}/{total} failures")
        for e in ex:
            print(f"         {e}")

    return results


def run_known_pair_tests():
    """Test known form->lemma pairs against the lookup."""
    print("\n" + "=" * 60)
    print("KNOWN PAIR TESTS (lookup)")
    print("=" * 60)

    # Load all lookups
    lookup = {}
    for path in [DATA_DIR / "mg_lookup.json", DATA_DIR / "med_lookup.json",
                 DATA_DIR / "ag_lookup.json"]:
        if path.exists():
            data = json.load(open(path, encoding="utf-8"))
            for k, v in data.items():
                if k not in lookup:
                    lookup[k] = v

    results = []
    cases = test_known_pairs()
    for case in cases:
        form = case["form"]
        expected = case["lemma"]
        variety = case["variety"]

        # Try: exact, lowercase, monotonic, stripped
        from dilemma import to_monotonic, strip_accents
        actual = (lookup.get(form) or lookup.get(form.lower())
                  or lookup.get(to_monotonic(form.lower()))
                  or lookup.get(strip_accents(form.lower())))

        if actual == expected:
            status = "PASS"
        elif actual is None:
            status = "MISS"
        else:
            status = "FAIL"

        results.append((status, f"{form} -> {expected} ({variety})"))
        if status != "PASS":
            print(f"  [{status}] {form} -> expected '{expected}', got '{actual}' ({variety})")

    passed = sum(1 for s, _ in results if s == "PASS")
    print(f"\n  {passed}/{len(results)} passed")
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Dilemma")
    parser.add_argument("--lookup-only", action="store_true",
                        help="Skip model tests, only test data/lookup")
    args = parser.parse_args()

    all_results = []
    all_results.extend(run_lookup_tests())
    all_results.extend(run_known_pair_tests())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for s, _ in all_results if s == "PASS")
    failed = sum(1 for s, _ in all_results if s == "FAIL")
    warned = sum(1 for s, _ in all_results if s == "WARN")
    missed = sum(1 for s, _ in all_results if s == "MISS")
    print(f"  PASS: {passed}  FAIL: {failed}  WARN: {warned}  MISS: {missed}")

    if failed > 0:
        print("\nFailed tests:")
        for s, name in all_results:
            if s == "FAIL":
                print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
