#!/usr/bin/env python3
"""Evaluate Dilemma on PapyGreek documentary papyri.

PapyGreek treebanks contain manually annotated documentary Greek with
dual orig/reg annotations. This evaluates:
1. Lemmatization accuracy on original (scribal) forms
2. Normalizer effectiveness (does normalize=True help?)
3. Form regularization (can we predict reg_form from orig_form?)

Usage:
    python eval_papygreek.py
    python eval_papygreek.py --normalize           # test with normalizer
    python eval_papygreek.py --papygreek ~/path    # custom path
"""

import argparse
import json
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PAPYGREEK = Path.home() / "Documents" / "papygreek-treebanks" / "documentary"


def nfc(s):
    return unicodedata.normalize("NFC", s)


def strip_accents(s):
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def to_monotonic(s):
    _STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
    _TO_ACUTE = {0x0300, 0x0342}
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _STRIP:
            continue
        if cp in _TO_ACUTE:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def parse_papygreek(papygreek_dir):
    """Parse all PapyGreek XML files, extract tokens with orig/reg pairs."""
    tokens = []
    xml_files = list(Path(papygreek_dir).rglob("*.xml"))
    print(f"Found {len(xml_files)} PapyGreek files")

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        doc_meta = tree.find(".//document_meta")
        date_from = doc_meta.get("date_not_before", "") if doc_meta is not None else ""

        for word in tree.findall(".//word"):
            orig = nfc(word.get("orig_form", ""))
            reg = nfc(word.get("form_reg", ""))
            lemma_orig = nfc(word.get("lemma_orig", ""))
            lemma_reg = nfc(word.get("lemma_reg", ""))
            postag = word.get("postag_reg", "")

            # Skip punctuation and non-Greek
            if postag and postag[0] == "u":
                continue
            if not orig or not is_greek(orig):
                continue

            # Skip bracketed/lacuna forms
            if "[" in orig or "]" in orig or "?" in orig:
                continue

            tokens.append({
                "orig": orig,
                "reg": reg,
                "lemma_orig": lemma_orig,
                "lemma_reg": lemma_reg,
                "postag": postag,
                "date": date_from,
                "file": xml_file.stem,
            })

    return tokens


def evaluate(tokens, use_normalize=False):
    """Run Dilemma on orig forms and compare with gold lemmas."""
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    from dilemma import Dilemma

    d = Dilemma(lang="all", resolve_articles=True, normalize=use_normalize)
    print(f"Dilemma loaded (normalize={use_normalize})")

    results = {
        "total": 0,
        "lemma_strict": 0,
        "lemma_mono": 0,
        "lemma_stripped": 0,
        "has_variation": 0,  # orig != reg
        "variation_lemma_strict": 0,
        "variation_lemma_stripped": 0,
    }

    errors_by_type = Counter()
    variation_examples = []

    for t in tokens:
        orig = t["orig"]
        gold_lemma = t["lemma_reg"]  # use regularized lemma as gold
        reg_form = t["reg"]

        if not gold_lemma:
            continue

        results["total"] += 1
        pred = d.lemmatize(orig)

        strict = (pred == gold_lemma)
        mono = (to_monotonic(pred).lower() == to_monotonic(gold_lemma).lower())
        stripped = (strip_accents(pred.lower()) == strip_accents(gold_lemma.lower()))

        if strict:
            results["lemma_strict"] += 1
        if mono:
            results["lemma_mono"] += 1
        if stripped:
            results["lemma_stripped"] += 1

        # Track tokens where orig != reg (scribal variations)
        if strip_accents(orig.lower()) != strip_accents(reg_form.lower()):
            results["has_variation"] += 1
            if strict:
                results["variation_lemma_strict"] += 1
            if stripped:
                results["variation_lemma_stripped"] += 1
            if not stripped and len(variation_examples) < 30:
                variation_examples.append(
                    (orig, reg_form, gold_lemma, pred))

    return results, variation_examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dilemma on PapyGreek")
    parser.add_argument("--papygreek", type=str, default=str(DEFAULT_PAPYGREEK))
    parser.add_argument("--normalize", action="store_true",
                        help="Enable orthographic normalizer")
    args = parser.parse_args()

    tokens = parse_papygreek(args.papygreek)
    print(f"Total tokens: {len(tokens)}")

    if not tokens:
        print("No tokens found.")
        return

    # Count variations
    variations = sum(1 for t in tokens
                     if strip_accents(t["orig"].lower()) != strip_accents(t["reg"].lower()))
    print(f"Tokens with scribal variation (orig != reg): {variations} "
          f"({100*variations/len(tokens):.1f}%)")

    results, examples = evaluate(tokens, use_normalize=args.normalize)
    total = results["total"]

    print(f"\n{'='*60}")
    print(f"Results ({total} tokens, normalize={args.normalize})")
    print(f"{'='*60}")
    print(f"  Lemma strict:   {results['lemma_strict']:>6}/{total} "
          f"({100*results['lemma_strict']/total:.1f}%)")
    print(f"  Lemma monotonic:{results['lemma_mono']:>6}/{total} "
          f"({100*results['lemma_mono']/total:.1f}%)")
    print(f"  Lemma stripped: {results['lemma_stripped']:>6}/{total} "
          f"({100*results['lemma_stripped']/total:.1f}%)")

    if results["has_variation"]:
        var = results["has_variation"]
        print(f"\n  Varied forms only ({var} tokens):")
        print(f"    Strict:   {results['variation_lemma_strict']:>6}/{var} "
              f"({100*results['variation_lemma_strict']/var:.1f}%)")
        print(f"    Stripped: {results['variation_lemma_stripped']:>6}/{var} "
              f"({100*results['variation_lemma_stripped']/var:.1f}%)")

    if examples:
        print(f"\n  Sample errors on varied forms:")
        for orig, reg, gold, pred in examples[:15]:
            print(f"    {orig:20s} (reg={reg:15s}) gold={gold:15s} pred={pred}")


if __name__ == "__main__":
    main()
