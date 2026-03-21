#!/usr/bin/env python3
"""Compare normalizer period profiles on PapyGreek date subsets.

Tests whether period-matched normalizer profiles (hellenistic, late_antique)
outperform the default "all" profile on their respective date ranges.
"""

import sys
import io
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_PAPYGREEK = Path.home() / "Documents" / "papygreek-treebanks" / "documentary"


def nfc(s):
    return unicodedata.normalize("NFC", s)


def strip_accents(s):
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def classify_period(date):
    if date is None:
        return "unknown"
    if date < 30:
        return "hellenistic"
    elif date < 600:
        return "late_antique"
    return "other"


def parse_papygreek(papygreek_dir):
    tokens = []
    for xml_file in Path(papygreek_dir).rglob("*.xml"):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        doc_meta = tree.find(".//document_meta")
        date_str = doc_meta.get("date_not_before", "") if doc_meta is not None else ""
        try:
            date = int(date_str)
        except ValueError:
            date = None

        for word in tree.findall(".//word"):
            orig = nfc(word.get("orig_form", ""))
            reg = nfc(word.get("form_reg", ""))
            lemma_reg = nfc(word.get("lemma_reg", ""))
            postag = word.get("postag_reg", "")

            if postag and postag[0] == "u":
                continue
            if not orig or not is_greek(orig):
                continue
            if "[" in orig or "]" in orig or "?" in orig:
                continue
            if not lemma_reg:
                continue

            tokens.append({
                "orig": orig,
                "reg": reg,
                "lemma_reg": lemma_reg,
                "period": classify_period(date),
            })

    return tokens


def eval_subset(d, subset):
    """Evaluate Dilemma on a token subset using lemmatize_batch."""
    forms = [t["orig"] for t in subset]
    preds = d.lemmatize_batch(forms)

    total = len(preds)
    strict = 0
    stripped = 0
    var_total = 0
    var_stripped = 0

    for t, pred in zip(subset, preds):
        gold = t["lemma_reg"]
        if pred == gold:
            strict += 1
        if strip_accents(pred.lower()) == strip_accents(gold.lower()):
            stripped += 1

        if strip_accents(t["orig"].lower()) != strip_accents(t["reg"].lower()):
            var_total += 1
            if strip_accents(pred.lower()) == strip_accents(gold.lower()):
                var_stripped += 1

    return {
        "total": total,
        "strict": strict,
        "stripped": stripped,
        "var_total": var_total,
        "var_stripped": var_stripped,
    }


def main():
    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    tokens = parse_papygreek(DEFAULT_PAPYGREEK)
    out.write(f"Total tokens: {len(tokens)}\n")
    period_counts = Counter(t["period"] for t in tokens)
    for p, c in period_counts.most_common():
        out.write(f"  {p:20s} {c:>6} tokens\n")

    from dilemma import Dilemma

    configs = [
        ("no_norm", False, None),
        ("all", True, None),
        ("hellenistic", True, "hellenistic"),
        ("late_antique", True, "late_antique"),
    ]

    subsets_map = {
        "hellenistic": [t for t in tokens if t["period"] == "hellenistic"],
        "late_antique": [t for t in tokens if t["period"] == "late_antique"],
        "all_tokens": tokens,
    }

    out.write(f"\n{'Config':20s} {'Subset':15s} {'Total':>6s} {'Strict':>8s} "
              f"{'Stripped':>10s} {'Var.N':>6s} {'Var.Str':>8s}\n")
    out.write("=" * 80 + "\n")

    for config_name, use_norm, period in configs:
        d = Dilemma(lang="all", resolve_articles=True,
                    normalize=use_norm, period=period)

        # Use greedy decoding for speed
        orig_predict = d._predict
        def _greedy(words, num_beams=1):
            return orig_predict(words, num_beams=1)
        d._predict = _greedy

        for subset_name in ["hellenistic", "late_antique", "all_tokens"]:
            subset = subsets_map[subset_name]
            if not subset:
                continue

            r = eval_subset(d, subset)
            var_pct = (f"{r['var_stripped']/r['var_total']:.1%}"
                       if r["var_total"] else "n/a")

            out.write(f"{config_name:20s} {subset_name:15s} {r['total']:>6d} "
                      f"{r['strict']/r['total']:>7.1%} "
                      f"{r['stripped']/r['total']:>9.1%} "
                      f"{r['var_total']:>6d} {var_pct:>8s}\n")

        out.write("\n")

    out.flush()
    out.detach()


if __name__ == "__main__":
    main()
