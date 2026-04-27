#!/usr/bin/env python3
"""Extract form->lemma pairs from Perseus Ancient Greek treebank (CoNLL-U).

The Perseus treebank (UD_Ancient_Greek-Perseus) contains gold-standard
annotations from the Ancient Greek Dependency Treebank (AGDT). Authors
include Sophocles, Aeschylus, Homer, Hesiod, Herodotus, Thucydides,
Plutarch, Polybius, and Athenaeus (~178K tokens across train/dev/test).

This data supplements PROIEL and Gorman treebanks already in the pipeline,
adding ~19K novel form-lemma pairs not covered by those sources.

Source: https://github.com/UniversalDependencies/UD_Ancient_Greek-Perseus
License: CC BY-NC-SA 3.0

Outputs:
    data/perseus_pairs.json - list of {form, lemma, pos} dicts

Usage:
    python extract_perseus.py
"""

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from dilemma.form_sanitize import sanitize_form  # noqa: E402

DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "perseus_pairs.json"

# Default location for Perseus CoNLL-U files
PERSEUS_DIR = Path.home() / "Documents" / "morphy" / "data" / "UD_Ancient_Greek-Perseus"

# UPOS tags to skip
SKIP_UPOS = {"PUNCT", "NUM", "X", "SYM"}

# Map UPOS to simpler POS labels (matching GLAUx/Diorisis format)
UPOS_TO_POS = {
    "NOUN": "noun",
    "VERB": "verb",
    "ADJ": "adj",
    "ADV": "adv",
    "PRON": "pron",
    "DET": "det",
    "ADP": "prep",
    "CCONJ": "conj",
    "SCONJ": "conj",
    "PART": "particle",
    "INTJ": "intj",
    "AUX": "verb",
    "PROPN": "noun",
}


def _is_greek(s: str) -> bool:
    """Check if string contains at least one Greek character."""
    return any(
        "\u0370" <= c <= "\u03FF"
        or "\u1F00" <= c <= "\u1FFF"
        for c in s
    )


def _normalize_nfc(s: str) -> str:
    # NFC + fix misplaced combining breathings (Perseus encodes elision and
    # aphaeresis with a combining psili U+0313; canonical AG orthography
    # uses U+1FBD GREEK KORONIS, and a leading mark should be reordered
    # onto the following base letter). See form_sanitize for the rules.
    return sanitize_form(unicodedata.normalize("NFC", s))


def parse_conllu(path: Path) -> list[dict]:
    """Parse a CoNLL-U file and extract form-lemma-pos triples."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) < 4:
                continue

            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue

            form = _normalize_nfc(fields[1])
            lemma = _normalize_nfc(fields[2])
            upos = fields[3]

            if upos in SKIP_UPOS:
                continue
            if not form or not lemma or lemma == "_":
                continue
            if not _is_greek(form):
                continue

            pos = UPOS_TO_POS.get(upos, upos.lower())
            pairs.append({"form": form, "lemma": lemma, "pos": pos, "upos": upos})

    return pairs


def extract_perseus_pairs(perseus_dir: Path = PERSEUS_DIR) -> list[dict]:
    """Extract deduplicated form-lemma pairs from all Perseus splits."""
    all_pairs = []
    files = sorted(perseus_dir.glob("grc_perseus-ud-*.conllu"))
    if not files:
        print(f"Error: no .conllu files found in {perseus_dir}")
        return []

    for path in files:
        raw = parse_conllu(path)
        split_name = path.stem.split("-")[-1]
        print(f"  {split_name}: {len(raw):,} tokens")
        all_pairs.extend(raw)

    print(f"  Total tokens (post-filter): {len(all_pairs):,}")

    # Deduplicate: for each (form, lemma) pair, keep the most common POS.
    # Then for each form, keep only the most frequent lemma.
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_pos: dict[tuple[str, str], Counter[str]] = {}

    for p in all_pairs:
        key = (p["form"], p["lemma"])
        pair_counts[key] += 1
        if key not in pair_pos:
            pair_pos[key] = Counter()
        pair_pos[key][p["pos"]] += 1

    # For each form, pick the most frequent lemma
    form_best: dict[str, tuple[str, str, int]] = {}
    for (form, lemma), count in pair_counts.items():
        best_pos = pair_pos[(form, lemma)].most_common(1)[0][0]
        if form not in form_best or count > form_best[form][2]:
            form_best[form] = (lemma, best_pos, count)

    result = []
    for form, (lemma, pos, _count) in sorted(form_best.items()):
        result.append({"form": form, "lemma": lemma, "pos": pos})

    n_propn = sum(1 for p in all_pairs if p["upos"] == "PROPN")
    print(f"  Proper nouns in corpus: {n_propn:,}")
    print(f"  Unique form->lemma pairs: {len(result):,}")

    return result


def main():
    if not PERSEUS_DIR.exists():
        print(f"Error: {PERSEUS_DIR} not found")
        print("Clone from https://github.com/UniversalDependencies/UD_Ancient_Greek-Perseus")
        return

    print(f"Perseus treebank ({PERSEUS_DIR}):")
    pairs = extract_perseus_pairs()

    if not pairs:
        return

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=0)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  -> {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
