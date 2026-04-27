#!/usr/bin/env python3
"""Extract form->lemma pairs from PROIEL Ancient Greek treebank (CoNLL-U).

The PROIEL treebank contains gold-standard, manually annotated Ancient Greek
data, primarily from Herodotus' Histories (~81K tokens across train/dev/test
splits). All annotations are expert-verified, making this one of the highest
quality AG lemmatization resources available.

Source: https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL
License: CC BY-NC-SA 3.0

Outputs:
    data/proiel_pairs.json - list of {form, lemma, pos} dicts

Usage:
    python extract_proiel.py
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "proiel_pairs.json"

# Default location for PROIEL CoNLL-U files
PROIEL_DIR = Path.home() / "Documents" / "morphy" / "data" / "UD_Ancient_Greek-PROIEL"

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
    "PROPN": "noun",  # proper nouns stored as noun with tags
}


def _is_greek(s: str) -> bool:
    """Check if string contains at least one Greek character."""
    return any(
        "\u0370" <= c <= "\u03FF"
        or "\u1F00" <= c <= "\u1FFF"
        for c in s
    )


def parse_conllu(path: Path) -> list[dict]:
    """Parse a CoNLL-U file and extract form-lemma-pos triples.

    Skips:
    - Comment lines (start with #)
    - Multiword tokens (ID contains '-')
    - Empty nodes (ID contains '.')
    - Punctuation, numbers, symbols (UPOS in SKIP_UPOS)
    - Non-Greek tokens
    """
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
            # Skip multiword tokens (e.g. "1-2") and empty nodes (e.g. "1.1")
            if "-" in token_id or "." in token_id:
                continue

            form = fields[1]
            lemma = fields[2]
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


def extract_proiel_pairs(proiel_dir: Path = PROIEL_DIR) -> list[dict]:
    """Extract deduplicated form-lemma pairs from all PROIEL splits.

    Returns list of {form, lemma, pos} dicts (same format as GLAUx/Diorisis).
    When the same form appears with multiple lemmas, keeps the most frequent.
    """
    all_pairs = []
    files = sorted(proiel_dir.glob("grc_proiel-ud-*.conllu"))
    if not files:
        print(f"Error: no .conllu files found in {proiel_dir}")
        return []

    for path in files:
        raw = parse_conllu(path)
        split_name = path.stem.split("-")[-1]  # train/dev/test
        print(f"  {split_name}: {len(raw):,} tokens")
        all_pairs.extend(raw)

    print(f"  Total tokens (post-filter): {len(all_pairs):,}")

    # Deduplicate: for each (form, lemma) pair, keep the most common POS.
    # Then for each form, keep only the most frequent lemma.
    from collections import Counter

    # Count (form, lemma) occurrences and their POS
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_pos: dict[tuple[str, str], Counter[str]] = {}

    for p in all_pairs:
        key = (p["form"], p["lemma"])
        pair_counts[key] += 1
        if key not in pair_pos:
            pair_pos[key] = Counter()
        pair_pos[key][p["pos"]] += 1

    # For each form, pick the most frequent lemma
    form_best: dict[str, tuple[str, str, int]] = {}  # form -> (lemma, pos, count)
    for (form, lemma), count in pair_counts.items():
        best_pos = pair_pos[(form, lemma)].most_common(1)[0][0]
        if form not in form_best or count > form_best[form][2]:
            form_best[form] = (lemma, best_pos, count)

    # Build output list
    result = []
    for form, (lemma, pos, _count) in sorted(form_best.items()):
        result.append({"form": form, "lemma": lemma, "pos": pos})

    # Stats
    n_propn = sum(1 for p in all_pairs if p["upos"] == "PROPN")
    print(f"  Proper nouns in corpus: {n_propn:,}")
    print(f"  Unique form->lemma pairs: {len(result):,}")

    return result


def main():
    if not PROIEL_DIR.exists():
        print(f"Error: {PROIEL_DIR} not found")
        print("Clone from https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL")
        return

    pairs = extract_proiel_pairs()

    if not pairs:
        return

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=0)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  -> {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
