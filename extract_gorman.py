#!/usr/bin/env python3
"""Extract form->lemma pairs from the Gorman treebank (CoNLL-U format).

The Gorman treebanks are dependency-annotated Ancient Greek texts created
by Vanessa Gorman. They cover a wide range of authors including
Herodotus, Thucydides, Xenophon, Demosthenes, Lysias, Polybius, and
others. The CoNLL-U files contain ~775K lines across train/dev/test
splits, with gold-standard form-lemma annotations.

Source: https://github.com/UniversalDependencies/UD_Ancient_Greek-Gorman
Annotator: Vanessa Gorman

Outputs:
    data/gorman_pairs.json - [{form, lemma, pos}, ...] list of pairs

Usage:
    python extract_gorman.py
"""

import json
import sys
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from dilemma.form_sanitize import sanitize_form  # noqa: E402

DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "gorman_pairs.json"

# Default location for Gorman CoNLL-U files (sibling repo)
GORMAN_DIR = SCRIPT_DIR.parent / "opla" / "data" / "Gorman"

# UPOS tags to skip (punctuation, symbols)
SKIP_UPOS = {"PUNCT", "SYM", "X", "NUM"}


def _is_greek(s: str) -> bool:
    """Check if string contains Greek characters."""
    return any(
        "\u0370" <= c <= "\u03FF"       # Greek and Coptic
        or "\u1F00" <= c <= "\u1FFF"    # Greek Extended
        for c in s
    )


def _normalize_nfc(s: str) -> str:
    """Normalize to NFC and fix misplaced combining breathing marks.

    Gorman encodes elision with a trailing combining psili (U+0313);
    the canonical spacing mark is U+1FBD GREEK KORONIS. A leading
    combining breathing is reordered onto the following base letter.
    """
    return sanitize_form(unicodedata.normalize("NFC", s))


def extract_gorman_pairs(gorman_dir: Path = GORMAN_DIR) -> list[dict]:
    """Parse Gorman CoNLL-U files and extract form->lemma pairs.

    Returns:
        List of {form, lemma, pos} dicts with unique form-lemma pairs.
    """
    conllu_files = sorted(gorman_dir.glob("gorman-*.conllu"))
    if not conllu_files:
        print(f"Error: no gorman-*.conllu files found in {gorman_dir}")
        return []

    print(f"Gorman treebank ({gorman_dir}):")
    print(f"  Files: {[f.name for f in conllu_files]}")

    seen = {}  # (form, lemma) -> pos
    total_tokens = 0
    skipped_punct = 0
    skipped_multiword = 0
    skipped_non_greek = 0
    skipped_empty = 0

    for conllu_path in conllu_files:
        with open(conllu_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                # Skip comments and blank lines
                if not line or line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) < 4:
                    continue

                token_id = fields[0]

                # Skip multiword tokens (e.g., "1-2") and empty nodes ("1.1")
                if "-" in token_id or "." in token_id:
                    skipped_multiword += 1
                    continue

                form = _normalize_nfc(fields[1])
                lemma = _normalize_nfc(fields[2])
                upos = fields[3]

                total_tokens += 1

                # Skip punctuation and symbols
                if upos in SKIP_UPOS:
                    skipped_punct += 1
                    continue

                # Skip empty forms/lemmas
                if not form or not lemma or form == "_" or lemma == "_":
                    skipped_empty += 1
                    continue

                # Skip non-Greek tokens
                if not _is_greek(form) or not _is_greek(lemma):
                    skipped_non_greek += 1
                    continue

                key = (form, lemma)
                if key not in seen:
                    seen[key] = upos.lower()

    # Convert to list format matching glaux/diorisis conventions
    pairs = [
        {"form": form, "lemma": lemma, "pos": pos}
        for (form, lemma), pos in sorted(seen.items())
    ]

    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Skipped (punct/sym): {skipped_punct:,}")
    print(f"  Skipped (multiword): {skipped_multiword:,}")
    print(f"  Skipped (non-Greek): {skipped_non_greek:,}")
    print(f"  Skipped (empty): {skipped_empty:,}")
    print(f"  Unique form-lemma pairs: {len(pairs):,}")

    return pairs


def main():
    if not GORMAN_DIR.exists():
        print(f"Error: {GORMAN_DIR} not found")
        print("Expected Gorman CoNLL-U files at this location.")
        return

    pairs = extract_gorman_pairs()

    if not pairs:
        return

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=0)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  -> {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
