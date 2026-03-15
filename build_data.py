#!/usr/bin/env python3
"""Build training data for Dilemma from Wiktionary kaikki JSONL dumps.

Scans both EN and EL Wiktionary dumps for Greek entries, extracts every
inflected form -> lemma pair from inflection tables. Produces:
  - data/mg_pairs.json: Modern Greek form->lemma training pairs
  - data/ag_pairs.json: Ancient Greek form->lemma training pairs
  - data/mg_lookup.json: flat lookup table {form: lemma}
  - data/ag_lookup.json: flat lookup table {form: lemma}

The kaikki dumps are from https://kaikki.org/dictionary/ and contain
complete Wiktionary entries in JSONL format with inflection paradigms.

Usage:
    python build_data.py                    # auto-detect dump locations
    python build_data.py --klisy ~/Klisy    # specify Klisy directory
    python build_data.py --download         # download dumps if missing
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

# Default kaikki dump locations
DEFAULT_KLISY = Path(os.environ.get(
    "KLISY_DIR", Path.home() / "Documents" / "Klisy" / "word_collector"))

DUMPS = {
    "el": {
        "en": "kaikki.org-en-dictionary-Greek.jsonl",
        "el": "kaikki.org-el-dictionary-Greek.jsonl",
    },
    "grc": {
        "en": "kaikki.org-en-dictionary-AncientGreek.jsonl",
        "el": "kaikki.org-el-dictionary-AncientGreek.jsonl",
    },
    "mgr": {
        "el": "kaikki.org-el-dictionary-MedievalGreek.jsonl",
    },
}

DOWNLOAD_URLS = {
    "kaikki.org-en-dictionary-Greek.jsonl":
        "https://kaikki.org/dictionary/Greek/kaikki.org-dictionary-Greek.jsonl",
    "kaikki.org-el-dictionary-Greek.jsonl":
        "https://kaikki.org/elwiktionary/Greek/kaikki.org-dictionary-Greek.jsonl",
    "kaikki.org-en-dictionary-AncientGreek.jsonl":
        "https://kaikki.org/dictionary/Ancient%20Greek/kaikki.org-dictionary-AncientGreek.jsonl",
    "kaikki.org-el-dictionary-AncientGreek.jsonl":
        "https://kaikki.org/elwiktionary/Ancient%20Greek/kaikki.org-dictionary-AncientGreek.jsonl",
    "kaikki.org-el-dictionary-MedievalGreek.jsonl":
        "https://kaikki.org/elwiktionary/Medieval%20Greek/words/kaikki.org-dictionary-MedievalGreek-words.jsonl",
}


def strip_length_marks(s: str) -> str:
    """Strip vowel length marks (breve/macron) from headwords."""
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in (0x0306, 0x0304):  # combining breve, combining macron
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


# Polytonic combining marks to strip for monotonic conversion
_POLYTONIC_STRIP = {
    0x0313,  # COMBINING COMMA ABOVE (smooth breathing)
    0x0314,  # COMBINING REVERSED COMMA ABOVE (rough breathing)
    0x0345,  # COMBINING GREEK YPOGEGRAMMENI (iota subscript)
    0x0306,  # COMBINING BREVE
    0x0304,  # COMBINING MACRON
}
_POLYTONIC_TO_ACUTE = {
    0x0300,  # COMBINING GRAVE ACCENT -> acute
    0x0342,  # COMBINING GREEK PERISPOMENI (circumflex) -> acute
}


def to_monotonic(s: str) -> str:
    """Convert polytonic Greek to monotonic (strip breathings, normalize accents)."""
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _POLYTONIC_STRIP:
            continue
        if cp in _POLYTONIC_TO_ACUTE:
            out.append("\u0301")  # combining acute
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def strip_accents(s: str) -> str:
    """Strip all accents and diacritics."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn"))


def download_dump(filename: str, dest_dir: Path):
    """Download a kaikki dump if missing."""
    url = DOWNLOAD_URLS.get(filename)
    if not url:
        print(f"  No download URL for {filename}")
        return False

    dest = dest_dir / filename
    if dest.exists():
        print(f"  {filename} already exists")
        return True

    print(f"  Downloading {filename}...")
    import urllib.request
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Downloaded {size_mb:.0f} MB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def _add_lookup(lookup: dict, form: str, lemma: str):
    """Add a form to the lookup under original, lowercase, monotonic, and stripped keys."""
    for key in (form, form.lower(), to_monotonic(form), to_monotonic(form).lower(),
                strip_accents(form.lower())):
        if key and key not in lookup:
            lookup[key] = lemma


def extract_pairs(jsonl_path: Path, lang: str) -> tuple[list[dict], dict]:
    """Extract form->lemma pairs from a kaikki JSONL dump.

    Returns:
        pairs: list of {form, lemma, pos, tags} dicts (for training)
        lookup: flat {form: lemma} dict (for fast inference)
    """
    if not jsonl_path.exists():
        print(f"  {lang}: not found at {jsonl_path}")
        return [], {}

    pairs = []
    lookup = {}
    skip_tags = {"romanization", "table-tags", "inflection-template", "class"}
    scanned = 0
    entries_with_forms = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            scanned += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            word = entry.get("word", "")
            pos = entry.get("pos", "")
            forms = entry.get("forms", [])
            if not word or not forms:
                continue

            lemma = strip_length_marks(word)
            entries_with_forms += 1

            # The headword itself maps to itself (original, monotonic, stripped)
            _add_lookup(lookup, lemma, lemma)

            for f_entry in forms:
                tags = f_entry.get("tags", [])
                if any(t in skip_tags for t in tags):
                    continue
                form = strip_length_marks(f_entry.get("form", ""))
                if not form or not any(c.isalpha() for c in form):
                    continue
                # Skip multi-word forms
                if " " in form:
                    continue
                # Skip non-Greek: must be entirely Greek letters + accents
                # Allow: Greek (U+0370-03FF), Extended Greek (U+1F00-1FFF),
                # combining diacriticals (U+0300-036F)
                if not all(
                    "\u0370" <= c <= "\u03FF"       # Greek and Coptic
                    or "\u1F00" <= c <= "\u1FFF"    # Greek Extended
                    or "\u0300" <= c <= "\u036F"    # Combining diacriticals
                    for c in form
                ):
                    continue

                morph_tags = [t for t in tags if t not in ("canonical",)]

                pairs.append({
                    "form": form,
                    "lemma": lemma,
                    "pos": pos,
                    "tags": morph_tags,
                })

                # Also add monotonic version as a training pair
                mono = to_monotonic(form)
                if mono != form:
                    pairs.append({
                        "form": mono,
                        "lemma": lemma,
                        "pos": pos,
                        "tags": morph_tags,
                    })

                # Lookup: original, lowercase, monotonic, accent-stripped
                _add_lookup(lookup, form, lemma)

    print(f"  {lang}: scanned {scanned} entries, "
          f"{entries_with_forms} with forms, "
          f"{len(pairs)} pairs, {len(lookup)} lookup entries")
    return pairs, lookup


def main():
    parser = argparse.ArgumentParser(description="Build Dilemma training data")
    parser.add_argument("--klisy", type=str, default=None,
                        help="Path to Klisy word_collector directory")
    parser.add_argument("--download", action="store_true",
                        help="Download kaikki dumps if missing")
    parser.add_argument("--lang", type=str, default="all",
                        choices=["el", "grc", "mgr", "all"],
                        help="Which language to build (default: all)")
    args = parser.parse_args()

    klisy_dir = Path(args.klisy) if args.klisy else DEFAULT_KLISY

    if args.download:
        print("Checking/downloading kaikki dumps...")
        for lang_dumps in DUMPS.values():
            for filename in lang_dumps.values():
                download_dump(filename, klisy_dir)
        print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    langs = ["el", "grc", "mgr"] if args.lang == "all" else [args.lang]

    for lang in langs:
        lang_name = {"el": "Modern Greek", "grc": "Ancient Greek", "mgr": "Medieval Greek"}[lang]
        prefix = {"el": "mg", "grc": "ag", "mgr": "med"}[lang]
        print(f"\n{'='*50}")
        print(f"Building {lang_name} ({lang})")
        print(f"{'='*50}")

        all_pairs = []
        all_lookup = {}

        for wikt_lang, filename in DUMPS[lang].items():
            path = klisy_dir / filename
            print(f"\nScanning {wikt_lang} Wiktionary: {path.name}")
            pairs, lookup = extract_pairs(path, f"{lang}-{wikt_lang}")
            all_pairs.extend(pairs)
            # Merge lookup (first wins)
            for k, v in lookup.items():
                if k not in all_lookup:
                    all_lookup[k] = v

        # Deduplicate pairs
        seen = set()
        unique_pairs = []
        for p in all_pairs:
            key = (p["form"], p["lemma"])
            if key not in seen:
                seen.add(key)
                unique_pairs.append(p)

        # Save training pairs
        pairs_path = DATA_DIR / f"{prefix}_pairs.json"
        with open(pairs_path, "w", encoding="utf-8") as f:
            json.dump(unique_pairs, f, ensure_ascii=False, indent=2)
        size_mb = pairs_path.stat().st_size / (1024 * 1024)
        print(f"\nTraining pairs: {len(unique_pairs)} ({size_mb:.1f} MB)")
        print(f"  -> {pairs_path}")

        # Save lookup table
        lookup_path = DATA_DIR / f"{prefix}_lookup.json"
        with open(lookup_path, "w", encoding="utf-8") as f:
            json.dump(all_lookup, f, ensure_ascii=False, separators=(",", ":"))
        size_mb = lookup_path.stat().st_size / (1024 * 1024)
        print(f"Lookup table: {len(all_lookup)} entries ({size_mb:.1f} MB)")
        print(f"  -> {lookup_path}")

        # Stats
        unique_lemmas = len(set(all_lookup.values()))
        print(f"Unique lemmas: {unique_lemmas}")


if __name__ == "__main__":
    main()
