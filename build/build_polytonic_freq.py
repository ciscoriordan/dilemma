#!/usr/bin/env python3
"""Build a polytonic Modern Greek word frequency corpus.

Extracts polytonic Greek word forms from glossAPI/Wikisource_Greek_texts
(5,394 texts, ~38M tokens) on HuggingFace.

For each attested polytonic form, records its monotonic equivalent and
corpus frequency. Output is used by lemma to replace blind polytonic
variant generation.

Output: data/mg_polytonic_freq.json

Usage:
    python build/build_polytonic_freq.py
    python build/build_polytonic_freq.py --test       # first 100 Wikisource texts only
    python build/build_polytonic_freq.py --stats       # show stats only, don't save
"""

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "mg_polytonic_freq.json"

# Regex for Greek words (basic + extended/polytonic ranges)
GREEK_WORD_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]+")

# Characters in the polytonic extended range
POLYTONIC_RE = re.compile(r"[\u1F00-\u1FFF]")


def to_monotonic(s):
    """Convert a polytonic Greek string to monotonic.

    Uses NFD decomposition to access individual diacritics, then:
    - Strips breathings (smooth 0313, rough 0314), iota subscript (0345),
      breve (0306), macron (0304)
    - Converts grave (0300) and circumflex/perispomeni (0342) to acute (0301)
    - Recomposes with NFC
    """
    STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
    TO_ACUTE = {0x0300, 0x0342}
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in STRIP:
            continue
        if cp in TO_ACUTE:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def extract_polytonic_tokens(text, counts):
    """Extract polytonic Greek tokens from text, updating counts.

    Returns the number of polytonic tokens found.
    """
    text_nfc = unicodedata.normalize("NFC", text)
    tokens = 0
    for match in GREEK_WORD_RE.finditer(text_nfc):
        word = match.group().lower()
        word = unicodedata.normalize("NFC", word)
        if POLYTONIC_RE.search(word):
            counts[word] += 1
            tokens += 1
    return tokens


def process_wikisource(counts, test_mode=False):
    """Process Wikisource Greek texts dataset."""
    from datasets import load_dataset

    print("Loading glossAPI/Wikisource_Greek_texts...", flush=True)
    if test_mode:
        ds = load_dataset(
            "glossAPI/Wikisource_Greek_texts", split="train", streaming=True
        )
    else:
        ds = load_dataset("glossAPI/Wikisource_Greek_texts", split="train")

    total_tokens = 0
    total_texts = 0
    t0 = time.time()

    if test_mode:
        # Streaming mode for test: take first 100
        for i, row in enumerate(ds):
            if i >= 100:
                break
            text = row.get("text", "")
            if not text:
                continue
            tokens = extract_polytonic_tokens(text, counts)
            total_tokens += tokens
            total_texts += 1
            if (i + 1) % 25 == 0:
                print(
                    f"  Wikisource [{i+1}/100 test]: "
                    f"{total_tokens:,} polytonic tokens, "
                    f"{len(counts):,} unique forms ({time.time()-t0:.1f}s)",
                    flush=True,
                )
    else:
        n_total = len(ds)
        for i, row in enumerate(ds):
            text = row.get("text", "")
            if not text:
                continue
            tokens = extract_polytonic_tokens(text, counts)
            total_tokens += tokens
            total_texts += 1
            if (i + 1) % 500 == 0 or (i + 1) == n_total:
                print(
                    f"  Wikisource [{i+1}/{n_total}]: "
                    f"{total_tokens:,} polytonic tokens, "
                    f"{len(counts):,} unique forms ({time.time()-t0:.1f}s)",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"  Wikisource done: {total_texts:,} texts, "
        f"{total_tokens:,} polytonic tokens ({elapsed:.1f}s)",
        flush=True,
    )
    return total_tokens



def main():
    parser = argparse.ArgumentParser(
        description="Build polytonic MG word frequency corpus"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: first 100 Wikisource texts only",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show stats only, don't save"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output path")
    args = parser.parse_args()

    t0 = time.time()
    counts = Counter()

    # Process Wikisource
    ws_tokens = process_wikisource(counts, test_mode=args.test)

    sources = ["glossAPI/Wikisource_Greek_texts"]
    if args.test:
        sources = ["glossAPI/Wikisource_Greek_texts (first 100)"]

    total_tokens = ws_tokens

    # Build forms dict with monotonic mapping
    forms = {}
    for form, count in counts.items():
        mono = to_monotonic(form)
        forms[form] = {"monotonic": mono, "count": count}

    # Stats
    print(f"\n{'='*60}")
    print(f"Total polytonic tokens:  {total_tokens:,}")
    print(f"Unique polytonic forms:  {len(forms):,}")
    print(f"Forms with count > 10:   {sum(1 for f in forms.values() if f['count'] > 10):,}")
    print(f"Forms with count > 100:  {sum(1 for f in forms.values() if f['count'] > 100):,}")
    print(f"Forms with count > 1000: {sum(1 for f in forms.values() if f['count'] > 1000):,}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    # Top 50
    print(f"\nTop 50 polytonic forms:")
    for form, count in counts.most_common(50):
        mono = to_monotonic(form)
        print(f"  {form:25s} -> {mono:20s}  {count:>10,}")

    # Spot checks
    print(f"\nSpot checks:")
    spot_checks = ["τοῦ", "ἀπὸ", "καὶ", "τὸν", "εἰς", "ἐπὶ", "αὐτοῦ", "ἐκ"]
    for word in spot_checks:
        word_nfc = unicodedata.normalize("NFC", word)
        if word_nfc in forms:
            f = forms[word_nfc]
            print(f"  {word_nfc} -> {f['monotonic']}  (count: {f['count']:,})")
        else:
            print(f"  {word_nfc} -> NOT FOUND")

    if args.stats:
        return

    # Build output
    output = {
        "_meta": {
            "sources": sources,
            "total_polytonic_tokens": total_tokens,
            "unique_polytonic_forms": len(forms),
            "build_date": str(date.today()),
        },
        "forms": forms,
    }

    # Sort forms by frequency (descending) for human readability
    output["forms"] = dict(
        sorted(forms.items(), key=lambda x: -x[1]["count"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {args.output}...", end=" ", flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=None, separators=(",", ":"))
    size_mb = args.output.stat().st_size / 1e6
    print(f"{size_mb:.1f} MB")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
