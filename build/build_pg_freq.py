#!/usr/bin/env python3
"""Extract token frequencies from the Patrologia Graeca corpus.

Reads .vert files (Sketch Engine XML format) from a PG.zip archive.
Each <w> element contains a tab-separated line:
    surface_form  stripped_form  lemma  stripped_lemma  POS

We use the stripped_form (field 2), which is already lowercased and
accent-stripped. We skip forms that contain no Greek characters.

Output: data/pg_freq.json
    {"_total_tokens": N, "_sources": [...],
     "_n_forms": N,
     "forms": {"θεος": [total], ...}}

Usage:
    python build/build_pg_freq.py
    python build/build_pg_freq.py --pg-zip /path/to/PG.zip
    python build/build_pg_freq.py --stats  # show stats only
"""

import argparse
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_PG_ZIP = Path("/tmp/pg_sample/PG.zip")
OUTPUT_PATH = DATA_DIR / "pg_freq.json"


def is_greek(s):
    """Check if string contains at least one Greek character."""
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)


def process_vert_lines(lines, file_counts):
    """Process lines from a .vert file, updating file_counts.

    Returns (tokens_added, lines_skipped).
    """
    tokens = 0
    skipped = 0
    in_word = False

    for line in lines:
        line = line.strip()

        if line.startswith("<w "):
            in_word = True
            continue
        elif line == "</w>":
            in_word = False
            continue
        elif line.startswith("<"):
            in_word = False
            continue

        if not in_word:
            continue

        # Tab-separated: surface_form  stripped_form  lemma  stripped_lemma  POS
        fields = line.split("\t")
        if len(fields) < 2:
            skipped += 1
            continue

        stripped_form = fields[1]
        if not stripped_form or not is_greek(stripped_form):
            skipped += 1
            continue

        file_counts[stripped_form] += 1
        tokens += 1

    return tokens, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Extract Patrologia Graeca token frequencies")
    parser.add_argument("--pg-zip", type=Path, default=DEFAULT_PG_ZIP,
                        help="Path to PG.zip")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output path")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats only, don't save")
    args = parser.parse_args()

    t0 = time.time()

    if not args.pg_zip.exists():
        print(f"ERROR: {args.pg_zip} not found")
        print("Download the PG corpus and place PG.zip at the expected path.")
        return

    form_counts = Counter()
    total_tokens = 0
    total_skipped = 0
    file_stats = []

    with zipfile.ZipFile(args.pg_zip, "r") as zf:
        vert_names = sorted(n for n in zf.namelist() if n.endswith(".vert"))
        print(f"Processing {len(vert_names)} .vert files from {args.pg_zip.name}...")

        for i, name in enumerate(vert_names):
            vol = name.split("/")[-1].replace("_tagged_text.vert", "")
            with zf.open(name) as f:
                lines = f.read().decode("utf-8").splitlines()

            tokens, skipped = process_vert_lines(lines, form_counts)
            total_tokens += tokens
            total_skipped += skipped
            file_stats.append((vol, tokens))

            print(f"  [{i+1}/{len(vert_names)}] {vol}: "
                  f"{tokens:,} tokens, "
                  f"running total {total_tokens:,}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal: {total_tokens:,} tokens, "
          f"{len(form_counts):,} unique stripped forms "
          f"({elapsed:.1f}s)")
    print(f"Skipped: {total_skipped:,} non-Greek/malformed lines")

    print(f"\nPer-volume breakdown:")
    for vol, count in sorted(file_stats, key=lambda x: -x[1]):
        print(f"  {vol:30s}: {count:>10,} tokens")

    if args.stats:
        return

    # Format: each form gets a list [total_count] for compatibility
    # with load_corpus_freq_file() which expects forms[form][0]
    forms_dict = {form: [count] for form, count in form_counts.items()}

    output = {
        "_total_tokens": total_tokens,
        "_sources": ["Patrologia Graeca (Byzantine/Patristic, Sketch Engine .vert)"],
        "_n_forms": len(form_counts),
        "forms": forms_dict,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {args.output}...", end=" ", flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = args.output.stat().st_size / 1e6
    print(f"{size_mb:.1f} MB ({time.time() - t0:.1f}s)")

    # Quick sanity check
    print("\nTop 20 forms by frequency:")
    for form, count in form_counts.most_common(20):
        print(f"  {form:20s}: {count:>10,}")


if __name__ == "__main__":
    main()
