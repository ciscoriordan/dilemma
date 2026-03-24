#!/usr/bin/env python3
"""Merge GLAUx and Diorisis corpus frequencies into a combined file.

Both corpora use the same format: accent-stripped lowercase forms mapped to
frequency arrays [total, philosophy, poetry, history, oratory, science,
narrative, epistles, religion, commentary, other].

The merged output uses the same format and same genre order, so it's a
drop-in replacement for glaux_freq.json. Token counts are summed across
both corpora.

Output: data/corpus_freq.json (combined GLAUx 17M + Diorisis 10M = ~27M tokens)

To use the merged frequencies, either:
  1. Point GLAUX_FREQ_PATH / FREQ_PATH at corpus_freq.json, or
  2. Replace glaux_freq.json with the merged file:
     cp data/corpus_freq.json data/glaux_freq.json

Usage:
    python build/merge_corpus_freq.py
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"

GLAUX_FREQ_PATH = DATA_DIR / "glaux_freq.json"
DIORISIS_FREQ_PATH = DATA_DIR / "diorisis_freq.json"
OUTPUT_PATH = DATA_DIR / "corpus_freq.json"

# Same genre order used by both files
GENRE_ORDER = [
    "philosophy", "poetry", "history", "oratory", "science",
    "narrative", "epistles", "religion", "commentary", "other",
]


def load_freq(path):
    """Load a frequency file and return (total_tokens, forms_dict)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    total = data.get("_total_tokens", 0)
    genres = data.get("_genres", [])
    forms = data.get("forms", {})

    # Verify genre order matches
    if genres != GENRE_ORDER:
        print(f"  WARNING: Genre order mismatch in {path.name}")
        print(f"    Expected: {GENRE_ORDER}")
        print(f"    Got:      {genres}")

    return total, forms


def main():
    parser = argparse.ArgumentParser(
        description="Merge GLAUx + Diorisis frequencies")
    parser.add_argument("--glaux", type=Path, default=GLAUX_FREQ_PATH)
    parser.add_argument("--diorisis", type=Path, default=DIORISIS_FREQ_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    t0 = time.time()
    n_genres = len(GENRE_ORDER)
    vec_len = 1 + n_genres  # [total, g1, g2, ...]

    # Load GLAUx
    print(f"Loading {args.glaux.name}...", end=" ", flush=True)
    glaux_total, glaux_forms = load_freq(args.glaux)
    print(f"{glaux_total:,} tokens, {len(glaux_forms):,} forms")

    # Load Diorisis
    print(f"Loading {args.diorisis.name}...", end=" ", flush=True)
    dior_total, dior_forms = load_freq(args.diorisis)
    print(f"{dior_total:,} tokens, {len(dior_forms):,} forms")

    # Merge: sum frequency vectors element-wise
    print("Merging...", end=" ", flush=True)
    merged = {}
    all_forms = set(glaux_forms.keys()) | set(dior_forms.keys())

    glaux_only = 0
    dior_only = 0
    both = 0

    for form in all_forms:
        g = glaux_forms.get(form)
        d = dior_forms.get(form)

        if g and d:
            # Both corpora have this form: sum the vectors
            merged[form] = [g[i] + d[i] for i in range(vec_len)]
            both += 1
        elif g:
            merged[form] = list(g)  # copy
            glaux_only += 1
        else:
            merged[form] = list(d)  # copy
            dior_only += 1

    combined_total = glaux_total + dior_total
    print(f"{len(merged):,} unique forms")

    print(f"\nOverlap statistics:")
    print(f"  GLAUx only:  {glaux_only:,}")
    print(f"  Diorisis only: {dior_only:,}")
    print(f"  Both corpora:  {both:,}")
    print(f"  Combined tokens: {combined_total:,} "
          f"(GLAUx {glaux_total:,} + Diorisis {dior_total:,})")

    print(f"\nGenre distribution (combined):")
    for i, g in enumerate(GENRE_ORDER):
        total_g = sum(v[1 + i] for v in merged.values())
        print(f"  {g:15s}: {total_g:>12,} tokens")

    # Write output
    print(f"\nWriting {args.output}...", end=" ", flush=True)
    output = {
        "_total_tokens": combined_total,
        "_genres": GENRE_ORDER,
        "_n_forms": len(merged),
        "_sources": ["GLAUx (17M tokens)", "Diorisis (10M tokens)"],
        "forms": merged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = args.output.stat().st_size / 1e6
    print(f"{size_mb:.0f} MB ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
