#!/usr/bin/env python3
"""Overlay LSJ expansion entries from a reference ag_lookup.json onto a new base.

The LSJ expansion (expand_lsj.py --expand, --expand-verbs, participles, etc.)
only ADDS entries that don't already exist in the base. So we can extract the
difference between the reference (committed) lookup and a known Wiktionary-only
base, and apply those additions to any new base.

Usage:
    python overlay_lsj.py <reference_lookup> <new_base_lookup>

This modifies <new_base_lookup> in-place, adding entries from <reference_lookup>
that are not already present.
"""
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python overlay_lsj.py <reference_lookup.json> <new_base_lookup.json>")
        sys.exit(1)

    ref_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])

    print(f"Loading reference: {ref_path}")
    with open(ref_path) as f:
        ref = json.load(f)
    print(f"  {len(ref):,} entries")

    print(f"Loading new base: {new_path}")
    with open(new_path) as f:
        new = json.load(f)
    print(f"  {len(new):,} entries")

    # Add entries from reference that are not in new base
    added = 0
    for k, v in ref.items():
        if k not in new:
            new[k] = v
            added += 1

    print(f"Added {added:,} entries from reference")
    print(f"Final size: {len(new):,}")

    print(f"Saving to {new_path}...")
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(new, f, ensure_ascii=False)
    size_mb = new_path.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")

if __name__ == "__main__":
    main()
