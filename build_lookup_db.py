#!/usr/bin/env python3
"""Build SQLite lookup database from JSON lookup tables.

Combines ag_lookup.json, mg_lookup.json, and med_lookup.json into a single
SQLite database with the priority merging logic from dilemma.py.

Output: data/lookup.db (~200 MB, vs ~600 MB JSON)
Startup: near-instant (vs ~11s for JSON loading)

Usage:
    python build_lookup_db.py
"""

import json
import sqlite3
import time
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DB_PATH = DATA_DIR / "lookup.db"

AG_PATH = DATA_DIR / "ag_lookup.json"
MG_PATH = DATA_DIR / "mg_lookup.json"
MED_PATH = DATA_DIR / "med_lookup.json"


def strip_accents(s):
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def _is_self_map(form, lemma):
    return (form == lemma
            or strip_accents(form.lower()) == strip_accents(lemma.lower()))


def build():
    t0 = time.time()

    # Load JSON tables
    print("Loading JSON lookup tables...")
    ag, mg, med = {}, {}, {}
    if AG_PATH.exists():
        with open(AG_PATH, encoding="utf-8") as f:
            ag = json.load(f)
        print(f"  AG: {len(ag):,} entries ({time.time()-t0:.1f}s)")
    t1 = time.time()
    if MG_PATH.exists():
        with open(MG_PATH, encoding="utf-8") as f:
            mg = json.load(f)
        print(f"  MG: {len(mg):,} entries ({time.time()-t1:.1f}s)")
    t2 = time.time()
    if MED_PATH.exists():
        with open(MED_PATH, encoding="utf-8") as f:
            med = json.load(f)
        print(f"  Med: {len(med):,} entries ({time.time()-t2:.1f}s)")

    # Build combined lookup (AG-first priority, same logic as dilemma.py)
    print("\nBuilding combined lookup (AG-first)...")
    combined = {}
    for data in [ag, med, mg]:
        for k, v in data.items():
            if k not in combined:
                combined[k] = v
            elif _is_self_map(k, combined[k]) and not _is_self_map(k, v):
                combined[k] = v
            elif (_is_self_map(k, combined[k])
                  and _is_self_map(k, v) and v == k
                  and combined[k] != k):
                combined[k] = v
    print(f"  Combined: {len(combined):,} entries")

    # Write SQLite database
    print(f"\nWriting {DB_PATH}...")
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA page_size=4096")

    # Deduplicated lemma table - ~200K distinct lemmas vs 12.5M form entries
    all_lemmas = sorted(set(combined.values()) | set(ag.values()))
    lemma_to_id = {lemma: i for i, lemma in enumerate(all_lemmas)}
    conn.execute("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    conn.executemany("INSERT INTO lemmas (id, text) VALUES (?, ?)",
                     enumerate(all_lemmas))
    print(f"  lemmas: {len(all_lemmas):,} distinct")

    # Main lookup: form -> lemma_id
    # src: source language that won in the merge ('a'=AG, 'e'=MG, 'm'=med)
    conn.execute("""CREATE TABLE lookup (
        form TEXT NOT NULL,
        lemma_id INTEGER NOT NULL,
        src CHAR(1) NOT NULL,
        lang CHAR(1) NOT NULL DEFAULT 'c',
        FOREIGN KEY (lemma_id) REFERENCES lemmas(id)
    )""")

    # Track which source provided each combined entry
    src_map = {}  # form -> source char
    for data, src_char in [(ag, 'a'), (med, 'm'), (mg, 'e')]:
        for k in data:
            if k not in src_map:
                src_map[k] = src_char
            elif _is_self_map(k, combined.get(k, '')) and not _is_self_map(k, data[k]):
                src_map[k] = src_char

    conn.executemany("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?, ?, ?, 'c')",
                     ((k, lemma_to_id[v], src_map.get(k, 'a')) for k, v in combined.items()))
    print(f"  combined: {len(combined):,} rows")

    # AG-only entries where AG differs from combined
    ag_extra = 0
    for k, v in ag.items():
        if combined.get(k) != v:
            conn.execute("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?, ?, 'a', 'a')",
                         (k, lemma_to_id[v]))
            ag_extra += 1
    print(f"  ag-only (differs from combined): {ag_extra:,} rows")

    conn.execute("CREATE INDEX idx_lookup_form_lang ON lookup (form, lang)")
    # For full-text search on lemmas (occasional use)
    conn.execute("CREATE INDEX idx_lemmas_text ON lemmas (text)")

    conn.commit()

    # Compact
    print("\nOptimizing...")
    conn.execute("ANALYZE")
    conn.execute("VACUUM")
    conn.commit()
    conn.close()

    size_mb = DB_PATH.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"\nDone: {DB_PATH} ({size_mb:.1f} MB, {elapsed:.1f}s)")


if __name__ == "__main__":
    build()
