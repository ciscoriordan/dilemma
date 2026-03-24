#!/usr/bin/env python3
"""Scroll word finder for Herculaneum papyri.

Given a partial letter sequence from a carbonized scroll, finds candidate
Greek words that match the visible letters, ranked by estimated frequency.

Pattern syntax:
    .  = any single letter
    *  = any sequence of letters (0 or more)
    [αο] = either α or ο (character class)
    All other characters match literally (accent-insensitive)

Examples:
    φιλοσοφ.α     -> φιλοσοφία
    ..θρωπ.ς      -> ἄνθρωπος, ἀνθρώπως, ...
    επι*ια        -> ἐπιθυμία, ἐπιστημία, ...
    λ[εη]γ.       -> λέγε, λέγω, λῆγε, ...

Usage:
    python vesuvius.py                    # interactive mode
    python vesuvius.py "φιλο.οφ*"        # single query
    python vesuvius.py --limit 50 "..."   # more results
    python vesuvius.py --build-index      # pre-build fast index (run once)
"""

import bisect
import gzip
import json
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "lookup.db"
INDEX_PATH = Path(__file__).parent / "data" / "vesuvius_index.json.gz"
FREQ_PATH = Path(__file__).parent / "data" / "glaux_freq.json"

# Genre index in glaux_freq.json (matches build/build_glaux_freq.py GENRE_ORDER)
GENRES = [
    "philosophy", "poetry", "history", "oratory", "science",
    "narrative", "epistles", "religion", "commentary", "other",
]


def strip_accents(s: str) -> str:
    """Remove all combining diacriticals (accents, breathings)."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def _build_index(db_path, index_path):
    """Pre-build a compact binary index from lookup.db.

    The index is a JSON file with:
    - forms: dict of stripped -> [polytonic, lemma, score]
    - by_length: dict of length -> sorted list of stripped forms

    This takes ~60s to build from SQLite but loads in ~5s from disk.
    """
    t0 = time.time()
    conn = sqlite3.connect(str(db_path))

    # Lemma fan-out
    print("Computing lemma fan-out...", end=" ", flush=True)
    fanout = {}
    for lemma_id, count in conn.execute(
        "SELECT lemma_id, COUNT(DISTINCT stripped) FROM lookup "
        "WHERE lang IN ('all', 'grc') AND stripped IS NOT NULL "
        "GROUP BY lemma_id"
    ):
        fanout[lemma_id] = count
    print(f"{len(fanout):,} lemmas")

    # Load all AG forms, dedup by stripped
    print("Loading forms...", end=" ", flush=True)
    forms = {}
    n_rows = 0
    for stripped, form, lemma, lemma_id, src in conn.execute("""
        SELECT l.stripped, l.form, m.text, l.lemma_id, l.src
        FROM lookup l
        JOIN lemmas m ON l.lemma_id = m.id
        WHERE l.lang IN ('all', 'grc')
          AND l.stripped IS NOT NULL
        ORDER BY l.stripped
    """):
        fan = fanout.get(lemma_id, 1)
        is_headword = (stripped == strip_accents(lemma.lower()))
        score = fan + (500 if is_headword else 0)
        if stripped not in forms or score > forms[stripped][2]:
            forms[stripped] = [form, lemma, score]
        n_rows += 1
    conn.close()
    print(f"{len(forms):,} unique ({n_rows:,} rows)")

    # Bucket by length
    print("Bucketing by length...", end=" ", flush=True)
    by_length = {}
    for stripped in forms:
        n = len(stripped)
        if n not in by_length:
            by_length[n] = []
        by_length[n].append(stripped)
    for n in by_length:
        by_length[n].sort()
    # Convert int keys to strings for JSON
    by_length_str = {str(k): v for k, v in by_length.items()}
    print("done")

    # Write as gzipped TSV (compact: ~60MB, loads in ~8s)
    # Format: one line per form, tab-separated: stripped\tpolytonic\tlemma\tscore
    print(f"Writing {index_path}...", end=" ", flush=True)
    with gzip.open(index_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for stripped in sorted(forms):
            entry = forms[stripped]
            f.write(f"{stripped}\t{entry[0]}\t{entry[1]}\t{entry[2]}\n")
    size_mb = index_path.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"{size_mb:.0f} MB ({elapsed:.1f}s)")


class ScrollFinder:
    """Pattern matcher for Ancient Greek word forms.

    Loads all AG forms from a pre-built index (fast) or directly from
    the Dilemma lookup database (slow first time). Forms are bucketed
    by length for fast pattern matching.

    Ranking: if GLAUx frequency data is available, uses real corpus
    token counts (optionally weighted by genre). Falls back to lemma
    fan-out as a frequency proxy.
    """

    def __init__(self, db_path=DB_PATH, index_path=INDEX_PATH,
                 freq_path=FREQ_PATH, genre=None):
        """
        Args:
            genre: Optional genre filter for ranking. One of:
                   philosophy, poetry, history, oratory, science,
                   narrative, epistles, religion, commentary, other.
                   When set, forms are ranked by frequency in that genre.
        """
        t0 = time.time()
        self._forms = {}       # stripped -> [polytonic, lemma, score]
        self._by_length = {}   # length -> sorted list of stripped forms
        self._genre = genre

        if index_path.exists():
            self._load_index(index_path)
        else:
            print(f"No index found at {index_path}")
            print("Building from SQLite (run with --build-index for fast startup)")
            self._load_sqlite(db_path)

        # Overlay real corpus frequencies if available
        if freq_path.exists():
            self._load_freq(freq_path, genre)

        self._load_time = time.time() - t0

    def _load_index(self, index_path):
        """Load from pre-built gzipped TSV index."""
        print("Loading index...", end=" ", flush=True)
        forms = {}
        by_length = {}
        with gzip.open(index_path, "rt", encoding="utf-8") as f:
            for line in f:
                stripped, polytonic, lemma, score = line.rstrip("\n").split("\t")
                forms[stripped] = [polytonic, lemma, int(score)]
                n = len(stripped)
                if n not in by_length:
                    by_length[n] = []
                by_length[n].append(stripped)
        # Input is pre-sorted, so buckets are already sorted
        self._forms = forms
        self._by_length = by_length
        print(f"{len(forms):,} forms")

    def _load_sqlite(self, db_path):
        """Load directly from SQLite (slow, ~60s)."""
        conn = sqlite3.connect(str(db_path))

        print("Loading lemma statistics...", end=" ", flush=True)
        fanout = {}
        for lemma_id, count in conn.execute(
            "SELECT lemma_id, COUNT(DISTINCT stripped) FROM lookup "
            "WHERE lang IN ('all', 'grc') AND stripped IS NOT NULL "
            "GROUP BY lemma_id"
        ):
            fanout[lemma_id] = count
        print(f"{len(fanout):,} lemmas")

        print("Loading forms...", end=" ", flush=True)
        forms = {}
        n_rows = 0
        for stripped, form, lemma, lemma_id, src in conn.execute("""
            SELECT l.stripped, l.form, m.text, l.lemma_id, l.src
            FROM lookup l
            JOIN lemmas m ON l.lemma_id = m.id
            WHERE l.lang IN ('all', 'grc')
              AND l.stripped IS NOT NULL
            ORDER BY l.stripped
        """):
            fan = fanout.get(lemma_id, 1)
            is_headword = (stripped == strip_accents(lemma.lower()))
            score = fan + (500 if is_headword else 0)
            if stripped not in forms or score > forms[stripped][2]:
                forms[stripped] = [form, lemma, score]
            n_rows += 1
        conn.close()
        self._forms = forms
        print(f"{len(forms):,} unique ({n_rows:,} rows)")

        print("Indexing by length...", end=" ", flush=True)
        by_length = {}
        for stripped in forms:
            n = len(stripped)
            if n not in by_length:
                by_length[n] = []
            by_length[n].append(stripped)
        for n in by_length:
            by_length[n].sort()
        self._by_length = by_length
        print("done")

    def _load_freq(self, freq_path, genre=None):
        """Overlay real corpus frequencies from GLAUx onto scores.

        If genre is specified, uses that genre's token count. Otherwise
        uses total corpus frequency. The corpus frequency replaces the
        fan-out proxy score for forms that appear in GLAUx.
        """
        print(f"Loading corpus frequencies"
              f"{f' (genre={genre})' if genre else ''}...", end=" ", flush=True)
        with open(freq_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        freq_forms = data["forms"]
        genres = data.get("_genres", GENRES)

        if genre and genre in genres:
            genre_idx = genres.index(genre) + 1  # +1 because [0] is total
        else:
            genre_idx = 0  # total

        # Update scores: corpus frequency (scaled) replaces fan-out proxy
        n_updated = 0
        for stripped, entry in self._forms.items():
            freq_entry = freq_forms.get(stripped)
            if freq_entry:
                count = freq_entry[genre_idx]
                # Headword bonus still applies
                is_headword = entry[2] >= 500  # had headword bonus
                hw_bonus = 500 if is_headword else 0
                # Use corpus count directly as score (+ headword bonus)
                entry[2] = count + hw_bonus
                n_updated += 1

        print(f"{n_updated:,}/{len(self._forms):,} forms updated")

    def find(self, pattern: str, limit: int = 20) -> list[dict]:
        """Find words matching a pattern.

        Args:
            pattern: Pattern with . for single unknown, * for variable gap,
                     [αο] for character classes. Letters match accent-free.
            limit: Max results to return.

        Returns:
            List of dicts with keys: form, lemma, score, stripped.
        """
        # Normalize input: strip accents, lowercase
        clean = strip_accents(pattern.lower())

        # Determine which length buckets to search
        has_star = "*" in clean
        if has_star:
            parts = clean.split("*")
            min_len = sum(self._part_len(p) for p in parts)
            max_len = max(self._by_length.keys()) if self._by_length else 30
            lengths = range(min_len, max_len + 1)
        else:
            fixed_len = self._pattern_len(clean)
            lengths = [fixed_len]

        # Build regex from pattern
        regex = self._pattern_to_regex(clean)
        try:
            compiled = re.compile("^" + regex + "$")
        except re.error as e:
            return [{"error": f"Invalid pattern: {e}"}]

        # Search matching buckets
        matches = []
        prefix = self._extract_prefix(clean)

        for n in lengths:
            bucket = self._by_length.get(n, [])
            if not bucket:
                continue

            # Optimization: if pattern starts with fixed chars, use bisect
            if len(prefix) >= 2:
                lo = bisect.bisect_left(bucket, prefix)
                prefix_hi = prefix[:-1] + chr(ord(prefix[-1]) + 1)
                hi = bisect.bisect_left(bucket, prefix_hi)
                search_space = bucket[lo:hi]
            else:
                search_space = bucket

            for stripped in search_space:
                if compiled.match(stripped):
                    entry = self._forms[stripped]
                    matches.append({
                        "form": entry[0],
                        "lemma": entry[1],
                        "score": entry[2],
                        "stripped": stripped,
                    })

        # Sort by score descending, then by length ascending
        matches.sort(key=lambda m: (-m["score"], len(m["stripped"])))
        return matches[:limit]

    def _pattern_to_regex(self, pattern: str) -> str:
        """Convert pattern syntax to regex."""
        result = []
        i = 0
        while i < len(pattern):
            ch = pattern[i]
            if ch == ".":
                result.append("[α-ωϊϋ]")
            elif ch == "*":
                result.append("[α-ωϊϋ]*")
            elif ch == "[":
                end = pattern.index("]", i)
                result.append(pattern[i:end + 1])
                i = end
            else:
                result.append(re.escape(ch))
            i += 1
        return "".join(result)

    def _pattern_len(self, pattern: str) -> int:
        """Count the fixed length of a pattern (no * wildcards)."""
        n = 0
        i = 0
        while i < len(pattern):
            if pattern[i] == "[":
                i = pattern.index("]", i) + 1
            else:
                i += 1
            n += 1
        return n

    def _part_len(self, part: str) -> int:
        """Length of a pattern part (between * wildcards)."""
        return self._pattern_len(part) if part else 0

    def _extract_prefix(self, pattern: str) -> str:
        """Extract the fixed prefix before any wildcard."""
        prefix = []
        for ch in pattern:
            if ch in ".*[":
                break
            prefix.append(ch)
        return "".join(prefix)


def interactive(finder, limit=20):
    """Interactive query loop."""
    genre_str = f", genre={finder._genre}" if finder._genre else ""
    print(f"\nScroll Word Finder ({len(finder._forms):,} AG forms{genre_str}, "
          f"loaded in {finder._load_time:.1f}s)")
    print("Pattern: . = unknown letter, * = variable gap, [αο] = either")
    print("Type 'q' to quit.\n")

    while True:
        try:
            pattern = input("pattern> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not pattern or pattern == "q":
            break

        t0 = time.time()
        results = finder.find(pattern, limit=limit)
        elapsed = time.time() - t0

        if not results:
            print(f"  No matches ({elapsed:.3f}s)\n")
            continue

        if "error" in results[0]:
            print(f"  {results[0]['error']}\n")
            continue

        for i, r in enumerate(results):
            print(f"  {i+1:3d}. {r['form']:20s} -> {r['lemma']:20s} "
                  f"(score={r['score']:>5d})")
        print(f"  [{len(results)} results, {elapsed:.3f}s]\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scroll word finder for Herculaneum papyri")
    parser.add_argument("pattern", nargs="?", help="Search pattern")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max results (default: 20)")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Path to lookup.db")
    parser.add_argument("--genre", type=str, default=None,
                        choices=GENRES,
                        help="Weight results by genre frequency")
    parser.add_argument("--build-index", action="store_true",
                        help="Build fast-loading index from lookup.db")
    args = parser.parse_args()

    if args.build_index:
        _build_index(Path(args.db), INDEX_PATH)
        return

    finder = ScrollFinder(Path(args.db), INDEX_PATH, FREQ_PATH, genre=args.genre)

    if args.pattern:
        results = finder.find(args.pattern, limit=args.limit)
        for i, r in enumerate(results):
            print(f"{i+1:3d}. {r['form']:20s} -> {r['lemma']:20s} "
                  f"(score={r['score']:>5d})")
    else:
        interactive(finder, limit=args.limit)


if __name__ == "__main__":
    main()
