#!/usr/bin/env python3
"""Build SQLite lookup database from per-language lookup tables.

Reads from raw_lookups.db (SQLite, written by build_data.py) when available,
falling back to JSON files. Combines AG, MG, and Medieval lookups with
AG-first priority merging.

Output:
    data/lookup.db       - main form->lemma lookup (~1.1 GB)
    data/spell_index.db  - stripped form->original form index for spell checking

Startup: near-instant via mmap (vs ~11s for JSON loading)

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
SPELL_DB_PATH = DATA_DIR / "spell_index.db"
RAW_DB_PATH = DATA_DIR / "raw_lookups.db"

AG_PATH = DATA_DIR / "ag_lookup.json"
AG_HEADWORDS_PATH = DATA_DIR / "ag_headwords.json"
MG_PATH = DATA_DIR / "mg_lookup.json"
MED_PATH = DATA_DIR / "med_lookup.json"
GLAUX_PAIRS_PATH = DATA_DIR / "glaux_pairs.json"
DIORISIS_PAIRS_PATH = DATA_DIR / "diorisis_pairs.json"
PROIEL_PAIRS_PATH = DATA_DIR / "proiel_pairs.json"
GORMAN_PAIRS_PATH = DATA_DIR / "gorman_pairs.json"
ETYMOLOGY_BRIDGES_PATH = DATA_DIR / "etymology_bridges.json"
LSJGR_BRIDGES_PATH = DATA_DIR / "lsjgr_bridges.json"
RELATED_LEMMAS_PATH = DATA_DIR / "related_lemmas.json"
HNC_PAIRS_PATH = DATA_DIR / "hnc_pairs.json"


def strip_accents(s):
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def _is_self_map(form, lemma):
    return (form == lemma
            or strip_accents(form.lower()) == strip_accents(lemma.lower()))


def _load_from_sqlite(table: str) -> dict:
    """Load a lookup table from raw_lookups.db."""
    if not RAW_DB_PATH.exists():
        return {}
    conn = sqlite3.connect(str(RAW_DB_PATH))
    try:
        rows = conn.execute(f"SELECT form, lemma FROM {table}").fetchall()
        conn.close()
        return dict(rows)
    except sqlite3.OperationalError:
        conn.close()
        return {}


def _load_from_json(path: Path) -> dict:
    """Load a lookup table from JSON (fallback)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_lookup(table: str, json_path: Path, label: str) -> dict:
    """Load lookup, preferring SQLite over JSON."""
    t0 = time.time()

    data = _load_from_sqlite(table)
    if data:
        print(f"  {label}: {len(data):,} entries from SQLite ({time.time()-t0:.1f}s)")
        return data

    data = _load_from_json(json_path)
    if data:
        print(f"  {label}: {len(data):,} entries from JSON ({time.time()-t0:.1f}s)")
    else:
        print(f"  {label}: no data found")
    return data


def build():
    t0 = time.time()

    print("Loading lookup tables...")
    ag = _load_lookup("ag", AG_PATH, "AG")
    el = _load_lookup("mg", MG_PATH, "MG")
    med = _load_lookup("med", MED_PATH, "Med")

    # Merge med into el: vernacular medieval Greek is the ancestor of
    # Modern Greek, and EL Wiktionary's "Medieval Greek" category contains
    # early MG vocabulary, not Byzantine literary Greek.
    med_merged = 0
    for k, v in med.items():
        if k not in el:
            el[k] = v
            med_merged += 1
    print(f"  Merged {med_merged:,} med entries into el ({len(el):,} total)")

    # Add HNC Golden Corpus pairs (gold-standard MG annotations).
    # Lower priority than Wiktionary: only add where form is not already present.
    hnc_added = 0
    if HNC_PAIRS_PATH.exists():
        t_h = time.time()
        with open(HNC_PAIRS_PATH, encoding="utf-8") as f:
            hnc_pairs = json.load(f)
        for form, lemma in hnc_pairs.items():
            if form not in el:
                el[form] = lemma
                hnc_added += 1
        print(f"  HNC: +{hnc_added:,} to el "
              f"({len(hnc_pairs):,} total, {len(hnc_pairs) - hnc_added:,} already present) "
              f"({time.time()-t_h:.1f}s)")
    else:
        print(f"  HNC: no hnc_pairs.json found, skipping")

    # Expand AG with PROIEL gold-standard pairs (33K forms from
    # Herodotus' Histories, manually annotated). Higher priority
    # than corpus-derived pairs (GLAUx, Diorisis) since every
    # lemma assignment is expert-verified.
    proiel_added_ag = 0
    if PROIEL_PAIRS_PATH.exists():
        t_p = time.time()
        with open(PROIEL_PAIRS_PATH, encoding="utf-8") as f:
            proiel_pairs = json.load(f)
        for p in proiel_pairs:
            form, lemma = p["form"], p["lemma"]
            if form not in ag:
                ag[form] = lemma
                proiel_added_ag += 1
        print(f"  PROIEL: +{proiel_added_ag:,} to AG "
              f"({len(proiel_pairs):,} total, "
              f"{len(proiel_pairs) - proiel_added_ag:,} already present) "
              f"({time.time()-t_p:.1f}s)")
    else:
        print(f"  PROIEL: no proiel_pairs.json found, skipping")

    # Expand AG with Gorman treebank pairs (79K unique form-lemma pairs
    # from 687K tokens of annotated Ancient Greek across multiple authors:
    # Herodotus, Thucydides, Xenophon, Demosthenes, Lysias, Polybius, etc.)
    # Gold-standard single annotator, same priority tier as PROIEL.
    gorman_added_ag = 0
    if GORMAN_PAIRS_PATH.exists():
        t_gr = time.time()
        with open(GORMAN_PAIRS_PATH, encoding="utf-8") as f:
            gorman_pairs = json.load(f)
        for p in gorman_pairs:
            form, lemma = p["form"], p["lemma"]
            if form not in ag:
                ag[form] = lemma
                gorman_added_ag += 1
        print(f"  Gorman: +{gorman_added_ag:,} to AG "
              f"({len(gorman_pairs):,} total, "
              f"{len(gorman_pairs) - gorman_added_ag:,} already present) "
              f"({time.time()-t_gr:.1f}s)")
    else:
        print(f"  Gorman: no gorman_pairs.json found, skipping")

    # Expand AG and Med with GLAUx corpus pairs (644K forms from
    # 8th c. BC - 4th c. AD Greek texts). These are corpus-derived
    # so lower confidence than Wiktionary, but fill coverage gaps.
    glaux_added_ag = 0
    glaux_added_med = 0
    glaux_skipped_med = 0
    if GLAUX_PAIRS_PATH.exists():
        t_g = time.time()
        with open(GLAUX_PAIRS_PATH, encoding="utf-8") as f:
            glaux_pairs = json.load(f)
        # Snapshot AG keys before GLAUx expansion, so we can check
        # whether a form had an original (Wiktionary) AG entry.
        ag_original = dict(ag)
        for p in glaux_pairs:
            form, lemma = p["form"], p["lemma"]
            # Add to AG if not already present
            if form not in ag:
                ag[form] = lemma
                glaux_added_ag += 1
            # Selectively add to el: only when the pair won't cause
            # a priority override conflict in the combined merge.
            if form not in el:
                if form not in ag_original:
                    el[form] = lemma
                    glaux_added_med += 1
                elif ag_original[form] == lemma:
                    el[form] = lemma
                    glaux_added_med += 1
                else:
                    glaux_skipped_med += 1
        print(f"  GLAUx: +{glaux_added_ag:,} to AG, "
              f"+{glaux_added_med:,} to el, "
              f"{glaux_skipped_med:,} el conflicts skipped "
              f"({time.time()-t_g:.1f}s)")

    # Expand AG and el with Diorisis corpus pairs (456K forms from
    # 10.2M tokens of ancient Greek texts). Lower confidence than GLAUx
    # (91.4% vs 98.8% lemma accuracy), so lowest priority: only added
    # when not already present from Wiktionary, LSJ, or GLAUx.
    dior_added_ag = 0
    dior_added_el = 0
    dior_skipped_el = 0
    dior_skipped_ag = 0
    if DIORISIS_PAIRS_PATH.exists():
        t_d = time.time()
        with open(DIORISIS_PAIRS_PATH, encoding="utf-8") as f:
            diorisis_pairs = json.load(f)
        # Snapshot AG keys before Diorisis expansion (includes Wiktionary + GLAUx)
        ag_before_dior = dict(ag)
        for p in diorisis_pairs:
            form, lemma = p["form"], p["lemma"]
            # Add to AG if not already present from any source
            if form not in ag:
                ag[form] = lemma
                dior_added_ag += 1
            else:
                dior_skipped_ag += 1
            # Selectively add to el: only when the pair won't cause
            # a priority override conflict in the combined merge.
            if form not in el:
                if form not in ag_before_dior:
                    el[form] = lemma
                    dior_added_el += 1
                elif ag_before_dior[form] == lemma:
                    el[form] = lemma
                    dior_added_el += 1
                else:
                    dior_skipped_el += 1
        print(f"  Diorisis: +{dior_added_ag:,} to AG ({dior_skipped_ag:,} skipped), "
              f"+{dior_added_el:,} to el, "
              f"{dior_skipped_el:,} el conflicts skipped "
              f"({time.time()-t_d:.1f}s)")

    # Load AG headwords to protect AG self-maps from EL overrides.
    # AG headwords that self-map (e.g. καθάπερ -> καθάπερ) are correct
    # citation forms and should not be replaced by EL form-of redirects.
    ag_headwords = set()
    if AG_HEADWORDS_PATH.exists():
        with open(AG_HEADWORDS_PATH, encoding="utf-8") as f:
            ag_headwords = set(json.load(f))
        ag_headwords |= {h.lower() for h in ag_headwords}
        ag_headwords |= {strip_accents(h.lower()) for h in ag_headwords}
        print(f"  AG headwords: {len(ag_headwords):,} (for self-map protection)")

    # Article and pronoun forms excluded from the lookup so that
    # resolve_articles=True/False in Dilemma controls their resolution.
    # Without this, fresh Wiktionary data maps τοῦ -> ὁ etc. in the
    # lookup itself, bypassing the resolve_articles flag.
    _EXCLUDED_ARTICLE_MAPS = {
        "ὁ", "ἡ", "τό", "τοῦ", "τῆς", "τῶν", "τόν", "τήν",
        "τά", "τοῖς", "ταῖς", "τῷ", "τῇ", "τούς", "τάς", "τοῖν", "ταῖν",
        "οἱ", "αἱ", "τώ",
        "τὸ", "τοὺς", "τὰ", "τὸν", "τὴν", "τὰς", "αἵ", "οἵ",
    }
    _ARTICLE_LEMMA = "ὁ"

    def _is_article_map(form, lemma):
        return (strip_accents(form.lower()) in {strip_accents(a.lower()) for a in _EXCLUDED_ARTICLE_MAPS}
                and strip_accents(lemma.lower()) == strip_accents(_ARTICLE_LEMMA.lower()))

    # Build combined lookup (AG-first priority)
    print("\nBuilding combined lookup (AG-first)...")
    combined = {}
    ag_protected = 0
    article_excluded = 0
    for data in [ag, el]:
        for k, v in data.items():
            if _is_article_map(k, v):
                article_excluded += 1
                continue
            if k not in combined:
                combined[k] = v
            elif _is_self_map(k, combined[k]) and not _is_self_map(k, v):
                # EL non-self-map overrides AG self-map, UNLESS the AG
                # self-map is a known AG headword (correct citation form).
                if k in ag_headwords or combined[k] in ag_headwords:
                    ag_protected += 1
                else:
                    combined[k] = v
            elif (_is_self_map(k, combined[k])
                  and _is_self_map(k, v) and v == k
                  and combined[k] != k):
                combined[k] = v
    if article_excluded:
        print(f"  Article forms excluded (controlled by resolve_articles): {article_excluded:,}")
    if ag_protected:
        print(f"  AG headword self-maps protected: {ag_protected:,}")
    print(f"  Combined: {len(combined):,} entries")

    # Write SQLite database
    print(f"\nWriting {DB_PATH}...")
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA page_size=4096")

    # Deduplicated lemma table
    all_lemmas = sorted(set(combined.values()) | set(ag.values()) | set(el.values()))
    lemma_to_id = {lemma: i for i, lemma in enumerate(all_lemmas)}
    conn.execute("CREATE TABLE lemmas (id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    conn.executemany("INSERT INTO lemmas (id, text) VALUES (?, ?)",
                     enumerate(all_lemmas))
    print(f"  lemmas: {len(all_lemmas):,} distinct")

    # Main lookup: form -> lemma_id
    conn.execute("""CREATE TABLE lookup (
        form TEXT NOT NULL,
        lemma_id INTEGER NOT NULL,
        src TEXT NOT NULL,
        lang TEXT NOT NULL DEFAULT 'all',
        FOREIGN KEY (lemma_id) REFERENCES lemmas(id)
    )""")

    # Track which source provided each combined entry
    src_map = {}
    for data, src_label in [(ag, 'grc'), (el, 'el')]:
        for k in data:
            if k not in src_map:
                src_map[k] = src_label
            elif _is_self_map(k, combined.get(k, '')) and not _is_self_map(k, data[k]):
                src_map[k] = src_label

    conn.executemany("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?, ?, ?, 'all')",
                     ((k, lemma_to_id[v], src_map.get(k, 'grc')) for k, v in combined.items()))
    print(f"  combined: {len(combined):,} rows")

    # AG-only entries where AG differs from combined (for polytonic-first lookup)
    ag_extra = 0
    for k, v in ag.items():
        if combined.get(k) != v and not _is_article_map(k, v):
            conn.execute("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?, ?, 'grc', 'grc')",
                         (k, lemma_to_id[v]))
            ag_extra += 1
    print(f"  grc-only (differs from combined): {ag_extra:,} rows")

    # MG-only entries where MG differs from combined (for lang="el" mode)
    mg_extra = 0
    for k, v in el.items():
        if combined.get(k) != v and not _is_article_map(k, v):
            conn.execute("INSERT INTO lookup (form, lemma_id, src, lang) VALUES (?, ?, 'el', 'el')",
                         (k, lemma_to_id[v]))
            mg_extra += 1
    print(f"  el-only (differs from combined): {mg_extra:,} rows")

    conn.execute("CREATE INDEX idx_lookup_form_lang ON lookup (form, lang)")
    conn.execute("CREATE INDEX idx_lemmas_text ON lemmas (text)")

    # Etymology bridges: MG lemma -> AG ancestor lemma
    conn.execute("""CREATE TABLE bridges (
        el_lemma_id INTEGER NOT NULL,
        grc_lemma_id INTEGER NOT NULL,
        FOREIGN KEY (el_lemma_id) REFERENCES lemmas(id),
        FOREIGN KEY (grc_lemma_id) REFERENCES lemmas(id)
    )""")
    bridge_pairs = set()
    bridge_skipped = 0
    if ETYMOLOGY_BRIDGES_PATH.exists():
        with open(ETYMOLOGY_BRIDGES_PATH, encoding="utf-8") as f:
            bridges = json.load(f)
        for mg_lemma, ag_ancestors in bridges.items():
            el_id = lemma_to_id.get(mg_lemma)
            if el_id is None:
                bridge_skipped += 1
                continue
            for ag_lemma in ag_ancestors:
                grc_id = lemma_to_id.get(ag_lemma)
                if grc_id is None:
                    bridge_skipped += 1
                    continue
                bridge_pairs.add((el_id, grc_id))
        etym_count = len(bridge_pairs)
        print(f"  etymology bridges: {etym_count:,} pairs ({bridge_skipped:,} skipped)")
    else:
        etym_count = 0
        print(f"  etymology bridges: not found")

    lsjgr_new = 0
    lsjgr_skipped = 0
    lsjgr_dup = 0
    if LSJGR_BRIDGES_PATH.exists():
        with open(LSJGR_BRIDGES_PATH, encoding="utf-8") as f:
            lsjgr_bridges = json.load(f)
        for mg_lemma, ag_ancestors in lsjgr_bridges.items():
            el_id = lemma_to_id.get(mg_lemma)
            if el_id is None:
                lsjgr_skipped += 1
                continue
            for ag_lemma in ag_ancestors:
                grc_id = lemma_to_id.get(ag_lemma)
                if grc_id is None:
                    lsjgr_skipped += 1
                    continue
                pair = (el_id, grc_id)
                if pair in bridge_pairs:
                    lsjgr_dup += 1
                else:
                    bridge_pairs.add(pair)
                    lsjgr_new += 1
        print(f"  lsjgr bridges: {lsjgr_new:,} new, {lsjgr_dup:,} dup, {lsjgr_skipped:,} skipped")
    else:
        print(f"  lsjgr bridges: not found")

    for el_id, grc_id in bridge_pairs:
        conn.execute("INSERT INTO bridges (el_lemma_id, grc_lemma_id) VALUES (?, ?)",
                     (el_id, grc_id))
    print(f"  bridges total: {len(bridge_pairs):,} (etymology: {etym_count:,}, lsjgr: {lsjgr_new:,})")
    conn.execute("CREATE INDEX idx_bridges_el ON bridges (el_lemma_id)")
    conn.execute("CREATE INDEX idx_bridges_grc ON bridges (grc_lemma_id)")

    # Related lemmas
    conn.execute("""CREATE TABLE related_lemmas (
        lemma_id INTEGER NOT NULL,
        related_id INTEGER NOT NULL,
        FOREIGN KEY (lemma_id) REFERENCES lemmas(id),
        FOREIGN KEY (related_id) REFERENCES lemmas(id)
    )""")
    related_count = 0
    related_skipped = 0
    if RELATED_LEMMAS_PATH.exists():
        with open(RELATED_LEMMAS_PATH, encoding="utf-8") as f:
            related_data = json.load(f)
        seen_pairs = set()
        for lemma, related_list in related_data.items():
            lemma_id = lemma_to_id.get(lemma)
            if lemma_id is None:
                related_skipped += 1
                continue
            for related in related_list:
                related_id = lemma_to_id.get(related)
                if related_id is None:
                    related_skipped += 1
                    continue
                pair = (lemma_id, related_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    conn.execute("INSERT INTO related_lemmas (lemma_id, related_id) VALUES (?, ?)",
                                 (lemma_id, related_id))
                    related_count += 1
        print(f"  related_lemmas: {related_count:,} pairs ({related_skipped:,} skipped)")
    else:
        print(f"  related_lemmas: not found")
    conn.execute("CREATE INDEX idx_related_lemma ON related_lemmas (lemma_id)")
    conn.execute("CREATE INDEX idx_related_related ON related_lemmas (related_id)")

    # Compact main DB
    print("\nOptimizing lookup.db...")
    conn.commit()
    conn.execute("ANALYZE")
    conn.execute("VACUUM")
    conn.commit()
    conn.close()

    size_mb = DB_PATH.stat().st_size / 1e6
    print(f"  lookup.db: {size_mb:.1f} MB")

    # Build separate spell index (stripped form -> original forms)
    # Groups all polytonic variants under each stripped form, with src
    # tags for AG-mode filtering. Uses a compact single-row-per-stripped
    # format to minimize size.
    print("\nBuilding spell_index.db...")
    if SPELL_DB_PATH.exists():
        SPELL_DB_PATH.unlink()

    spell_conn = sqlite3.connect(str(SPELL_DB_PATH))
    spell_conn.execute("PRAGMA journal_mode=DELETE")
    spell_conn.execute("PRAGMA page_size=4096")
    # Each stripped form gets one row. `forms` is newline-separated list
    # of "form\tsrc" pairs (or just "form" when src is empty).
    spell_conn.execute("""CREATE TABLE spell (
        stripped TEXT PRIMARY KEY,
        forms TEXT NOT NULL
    ) WITHOUT ROWID""")

    # Read all forms from main DB and group by stripped form
    main_conn = sqlite3.connect(str(DB_PATH))
    main_conn.execute("PRAGMA mmap_size=268435456")
    rows = main_conn.execute(
        "SELECT DISTINCT form, src FROM lookup").fetchall()
    main_conn.close()

    grouped: dict[str, list[tuple[str, str]]] = {}
    for form, src in rows:
        stripped = strip_accents(form.lower())
        if stripped not in grouped:
            grouped[stripped] = []
        grouped[stripped].append((form, src))

    # Deduplicate within each group
    spell_rows = []
    for stripped, pairs in grouped.items():
        seen: set[str] = set()
        parts = []
        for form, src in pairs:
            if form not in seen:
                seen.add(form)
                parts.append(f"{form}\t{src}" if src else form)
        spell_rows.append((stripped, "\n".join(parts)))

    spell_conn.executemany(
        "INSERT INTO spell (stripped, forms) VALUES (?, ?)", spell_rows)
    print(f"  unique stripped forms: {len(spell_rows):,}")

    spell_conn.commit()
    spell_conn.execute("ANALYZE")
    spell_conn.execute("VACUUM")
    spell_conn.close()

    spell_mb = SPELL_DB_PATH.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"  spell_index.db: {spell_mb:.1f} MB")
    print(f"\nDone ({elapsed:.1f}s, total: {size_mb + spell_mb:.1f} MB)")


if __name__ == "__main__":
    build()
