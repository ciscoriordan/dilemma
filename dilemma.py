"""Dilemma - Greek lemmatizer.

Fast lookup table for known forms, custom transformer model for unknown forms.

Usage:
    from dilemma import Dilemma

    m = Dilemma()                        # loads lookup table + model
    m.lemmatize("πάθης")                # -> "παθαίνω"
    m.lemmatize("πολεμούσαν")           # -> "πολεμώ"
    m.lemmatize_batch(["δώση", "σκότωσε"])  # -> ["δίνω", "σκοτώνω"]

    # Elision expansion (uses Wiktionary lookup)
    m.lemmatize("ἀλλ̓")                  # -> "ἀλλά"
    m.lemmatize("ἔφατ̓")                 # -> "φημί"

    # Verbose mode: returns all candidates with metadata
    m.lemmatize_verbose("ἔριδι")
    # -> [LemmaCandidate(lemma="ἔρις", lang="grc", proper=False),
    #     LemmaCandidate(lemma="Ἔρις", lang="grc", proper=True)]

    m.lemmatize_verbose("πόλεμο")
    # -> [LemmaCandidate(lemma="πόλεμος", lang="el"),
    #     LemmaCandidate(lemma="πόλεμος", lang="grc")]

    # Convention remapping: LSJ dictionary headwords
    m_lsj = Dilemma(convention="lsj")
    m_lsj.lemmatize("αἰνῶς")        # -> "αἰνός" (adverb -> adjective)
    m_lsj.lemmatize("εἶπον")        # -> "λέγω" (aorist -> present stem)
"""

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "model"
LOOKUP_DB_PATH = Path(__file__).parent / "data" / "lookup.db"
LSJ9_POS_LOOKUP_PATH = Path(__file__).parent / "data" / "lsj9_pos_lookup.json"
LSJ9_FREQUENCY_PATH = (Path(__file__).parent.parent / "lsjpre" / "output"
                       / "lsj9" / "lsj9_frequency.json")
LOOKUP_PATH = Path(__file__).parent / "data" / "mg_lookup.json"
AG_LOOKUP_PATH = Path(__file__).parent / "data" / "ag_lookup.json"
MED_LOOKUP_PATH = Path(__file__).parent / "data" / "med_lookup.json"
MG_POS_LOOKUP_PATH = Path(__file__).parent / "data" / "mg_pos_lookup.json"
AG_POS_LOOKUP_PATH = Path(__file__).parent / "data" / "ag_pos_lookup.json"
TREEBANK_POS_LOOKUP_PATH = Path(__file__).parent / "data" / "treebank_pos_lookup.json"
GLAUX_POS_LOOKUP_PATH = Path(__file__).parent / "data" / "glaux_pos_lookup.json"
LSJ_HEADWORDS_PATH = Path(__file__).parent / "data" / "lsj_headwords.json"
CUNLIFFE_HEADWORDS_PATH = Path(__file__).parent / "data" / "cunliffe_headwords.json"
LEMMA_EQUIVALENCES_PATH = Path(__file__).parent / "data" / "lemma_equivalences.json"
CONVENTION_DIR = Path(__file__).parent / "data"

_VALID_CONVENTIONS = {None, "lsj", "wiktionary"}


_POLYTONIC_STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
_POLYTONIC_TO_ACUTE = {0x0300, 0x0342}

# Elision mark: U+0313 COMBINING COMMA ABOVE (repurposed as apostrophe
# in polytonic Greek text). Also handle right single quote U+2019 and
# modifier letter apostrophe U+02BC.
_ELISION_MARKS = {"\u0313", "\u2019", "\u02BC", "'", "\u1FBD", "\u02B9"}

# Vowels to try when expanding elision (ordered by frequency in AG text)
_GREEK_VOWELS = "αεοιηυω"

# Article and pronoun resolution: maps forms to canonical lemma.
# Used when resolve_articles=True (for treebank evaluation).
_ARTICLE_LEMMA = "ὁ"
_ARTICLE_FORMS = {
    # Polytonic
    "ὁ", "ἡ", "τό", "τοῦ", "τῆς", "τῶν", "τόν", "τήν",
    "τά", "τοῖς", "ταῖς", "τῷ", "τῇ", "τούς", "τάς", "τοῖν", "ταῖν",
    "οἱ", "αἱ", "τώ",
    # Grave variants
    "τὸ", "τοὺς", "τὰ", "τὸν", "τὴν", "τὰς", "αἵ", "οἵ",
    # Monotonic
    "ο", "η", "το", "του", "της", "των", "τον", "την",
    "τα", "τους", "τοις", "οι", "αι",
    # Stripped (no accents/breathings)
    "τω", "ται",
}

_PRONOUN_LEMMAS = {
    # 1st person -> ἐγώ
    "μοι": "ἐγώ", "μοί": "ἐγώ", "μου": "ἐγώ", "με": "ἐγώ",
    "ἐμοί": "ἐγώ", "ἐμοῦ": "ἐγώ", "ἐμέ": "ἐγώ",
    "ἡμεῖς": "ἐγώ", "ἡμῶν": "ἐγώ", "ἡμῖν": "ἐγώ", "ἡμᾶς": "ἐγώ",
    # 2nd person -> σύ
    "σοι": "σύ", "σοί": "σύ", "σου": "σύ", "σε": "σύ",
    "σοῦ": "σύ",
    "ὑμεῖς": "σύ", "ὑμῶν": "σύ", "ὑμῖν": "σύ", "ὑμᾶς": "σύ",
}


@dataclass
class LemmaCandidate:
    """A lemma candidate with metadata for disambiguation."""
    lemma: str
    lang: str = ""       # "el" (SMG), "grc" (AG), "med" (Medieval), "" (unknown)
    proper: bool = False  # True if lemma is a proper noun (capitalized headword)
    source: str = ""      # "lookup", "elision", "crasis", "model", "article"
    score: float = 1.0    # confidence (1.0 for lookup, lower for model)
    via: str = ""         # how the lookup matched: "exact", "lower", "mono",
                          # "stripped", "elision:ε" (which vowel expanded), etc.


def to_monotonic(s: str) -> str:
    """Convert polytonic Greek to monotonic."""
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _POLYTONIC_STRIP:
            continue
        if cp in _POLYTONIC_TO_ACUTE:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def grave_to_acute(s: str) -> str:
    """Convert grave accents to acute, preserving all other diacritics.

    In Greek orthography, grave (βαρεῖα) is a positional variant of acute —
    it appears on the last syllable when followed by another word. So ὣς = ὡς,
    τὸν = τόν, etc. This is a lighter normalization than to_monotonic(), which
    also strips breathings and circumflex.
    """
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        if ord(ch) == 0x0300:  # COMBINING GRAVE ACCENT
            out.append("\u0301")  # COMBINING ACUTE ACCENT
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def strip_accents(s: str) -> str:
    """Strip all accents for fuzzy matching."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


# OCR confusion pairs: (char_a, char_b) -> substitution cost (0.0 to 1.0).
# Normal substitution costs 1.0. OCR-common confusions cost less.
# Built from GCV analysis of LSJ supplement OCR output.
_OCR_CONFUSIONS: dict[tuple[str, str], float] = {}


def _build_ocr_confusions():
    """Build OCR confusion cost matrix (lazy, called once)."""
    if _OCR_CONFUSIONS:
        return

    pairs = [
        # Greek/Latin script mixing (cost ~0, same glyph)
        ("ο", "o", 0.1), ("Ο", "O", 0.1),
        ("ρ", "p", 0.1), ("Ρ", "P", 0.1),
        ("ν", "v", 0.1), ("Ν", "N", 0.1),
        ("τ", "t", 0.1), ("Τ", "T", 0.1),
        ("κ", "k", 0.1), ("Κ", "K", 0.1),
        ("α", "a", 0.1), ("Α", "A", 0.1),
        ("η", "n", 0.2), ("Η", "H", 0.1),
        ("ε", "e", 0.1), ("Ε", "E", 0.1),
        ("ι", "i", 0.1),
        ("υ", "u", 0.1), ("Υ", "Y", 0.1),
        ("χ", "x", 0.1), ("Χ", "X", 0.1),
        ("ω", "w", 0.2),
        ("β", "B", 0.2),
        ("γ", "y", 0.3),  # GCV-specific
        ("δ", "d", 0.3),  # GCV-specific

        # Cyrillic contamination (GCV-specific, same glyph)
        ("ο", "\u043e", 0.0),  # Cyrillic о
        ("α", "\u0430", 0.0),  # Cyrillic а
        ("ε", "\u0435", 0.0),  # Cyrillic е
        ("υ", "\u0443", 0.0),  # Cyrillic у
        ("κ", "\u043a", 0.0),  # Cyrillic к
        ("ρ", "\u0440", 0.0),  # Cyrillic р

        # Common OCR letter confusions (similar shapes)
        ("ο", "σ", 0.5),  # round shapes
        ("θ", "δ", 0.5),  # similar with crossbar
        ("η", "π", 0.5),  # similar verticals
        ("ν", "μ", 0.7),  # similar verticals
        ("ρ", "β", 0.7),  # descender confusion
        ("ι", "ΐ", 0.3),  # diaeresis confusion
        ("υ", "ΰ", 0.3),

        # Number/letter confusions (GCV Roman numeral issue)
        ("1", "I", 0.1),
        ("1", "l", 0.1),

        # GCV descender confusions: J/j for ψ/ὑ
        ("ψ", "J", 0.2), ("ψ", "j", 0.2),
        ("υ", "J", 0.3), ("υ", "j", 0.3),
        ("ὑ", "J", 0.2), ("ὑ", "j", 0.2),
        # GCV: Q/q for θ/σ
        ("θ", "Q", 0.3), ("θ", "q", 0.3),
        ("σ", "Q", 0.3), ("σ", "q", 0.3),
    ]

    for a, b, cost in pairs:
        _OCR_CONFUSIONS[(a, b)] = cost
        _OCR_CONFUSIONS[(b, a)] = cost


def _weighted_levenshtein(a: str, b: str) -> float:
    """Weighted Levenshtein distance using OCR confusion costs.

    Decomposes to NFD first so that combining diacritics are handled
    separately from base characters:
    - Combining diacritics: insert/delete costs 0.1 (nearly free)
    - OCR-confused base character pairs: uses cost matrix (0.1-0.5)
    - Normal substitutions: cost 1.0

    This makes breathing/accent errors nearly free while correctly
    penalizing real letter substitutions.
    """
    _build_ocr_confusions()

    a_nfd = unicodedata.normalize("NFD", a)
    b_nfd = unicodedata.normalize("NFD", b)

    if len(a_nfd) < len(b_nfd):
        return _weighted_levenshtein(b, a)
    if not b_nfd:
        return float(len(a_nfd))

    def _char_cost(c: str) -> float:
        """Cost to insert or delete a character."""
        if unicodedata.combining(c):
            return 0.1  # diacritics are nearly free
        return 1.0

    prev = [0.0]
    for j in range(len(b_nfd)):
        prev.append(prev[-1] + _char_cost(b_nfd[j]))

    for i, ca in enumerate(a_nfd):
        ins_cost_a = _char_cost(ca)
        curr = [prev[0] + ins_cost_a]
        for j, cb in enumerate(b_nfd):
            if ca == cb:
                sub_cost = 0.0
            elif unicodedata.combining(ca) and unicodedata.combining(cb):
                sub_cost = 0.1  # swapping one diacritic for another
            else:
                sub_cost = _OCR_CONFUSIONS.get((ca, cb), 1.0)
            ins_cost_b = _char_cost(cb)
            curr.append(min(
                curr[j] + ins_cost_b,     # insert b[j]
                prev[j + 1] + ins_cost_a,  # delete a[i]
                prev[j] + sub_cost,        # substitute
            ))
        prev = curr
    return prev[-1]


def _strip_elision(word: str) -> str | None:
    """Strip trailing elision mark from an elided word form.

    Returns the consonant stem, or None if no elision detected.
    The elision mark in polytonic text is U+0313 (COMBINING COMMA ABOVE)
    attached to the final consonant, e.g. ἀλλ + U+0313 for ἀλλ̓.

    IMPORTANT: U+0313 also serves as smooth breathing at the START of
    polytonic words (ἐ = ε + U+0313). We only treat it as elision when
    it appears after the first base character cluster (i.e. not on the
    initial letter).
    """
    nfd = unicodedata.normalize("NFD", word)
    if len(nfd) < 2:
        return None

    # Count base (non-combining) characters
    base_count = sum(1 for ch in nfd if unicodedata.category(ch) != "Mn")

    if base_count <= 1:
        # Single base char (like δ̓, τ̓, γ̓): the U+0313 IS the elision
        # mark, not a breathing. Return the bare consonant as stem.
        if nfd[-1] in _ELISION_MARKS:
            stem = unicodedata.normalize("NFC", nfd[:-1])
            if stem:
                return stem
        return None

    # Multi-char word: U+0313 is only elision when it's on the LAST
    # base character. Anywhere else (initial letter, diphthong like ου)
    # it's a smooth breathing mark.
    #
    # Find the last base character, then check if U+0313 follows it
    # with no more base characters after.
    last_base_idx = -1
    for i in range(len(nfd) - 1, -1, -1):
        if unicodedata.category(nfd[i]) != "Mn":
            last_base_idx = i
            break

    if last_base_idx < 0:
        return None

    # Check combining marks after the last base char for elision mark
    for i in range(last_base_idx + 1, len(nfd)):
        if nfd[i] in _ELISION_MARKS:
            stem = unicodedata.normalize("NFC", nfd[:i])
            if stem:
                return stem

    # Also check non-combining elision marks (right quote, modifier apostrophe)
    # at the very end of the string
    if nfd[-1] in _ELISION_MARKS and unicodedata.category(nfd[-1]) != "Mn":
        stem = unicodedata.normalize("NFC", nfd[:-1])
        if stem:
            return stem

    return None


class LookupDB:
    """Dict-like wrapper around SQLite lookup table.

    Provides .get(key) that queries SQLite instead of loading the full
    12.5M-entry dict into memory. Supports lazy bulk-load into a dict
    for batch operations.

    The DB has two tables:
      - lemmas(id, text): deduplicated lemma strings (~700K)
      - lookup(form, lemma_id, lang): form->lemma mappings (~12.5M)
        lang='all' for combined, lang='grc' for AG-only overrides,
        lang='el' for MG-only overrides
    """

    def __init__(self, db_path, lang='all'):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
        self._lang = lang
        self._dict = None  # lazy-loaded for batch ops
        self._query = (
            "SELECT l.text FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
            "WHERE k.form = ? AND k.lang = ? LIMIT 1"
        )

    def get(self, key, default=None):
        """Dict-compatible .get() backed by SQLite."""
        if self._dict is not None:
            return self._dict.get(key, default)
        row = self._conn.execute(self._query, (key, self._lang)).fetchone()
        if row:
            return row[0]
        # AG-only table ('grc') falls through to combined ('all') for entries
        # where AG agrees with combined (not stored separately to save space).
        # Same logic for MG-only ('el').
        if self._lang in ('grc', 'el'):
            row = self._conn.execute(self._query, (key, 'all')).fetchone()
            return row[0] if row else default
        return default

    def __contains__(self, key):
        if self._dict is not None:
            return key in self._dict
        row = self._conn.execute(self._query, (key, self._lang)).fetchone()
        if row:
            return True
        if self._lang in ('grc', 'el'):
            return self._conn.execute(self._query, (key, 'all')).fetchone() is not None
        return False

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __bool__(self):
        return True

    def items(self):
        """Iterate all entries (loads nothing extra into memory)."""
        cursor = self._conn.execute(
            "SELECT k.form, l.text FROM lookup k JOIN lemmas l ON k.lemma_id = l.id "
            "WHERE k.lang = ?", (self._lang,))
        return cursor

    def __iter__(self):
        cursor = self._conn.execute(
            "SELECT form FROM lookup WHERE lang = ?", (self._lang,))
        for row in cursor:
            yield row[0]

    def __len__(self):
        row = self._conn.execute(
            "SELECT COUNT(*) FROM lookup WHERE lang = ?", (self._lang,)).fetchone()
        return row[0]

    def bulk_load(self):
        """Load entire table into a dict for fast batch operations."""
        if self._dict is not None:
            return
        self._dict = {}
        for form, lemma in self.items():
            self._dict[form] = lemma

    def spell_lookup_stripped(self, candidates: set[str],
                             src_filter: str = None
                             ) -> dict[str, list[str]]:
        """Look up stripped forms in the DB, return {stripped: [original_forms]}.

        Uses the indexed 'stripped' column for fast batch lookup.

        Args:
            candidates: Set of accent-stripped forms to look up.
            src_filter: If set, only return forms from this source
                (e.g., 'grc' for AG-sourced forms only).
        """
        if not candidates:
            return {}
        result: dict[str, list[str]] = {}
        candidate_list = list(candidates)
        for i in range(0, len(candidate_list), 900):
            batch = candidate_list[i:i + 900]
            placeholders = ",".join("?" * len(batch))
            if src_filter:
                query = (
                    f"SELECT DISTINCT stripped, form FROM lookup "
                    f"WHERE stripped IN ({placeholders}) AND src = ?"
                )
                batch = batch + [src_filter]
            elif self._lang in ('grc', 'el'):
                query = (
                    f"SELECT DISTINCT stripped, form FROM lookup "
                    f"WHERE stripped IN ({placeholders}) "
                    f"AND lang IN ('{self._lang}', 'all')"
                )
            else:
                query = (
                    f"SELECT DISTINCT stripped, form FROM lookup "
                    f"WHERE stripped IN ({placeholders}) AND lang = ?"
                )
                batch = batch + [self._lang]
            for stripped, form in self._conn.execute(query, batch):
                if stripped not in result:
                    result[stripped] = []
                result[stripped].append(form)
        return result

    def has_stripped(self, stripped: str) -> bool:
        """Check if a stripped form exists in the DB."""
        if self._lang in ('grc', 'el'):
            row = self._conn.execute(
                "SELECT 1 FROM lookup WHERE stripped = ? AND lang IN (?, 'all') LIMIT 1",
                (stripped, self._lang)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT 1 FROM lookup WHERE stripped = ? AND lang = ? LIMIT 1",
                (stripped, self._lang)).fetchone()
        return row is not None

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


class Dilemma:
    def __init__(self, lang="all", device=None, scale=None,
                 resolve_articles=False, normalize=False, period=None,
                 convention=None):
        """Initialize Dilemma.

        Args:
            lang: "all" (default) for MG+AG+Medieval combined,
                  "el" for MG only, "grc" for AG only
            device: "cpu", "cuda", etc. Auto-detected if None.
            scale: Model scale (0-4). None auto-detects the best available.
                   Larger scales = more training data = better generalization
                   on unseen forms. Lookup table is the same for all scales.
            resolve_articles: if True, resolve article forms (τῆς, τόν,
                  etc.) to the canonical lemma ὁ, and pronoun clitics
                  (μοι, σοι) to their pronoun lemma (ἐγώ, σύ). Default
                  False, which keeps articles/pronouns as self-mappings
                  (better for alignment where you want surface-form
                  matching). Set True for evaluation against treebanks
                  like DiGreC/AGDT which use ὁ as the article lemma.
            normalize: if True, enable orthographic normalization for
                  Byzantine/papyrological texts. Generates candidate
                  spellings (fixing itacism, missing iota subscripta,
                  etc.) and checks them against the lookup table.
            period: Historical period for normalization rule weights.
                  One of: "hellenistic", "late_antique", "byzantine",
                  "all" (default). Only used when normalize=True.
            convention: Lemma convention for output remapping. Controls
                  which citation form is returned when multiple conventions
                  exist for the same word (e.g. LSJ vs Wiktionary headwords).
                  One of: None (default, no remapping), "wiktionary" (same
                  as None), "lsj" (remap to LSJ dictionary headwords using
                  lemma equivalence groups).
        """
        if convention not in _VALID_CONVENTIONS:
            raise ValueError(
                f"Unknown convention {convention!r}. "
                f"Valid values: {sorted(c for c in _VALID_CONVENTIONS if c)}, or None."
            )
        if lang == "both":
            lang = "all"
        self.lang = lang
        self._scale = scale
        self._resolve_articles = resolve_articles
        self._model = None
        self._vocab = None
        self._device = device
        self._normalizer = None
        if normalize:
            from normalize import Normalizer
            self._normalizer = Normalizer(period=period)

        # Lookup tables: SQLite-backed (instant startup) or dict (JSON fallback)
        self._mg_lookup = {}
        self._med_lookup = {}
        self._ag_lookup = {}
        self._lookup = {}
        self._using_db = False

        # POS-indexed disambiguation table: {form: {upos: lemma}}
        self._pos_lookup: dict[str, dict[str, str]] = {}
        # AG-only POS lookup for polytonic-first disambiguation
        self._pos_ag_lookup: dict[str, dict[str, str]] = {}

        # Headword frequency table (lazy-loaded from lsj9_frequency.json)
        self._hw_frequency: dict[str, int] | None = None

        self._load_lookups()
        self._convention_map = self._build_convention_map(convention)
        self._check_backend()

    def _check_backend(self):
        """Warn once at init if no model backend is available."""
        model_dir = self._find_model_dir()
        has_onnx_files = (model_dir / "encoder.onnx").exists()
        has_pt_file = (model_dir / "model.pt").exists()
        if has_onnx_files:
            try:
                import onnxruntime  # noqa: F401
                return
            except ImportError:
                pass
        if has_pt_file:
            try:
                import torch  # noqa: F401
                return
            except ImportError:
                pass
        import warnings
        warnings.warn(
            "No model backend available. Install onnxruntime (~50 MB) or "
            "torch (~2 GB) for unseen-word inference. Lookup table still "
            "works, but unknown forms will return unchanged. "
            "pip install onnxruntime",
            stacklevel=3,
        )

    def _load_lookups(self):
        """Load lookup tables from SQLite (instant) or JSON (fallback).

        SQLite path (lookup.db): pre-merged combined table with AG priority,
        plus AG-only overrides for polytonic disambiguation. Near-instant
        startup, 0.05ms/query, supports lazy bulk_load() for batch ops.

        JSON fallback: loads all three JSON files and merges at init (~11s).
        """
        if LOOKUP_DB_PATH.exists():
            # SQLite: instant startup for all language modes
            if self.lang == "grc":
                self._lookup = LookupDB(LOOKUP_DB_PATH, lang='grc')
                self._ag_lookup = self._lookup
            elif self.lang == "el":
                self._lookup = LookupDB(LOOKUP_DB_PATH, lang='all')
                self._ag_lookup = LookupDB(LOOKUP_DB_PATH, lang='grc')
                self._mg_lookup = LookupDB(LOOKUP_DB_PATH, lang='el')
            else:
                self._lookup = LookupDB(LOOKUP_DB_PATH, lang='all')
                self._ag_lookup = LookupDB(LOOKUP_DB_PATH, lang='grc')
            self._using_db = True
        else:
            # JSON fallback
            self._load_lookups_json()

        self._load_pos_lookups()

    def _load_lookups_json(self):
        """Fallback: load JSON lookup tables and merge (~11s)."""
        if LOOKUP_PATH.exists():
            with open(LOOKUP_PATH, encoding="utf-8") as f:
                self._mg_lookup = json.load(f)
        if MED_LOOKUP_PATH.exists():
            with open(MED_LOOKUP_PATH, encoding="utf-8") as f:
                self._med_lookup = json.load(f)
        if AG_LOOKUP_PATH.exists():
            with open(AG_LOOKUP_PATH, encoding="utf-8") as f:
                self._ag_lookup = json.load(f)

        def _is_self_map(form, lemma):
            return (form == lemma
                    or strip_accents(form.lower()) == strip_accents(lemma.lower()))

        if self.lang == "all":
            for data in [self._ag_lookup, self._med_lookup, self._mg_lookup]:
                for k, v in data.items():
                    if k not in self._lookup:
                        self._lookup[k] = v
                    elif _is_self_map(k, self._lookup[k]) and not _is_self_map(k, v):
                        self._lookup[k] = v
                    elif (_is_self_map(k, self._lookup[k])
                          and _is_self_map(k, v) and v == k
                          and self._lookup[k] != k):
                        self._lookup[k] = v
        elif self.lang == "el":
            for data in [self._mg_lookup, self._med_lookup]:
                for k, v in data.items():
                    if k not in self._lookup:
                        self._lookup[k] = v
                    elif _is_self_map(k, self._lookup[k]) and not _is_self_map(k, v):
                        self._lookup[k] = v
        elif self.lang == "grc":
            self._lookup = dict(self._ag_lookup)

    def _load_pos_lookups(self):
        """Load POS disambiguation tables from JSON.

        Builds two tables:
        - _pos_ag_lookup: AG-only sources (treebank, GLAUx, AG Wiktionary, LSJ9)
        - _pos_lookup: combined (AG sources + MG Wiktionary)

        For polytonic input (breathing marks, circumflex), lemmatize_pos() and
        lemmatize_batch_pos() check _pos_ag_lookup first before the combined
        table, mirroring the AG-first logic in the main lookup.

        Priority within each table: treebank (gold) > GLAUx (corpus) >
        MG Wiktionary (combined only) > AG Wiktionary > LSJ9 grammar.
        """
        def _add_to(target, source_data, overwrite=False):
            """Merge source_data into target POS dict."""
            for form, upos_lemmas in source_data.items():
                if form not in target:
                    target[form] = {}
                if overwrite:
                    target[form].update(upos_lemmas)
                else:
                    for upos, lemma in upos_lemmas.items():
                        if upos not in target[form]:
                            target[form][upos] = lemma

        # 1. Treebank POS lookup (gold-annotated, highest priority for AG)
        if self.lang in ("all", "grc") and TREEBANK_POS_LOOKUP_PATH.exists():
            with open(TREEBANK_POS_LOOKUP_PATH, encoding="utf-8") as f:
                tb_pos = json.load(f)
            _add_to(self._pos_ag_lookup, tb_pos, overwrite=True)
            _add_to(self._pos_lookup, tb_pos, overwrite=True)

        # 2. GLAUx POS lookup (corpus-derived, 8.7K entries)
        if self.lang in ("all", "grc") and GLAUX_POS_LOOKUP_PATH.exists():
            with open(GLAUX_POS_LOOKUP_PATH, encoding="utf-8") as f:
                glaux_pos = json.load(f)
            _add_to(self._pos_ag_lookup, glaux_pos)
            _add_to(self._pos_lookup, glaux_pos)

        # 3. MG POS lookup (Wiktionary-derived, combined table only)
        if self.lang in ("all", "el") and MG_POS_LOOKUP_PATH.exists():
            with open(MG_POS_LOOKUP_PATH, encoding="utf-8") as f:
                mg_pos = json.load(f)
            _add_to(self._pos_lookup, mg_pos)  # combined only, not AG

        # 4. AG Wiktionary POS lookup (fills remaining gaps)
        if self.lang in ("all", "grc") and AG_POS_LOOKUP_PATH.exists():
            with open(AG_POS_LOOKUP_PATH, encoding="utf-8") as f:
                ag_pos = json.load(f)
            _add_to(self._pos_ag_lookup, ag_pos)
            _add_to(self._pos_lookup, ag_pos)

        # 5. LSJ9 grammar-derived POS (407K forms with NOUN/ADJ from
        #    the grammar field: ὁ/ἡ/τό -> NOUN, ον/ές -> ADJ)
        if self.lang in ("all", "grc") and LSJ9_POS_LOOKUP_PATH.exists():
            with open(LSJ9_POS_LOOKUP_PATH, encoding="utf-8") as f:
                lsj9_pos = json.load(f)
            _add_to(self._pos_ag_lookup, lsj9_pos)
            _add_to(self._pos_lookup, lsj9_pos)

    def _build_convention_map(self, convention: str | None) -> dict[str, str]:
        """Build a lemma remapping dict for the given convention.

        For "lsj", each equivalence group is resolved to the first member
        that appears in the LSJ headword list. All other members map to it.
        For None or "wiktionary", returns an empty dict (no remapping).

        After auto-deriving from equivalences, explicit overrides from
        data/convention_{name}.json are applied (if the file exists).
        The override file format is: {"mappings": {"from_lemma": "to_lemma"}}.
        """
        if convention is None or convention == "wiktionary":
            return {}

        remap = {}

        # Auto-derive from equivalence groups + LSJ headwords
        if LEMMA_EQUIVALENCES_PATH.exists():
            with open(LEMMA_EQUIVALENCES_PATH, encoding="utf-8") as f:
                equiv_data = json.load(f)

            # Load LSJ headwords for cross-referencing.
            # LSJ headwords often include vowel-length marks (macron U+0304,
            # breve U+0306) like βᾰρύς. Strip these so they match our
            # plain equivalence group members.
            lsj_headwords = set()
            if LSJ_HEADWORDS_PATH.exists():
                with open(LSJ_HEADWORDS_PATH, encoding="utf-8") as f:
                    raw = json.load(f)
                lsj_headwords = set(raw)
                for h in raw:
                    nfd = unicodedata.normalize("NFD", h)
                    stripped = "".join(
                        c for c in nfd if ord(c) not in (0x0304, 0x0306))
                    stripped = unicodedata.normalize("NFC", stripped)
                    if stripped != h:
                        lsj_headwords.add(stripped)

            for group in equiv_data.get("groups", []):
                if len(group) < 2:
                    continue

                # Find the canonical for this convention: first member
                # in the LSJ headword list. Fall back to the first
                # member of the group if none are LSJ headwords.
                canonical = group[0]
                for member in group:
                    if member in lsj_headwords:
                        canonical = member
                        break

                for member in group:
                    if member != canonical:
                        remap[member] = canonical

        # Apply explicit overrides from convention file
        override_path = CONVENTION_DIR / f"convention_{convention}.json"
        if override_path.exists():
            with open(override_path, encoding="utf-8") as f:
                overrides = json.load(f)
            for from_lemma, to_lemma in overrides.get("mappings", {}).items():
                remap[from_lemma] = to_lemma

        return remap

    def _apply_convention(self, lemma: str) -> str:
        """Remap a lemma according to the active convention."""
        if self._convention_map:
            return self._convention_map.get(lemma, lemma)
        return lemma

    def _find_model_dir(self):
        """Find the best available model directory.

        Search order: {lang}/ (full model), then {lang}-s3/-s2/-s1 (legacy),
        then combined/ as fallback if no language-specific model exists.
        """
        lang_dir = {"el": "el", "grc": "grc", "all": "combined"}[self.lang]

        # Explicit scale requested
        if self._scale is not None:
            for prefix in [lang_dir, "combined"]:
                # New naming: {lang}-test or {lang} (full)
                if str(self._scale) == "test":
                    candidate = MODEL_DIR / f"{prefix}-test"
                else:
                    candidate = MODEL_DIR / prefix
                if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                    return candidate
                # Legacy naming: -s1, -s2, -s3
                candidate = MODEL_DIR / f"{prefix}-s{self._scale}"
                if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                    return candidate

        # Auto-detect: prefer {lang}/ (full), then legacy -s3/-s2/-s1
        for prefix in [lang_dir, "combined"]:
            candidate = MODEL_DIR / prefix
            if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                return candidate
            for s in [3, 2, 1]:
                candidate = MODEL_DIR / f"{prefix}-s{s}"
                if (candidate / "encoder.onnx").exists() or (candidate / "model.pt").exists():
                    return candidate

        return MODEL_DIR / lang_dir

    def _load_model(self):
        """Lazy-load the model on first use. Prefers ONNX, falls back to PyTorch."""
        if self._model is not None:
            return

        model_path = self._find_model_dir()

        # Try ONNX first (no PyTorch dependency, ~50MB vs ~2GB)
        if (model_path / "encoder.onnx").exists():
            self._load_onnx(model_path)
            return

        # Fall back to PyTorch
        self._load_pytorch(model_path)

    def _load_onnx(self, model_path):
        """Load ONNX model and lightweight vocab."""
        from onnx_inference import OnnxLemmaModel, CharVocabLight
        vocab_path = model_path / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"No vocab.json at {vocab_path}. "
                f"Run: python export_onnx.py"
            )
        self._vocab = CharVocabLight(vocab_path)
        self._model = OnnxLemmaModel(model_path)
        self._device = "cpu"
        self._use_onnx = True

    def _load_pytorch(self, model_path):
        """Load PyTorch model (original path).

        Detects and loads morphology heads (POS, nominal, verbal) when
        present in the checkpoint. Head label mappings are stored in
        self._head_labels for inference use.
        """
        import torch
        from model import CharVocab, LemmaTransformer

        pt_path = model_path / "model.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"No trained model at {pt_path}. "
                f"Run: python train.py --lang {self.lang}"
            )

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
        self._vocab = CharVocab()
        self._vocab.load_state_dict(checkpoint["vocab"])
        cfg = checkpoint["config"]

        # Detect morphology heads from state dict keys
        state = checkpoint["model_state_dict"]
        head_cfg = checkpoint.get("head_config", {})

        # Infer head dimensions from weights if head_config is missing
        num_pos = cfg.get("num_pos_tags", 0)
        if not num_pos and "pos_head.weight" in state:
            num_pos = state["pos_head.weight"].shape[0]
            cfg["num_pos_tags"] = num_pos

        self._model = LemmaTransformer(**cfg)

        # Create nom/verb heads if weights exist
        d_model = cfg.get("d_model", 256)
        if "nom_head.weight" in state:
            num_nom = state["nom_head.weight"].shape[0]
            import torch.nn as nn
            self._model.nom_head = nn.Linear(d_model, num_nom)
        if "verb_head.weight" in state:
            num_verb = state["verb_head.weight"].shape[0]
            import torch.nn as nn
            self._model.verb_head = nn.Linear(d_model, num_verb)

        self._model.load_state_dict(state)
        self._model.to(device)
        self._model.eval()
        self._use_onnx = False

        # Store label mappings for inference
        self._head_labels = {
            "pos": {int(k): v for k, v in head_cfg.get("pos_labels", {}).items()},
            "nom": {int(k): v for k, v in head_cfg.get("nom_labels", {}).items()},
            "verb": {int(k): v for k, v in head_cfg.get("verb_labels", {}).items()},
        }
        # Build fallback POS label map if not saved
        if num_pos and not self._head_labels["pos"]:
            _POS_FALLBACK = {
                0: "verb", 1: "noun", 2: "adj", 3: "adv", 4: "name",
                5: "pron", 6: "num", 7: "prep", 8: "article", 9: "character",
            }
            self._head_labels["pos"] = {
                i: _POS_FALLBACK.get(i, f"tag{i}") for i in range(num_pos)
            }

    def _resolve_closed_class(self, word: str) -> str | None:
        """Resolve articles/pronouns to canonical lemma if enabled."""
        if not self._resolve_articles:
            return None
        if (word in _ARTICLE_FORMS
                or to_monotonic(word) in _ARTICLE_FORMS):
            # Don't use strip_accents here - it's too aggressive for short
            # words (e.g. ἤ "or" becomes η which matches the article)
            return _ARTICLE_LEMMA
        if word in _PRONOUN_LEMMAS:
            return _PRONOUN_LEMMAS[word]
        mono = to_monotonic(word)
        if mono in _PRONOUN_LEMMAS:
            return _PRONOUN_LEMMAS[mono]
        return None

    def _lookup_word(self, word: str) -> str | None:
        """Try lookup cascade: exact -> lowercase -> grave_to_acute -> monotonic -> stripped.

        Grave-to-acute is tried before monotonic because it preserves
        breathings and circumflex (lighter normalization). ὣς → ὡς works
        here without losing the breathing that monotonic would strip.

        For polytonic input (breathings/circumflex present), AG lookup is
        tried first to avoid MG lemma forms (biblion vs biblion).

        For lang="el" (MG mode), MG-specific entries are checked first
        (monotonic input -> el lookup), then falls back to combined.

        Skips mono/stripped matches that are trivially short (1-2 chars)
        and map to themselves - these are usually false positives from
        accent stripping on elided or particle forms.
        """
        # For MG mode, try MG-specific lookup first. MG entries take
        # priority over AG when there's a conflict (e.g. MG lemma
        # forms preferred over AG lemma forms for the same surface form).
        if self.lang == "el" and self._mg_lookup:
            tbl = self._mg_lookup
            hit = tbl.get(word) or tbl.get(word.lower())
            if not hit:
                mono = to_monotonic(word.lower())
                stripped = strip_accents(word.lower())
                for variant in [mono, stripped]:
                    hit = tbl.get(variant)
                    if hit and not (len(variant) <= 2 and hit == variant):
                        break
                    hit = None
            if hit:
                return hit

        # For polytonic input (breathings/circumflex), try AG-only lookup
        # first. Even though AG has priority in the combined table, the
        # normalization cascade (mono/stripped) can still land on MG entries.
        nfd = unicodedata.normalize("NFD", word)
        has_poly = any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                       for ch in nfd)
        if has_poly and self.lang == "all" and self._ag_lookup:
            tbl = self._ag_lookup
            hit = tbl.get(word) or tbl.get(word.lower())
            if not hit:
                acute = grave_to_acute(word)
                if acute != word:
                    hit = tbl.get(acute) or tbl.get(acute.lower())
            if not hit:
                for variant in [to_monotonic(word.lower()),
                                strip_accents(word.lower())]:
                    hit = tbl.get(variant)
                    if hit and not (len(variant) <= 2 and hit == variant):
                        break
                    hit = None
            if hit:
                return hit

        lemma = self._lookup.get(word) or self._lookup.get(word.lower())
        if lemma:
            return lemma
        # Grave → acute (lightest normalization, preserves breathings)
        acute = grave_to_acute(word)
        if acute != word:
            lemma = self._lookup.get(acute) or self._lookup.get(acute.lower())
            if lemma:
                return lemma
        mono = to_monotonic(word.lower())
        stripped = strip_accents(word.lower())
        for variant in [mono, stripped]:
            hit = self._lookup.get(variant)
            if hit and not (len(variant) <= 2 and hit == variant):
                return hit
        return None


    def _expand_elision(self, word: str) -> str | None:
        """Try to resolve an elided form by expanding with vowels.

        Strips the elision mark, appends each Greek vowel, and checks
        if the expanded form is in the lookup table. Prefers expansions
        where the lemma is a real word (differs from the expanded form)
        over self-mapping headwords.
        """
        candidates = self._expand_elision_all(word)
        if not candidates:
            return None
        # _expand_elision_all returns results pre-sorted by vowel frequency
        return candidates[0][1]

    def _expand_elision_all(self, word: str) -> list[tuple[str, str, str]]:
        """Return ALL valid elision expansions as (expanded, lemma, vowel) triples.

        For polytonic input (has breathings/circumflex), prioritizes AG lookup.
        Elision is overwhelmingly an AG phenomenon; MG monotonic forms would
        pollute results with false matches.

        Results are sorted by vowel frequency in elision contexts (ε, α, ο
        most common), then by lemma length. This ensures both lemmatize()
        and lemmatize_verbose() get the same ranking.
        """
        stem = _strip_elision(word)
        if not stem:
            return []

        # Detect if input is polytonic (has AG-style diacritics).
        # Also treat any non-combining elision mark (U+1FBD KORONIS,
        # U+02BC, U+2019) as indicating AG context, since elision is
        # overwhelmingly an AG phenomenon.
        nfd = unicodedata.normalize("NFD", word)
        has_polytonic = (any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                            for ch in nfd)
                         or any(ch in _ELISION_MARKS
                                and unicodedata.category(ch) != "Mn"
                                for ch in word))

        # Choose which lookups to search
        if has_polytonic:
            tables = [(self._ag_lookup, "grc")]
        else:
            tables = [(self._mg_lookup, "el"),
                      (self._med_lookup, "med"),
                      (self._ag_lookup, "grc")]

        results = []
        seen_lemmas = set()

        all_vowels = list(_GREEK_VOWELS)
        # Also try accented vowels
        accented = {"α": "ά", "ε": "έ", "ι": "ί", "ο": "ό",
                    "η": "ή", "υ": "ύ", "ω": "ώ"}
        all_candidates = [(v, v) for v in all_vowels]
        all_candidates += [(acc, v) for v, acc in accented.items()]

        for table, lang in tables:
            for suffix, vowel_name in all_candidates:
                expanded = stem + suffix
                # Try full normalization cascade
                lemma = None
                for variant in (expanded, expanded.lower(),
                                grave_to_acute(expanded),
                                to_monotonic(expanded.lower()),
                                strip_accents(expanded.lower())):
                    lemma = table.get(variant)
                    if lemma:
                        break
                if lemma and lemma not in seen_lemmas:
                    seen_lemmas.add(lemma)
                    results.append((expanded, lemma, suffix))

        # Common elided function words - these should always win over
        # content words when ambiguous. Maps stem -> expected lemma.
        _ELISION_PRIORITY = {
            "αλλ", "μετ", "παρ", "κατ", "δι", "απ", "επ",
            "υπ", "αφ", "εφ", "υφ", "μηδ", "ουδ", "αντ", "περ",
        }
        stem_lower = strip_accents(stem.lower())

        # Sort by: (1) function word priority, (2) vowel frequency, (3) lemma length
        _VOWEL_RANK = {v: i for i, v in enumerate("εαοιηυω")}
        _ACC_VOWEL_RANK = {"ά": 1, "έ": 0, "ί": 4, "ό": 2,
                           "ή": 3, "ύ": 5, "ώ": 6}

        # Function word lemmas (prepositions, particles, conjunctions)
        _FUNCTION_LEMMAS = {
            "ἀλλά", "μετά", "παρά", "κατά", "διά", "ἀπό", "ἐπί",
            "ὑπό", "ἀπό", "ἐπί", "ὑπό", "μηδέ", "οὐδέ", "ἀντί",
            "περί", "δέ", "γε", "τε", "ἄρα", "ἔτι",
        }

        def _rank(item):
            expanded, lemma, vowel = item
            # Prefer function words when stem is a known elision stem
            is_function = 0 if (stem_lower in _ELISION_PRIORITY
                                and lemma in _FUNCTION_LEMMAS) else 1
            # Deprioritize proper nouns
            is_proper = 1 if lemma and lemma[0].isupper() else 0
            vrank = _ACC_VOWEL_RANK.get(vowel, _VOWEL_RANK.get(vowel, 10))
            return (is_function, is_proper, vrank, len(lemma))

        results.sort(key=_rank)
        return results

    def _lang_of(self, form: str) -> str:
        """Determine which language table a form comes from."""
        if self._using_db:
            return self._lang_of_db(form)
        lower = form.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        for variant in [form, lower, mono, stripped]:
            if variant in self._mg_lookup:
                return "el"
            if variant in self._med_lookup:
                return "med"
            if variant in self._ag_lookup:
                return "grc"
        return ""

    def _lang_of_db(self, form: str) -> str:
        """SQLite-backed language detection via the src column."""
        _SRC_TO_LANG = {'grc': 'grc', 'el': 'el', 'med': 'med'}
        conn = self._lookup._conn
        lower = form.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        query = "SELECT src FROM lookup WHERE form = ? AND lang = 'all' LIMIT 1"
        for variant in [form, lower, mono, stripped]:
            row = conn.execute(query, (variant,)).fetchone()
            if row:
                return _SRC_TO_LANG.get(row[0], "")
        return ""

    def _is_proper(self, lemma: str) -> bool:
        """Check if a lemma is a proper noun (capitalized headword)."""
        return bool(lemma) and lemma[0].isupper()

    # ---- Morphology head inference ----

    # POS tag to UPOS mapping for POS-lookup integration
    _POS_TO_UPOS = {
        "verb": "VERB", "noun": "NOUN", "adj": "ADJ", "adv": "ADV",
        "name": "PROPN", "pron": "PRON", "num": "NUM", "prep": "ADP",
        "article": "DET", "character": "PROPN",
    }

    def predict_pos_tag(self, word: str) -> str:
        """Predict POS tag for a Greek word using the model's POS head.

        Returns a Wiktionary-style POS label ("verb", "noun", "adj", etc.)
        or "" if no POS head is available.

        Requires the model to be loaded (lazy-loads on first call).
        """
        self._load_model()
        if hasattr(self._model, 'has_pos_head') and self._model.has_pos_head:
            # ONNX path
            src, mask = self._encode_word(word)
            tags = self._model.predict_pos(src, mask)
            return tags[0]
        return ""

    def predict_pos_batch(self, words: list[str]) -> list[str]:
        """Predict POS tags for a batch of Greek words.

        Returns list of Wiktionary-style POS labels.
        """
        self._load_model()
        if hasattr(self._model, 'has_pos_head') and self._model.has_pos_head:
            src, mask = self._encode_words(words)
            return self._model.predict_pos(src, mask)
        return [""] * len(words)

    def _encode_word(self, word: str):
        """Encode a single word for ONNX inference. Returns (src, mask) arrays."""
        import numpy as np
        ids = self._vocab.encode(word)
        max_len = 48  # ONNX_MAX_LEN
        ids = ids + [0] * (max_len - len(ids))
        src = np.array([ids[:max_len]], dtype=np.int64)
        mask = (src == 0)
        return src, mask

    def _encode_words(self, words: list[str]):
        """Encode a batch of words for ONNX inference."""
        import numpy as np
        max_len = 48
        batch = []
        for w in words:
            ids = self._vocab.encode(w)
            ids = ids + [0] * (max_len - len(ids))
            batch.append(ids[:max_len])
        src = np.array(batch, dtype=np.int64)
        mask = (src == 0)
        return src, mask

    # ---- Compound decomposition ----

    # Linking vowels at the junction of Greek compounds
    _COMPOUND_LINK_VOWELS = set("οιυ")
    _MIN_COMPOUND_PREFIX = 2   # e.g. εὐ-, τρι-
    _MIN_COMPOUND_BASE = 3     # need enough for inflection

    def _decompose_compound(self, word: str) -> str | None:
        """Try to lemmatize an unknown compound by splitting at linking vowels.

        Greek compounds: first-stem + linking-vowel (ο/ι/υ) + second-element.
        The second element inflects like its standalone form. Strategy: split
        at each linking vowel (left to right, preferring longer bases), look
        up the base, and reconstruct prefix + base_lemma.

        Returns the reconstructed compound lemma, or None.
        """
        lower = word.lower()
        stripped = strip_accents(lower)

        if len(stripped) < self._MIN_COMPOUND_PREFIX + self._MIN_COMPOUND_BASE + 1:
            return None

        # Try split points left to right (longest base first = most reliable)
        for i in range(self._MIN_COMPOUND_PREFIX - 1,
                       len(stripped) - self._MIN_COMPOUND_BASE):
            if stripped[i] not in self._COMPOUND_LINK_VOWELS:
                continue

            prefix = stripped[:i + 1]   # includes linking vowel
            base = stripped[i + 1:]

            if len(base) < self._MIN_COMPOUND_BASE:
                continue

            # Look up the base in the lookup table
            base_lemma = self._lookup_word(base)
            if not base_lemma:
                continue

            base_lemma_s = strip_accents(base_lemma.lower())

            # Guard: skip if lookup returned identity (no real lemmatization)
            if base_lemma_s == base:
                continue

            # Guard: skip if base_lemma is suspiciously short (false match)
            if len(base_lemma_s) < 2:
                continue

            # Guard: base_lemma should be shorter or equal to base
            # (lemmatization removes inflection, doesn't add length)
            if len(base_lemma_s) > len(base) + 2:
                continue

            # Reconstruct compound lemma
            return prefix + base_lemma_s

        return None

    def _decompose_compound_all(self, word: str) -> list[tuple[str, str, str]]:
        """Return ALL valid compound decompositions.

        Returns list of (compound_lemma, base_lemma, prefix) triples,
        ordered by base length descending (longest base = most specific).
        Used by lemmatize_verbose for multi-candidate output.
        """
        lower = word.lower()
        stripped = strip_accents(lower)
        results = []
        seen = set()

        if len(stripped) < self._MIN_COMPOUND_PREFIX + self._MIN_COMPOUND_BASE + 1:
            return results

        for i in range(self._MIN_COMPOUND_PREFIX - 1,
                       len(stripped) - self._MIN_COMPOUND_BASE):
            if stripped[i] not in self._COMPOUND_LINK_VOWELS:
                continue

            prefix = stripped[:i + 1]
            base = stripped[i + 1:]

            if len(base) < self._MIN_COMPOUND_BASE:
                continue

            base_lemma = self._lookup_word(base)
            if not base_lemma:
                continue

            base_lemma_s = strip_accents(base_lemma.lower())
            if base_lemma_s == base or len(base_lemma_s) < 2:
                continue
            if len(base_lemma_s) > len(base) + 2:
                continue

            compound = prefix + base_lemma_s
            if compound not in seen:
                seen.add(compound)
                results.append((compound, base_lemma, prefix))

        return results

    def lemmatize(self, word: str) -> str:
        """Lemmatize a single Greek word.

        Resolution order:
          1. Article/pronoun resolution (if resolve_articles=True)
          2. Crasis table (small, hand-curated)
          3. Lookup table (instant, 5M+ forms)
          4. Elision expansion (strip mark, try vowels against lookup)
          5. Normalizer (orthographic variants)
          6. Compound decomposition (split at linking vowel, look up base)
          7. Model with beam search + headword filter

        If a convention is set, the output lemma is remapped accordingly.
        """
        # Resolve articles/pronouns to canonical lemma
        closed = self._resolve_closed_class(word)
        if closed is not None:
            return self._apply_convention(closed)

        # Check crasis first (before lookup, since crasis forms are
        # Wiktionary headwords that self-map in the lookup)
        from crasis import resolve_crasis
        crasis_result = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
        if crasis_result is not None:
            return self._apply_convention(crasis_result)

        # Lookup: exact -> lowercase -> monotonic -> accent-stripped
        lemma = self._lookup_word(word)
        if lemma:
            return self._apply_convention(lemma)

        # Elision expansion (after lookup, so known words like εἰ/οὐ
        # aren't falsely caught by smooth-breathing-as-elision)
        elision_lemma = self._expand_elision(word)
        if elision_lemma:
            return self._apply_convention(elision_lemma)

        # Normalizer: try orthographic variants against lookup
        if self._normalizer:
            for candidate in self._normalizer.normalize(word):
                lemma = self._lookup_word(candidate)
                if lemma:
                    return self._apply_convention(lemma)

        # Fall back to model
        self._load_model()
        pred = self._predict([word])[0]

        # Compound decomposition: only when the model returns identity
        # (model couldn't lemmatize). Uses accent-stripped comparison since
        # the model may return slight accent variants of the input.
        if strip_accents(pred.lower()) == strip_accents(word.lower()):
            compound = self._decompose_compound(word)
            if compound:
                return self._apply_convention(compound)

        return self._apply_convention(pred)

    def _pos_table_lookup(self, word: str, upos: str) -> str | None:
        """Look up POS-specific lemma from POS disambiguation tables.

        For polytonic input, checks the AG-only POS table first to avoid
        MG lemma overrides on Ancient Greek text. Returns None if no match.
        """
        lower = word.lower()
        acute = grave_to_acute(lower)
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        variants = (word, lower, acute, mono, stripped)

        # For polytonic input, try AG-only POS first
        nfd = unicodedata.normalize("NFD", word)
        has_poly = any(ord(ch) in _POLYTONIC_STRIP | _POLYTONIC_TO_ACUTE
                       for ch in nfd)
        if has_poly and self.lang == "all" and self._pos_ag_lookup:
            for variant in variants:
                pos_entry = self._pos_ag_lookup.get(variant)
                if pos_entry and upos in pos_entry:
                    return pos_entry[upos]

        # Combined POS lookup
        for variant in variants:
            pos_entry = self._pos_lookup.get(variant)
            if pos_entry and upos in pos_entry:
                return pos_entry[upos]

        return None

    def lemmatize_pos(self, word: str, upos: str) -> str:
        """Lemmatize with POS-aware disambiguation.

        POS is used to disambiguate among multiple candidates from the
        regular lookup, not to override it. The regular lookup (without
        POS) already produces good results; POS should only help pick
        between ambiguous candidates, never make things worse.

        Algorithm:
          1. Run regular lemmatize_verbose() to get all candidates.
          2. If there is only one candidate, return it (POS can't help).
          3. If there are multiple candidates, check POS tables for a
             POS-specific lemma and see if any candidate matches it.
          4. If a match is found, return it. Otherwise return the top
             candidate (same as regular lookup).

        Args:
            word: Greek word form.
            upos: Universal POS tag (NOUN, VERB, ADJ, etc.).

        Returns:
            The lemma string.
        """
        # Get all candidates from regular lookup
        candidates = self.lemmatize_verbose(word)

        if not candidates:
            # Should not happen (verbose always adds identity), but be safe
            return self.lemmatize(word)

        if len(candidates) == 1:
            # Only one candidate - POS can't help, return it directly
            return candidates[0].lemma

        # Multiple candidates - use POS to disambiguate
        pos_lemma = self._pos_table_lookup(word, upos)
        if pos_lemma is not None:
            pos_lemma_conv = self._apply_convention(pos_lemma)
            # Check if any candidate matches the POS-specific lemma
            for c in candidates:
                if c.lemma == pos_lemma_conv:
                    return c.lemma
            # Also check with accent-stripped comparison (POS tables and
            # lookup tables may use slightly different accent conventions)
            pos_stripped = strip_accents(pos_lemma_conv.lower())
            for c in candidates:
                if strip_accents(c.lemma.lower()) == pos_stripped:
                    return c.lemma

        # No POS match among candidates - return top candidate
        # (same result as regular lemmatize)
        return candidates[0].lemma

    def lemmatize_batch_pos(self, words: list[str], upos_tags: list[str]) -> list[str]:
        """Lemmatize a batch of words with POS-aware disambiguation.

        POS is used to disambiguate among multiple candidates, not to
        override the regular lookup. This preserves the batch model
        optimization from lemmatize_batch() while using POS only when
        it can help (multiple valid candidates for a form).

        Algorithm:
          1. Run lemmatize_batch() to get baseline results (efficient,
             batches model inference for unknown words).
          2. For each word where POS tables suggest a different lemma,
             call lemmatize_verbose() to get all candidates and check
             if the POS-specific lemma is among them.
          3. If the POS lemma matches a candidate, use it. Otherwise
             keep the baseline result.

        Args:
            words: List of Greek word forms.
            upos_tags: List of UPOS tags, one per word.

        Returns:
            List of lemma strings.
        """
        assert len(words) == len(upos_tags), (
            f"words and upos_tags must have same length: {len(words)} vs {len(upos_tags)}"
        )

        # Step 1: Get baseline results from regular batch lemmatization
        results = self.lemmatize_batch(words)

        # Step 2: For each word, check if POS could improve the result
        for i, (word, upos) in enumerate(zip(words, upos_tags)):
            pos_lemma = self._pos_table_lookup(word, upos)
            if pos_lemma is None:
                # No POS data for this form/UPOS - keep baseline
                continue

            pos_lemma_conv = self._apply_convention(pos_lemma)
            if pos_lemma_conv == results[i]:
                # POS agrees with baseline - no change needed
                continue

            # POS suggests a different lemma than baseline. Check if
            # the POS lemma is among the valid candidates.
            candidates = self.lemmatize_verbose(word)
            if len(candidates) <= 1:
                # Only one candidate - POS can't help, keep baseline
                continue

            # Check if any candidate matches the POS-specific lemma
            pos_stripped = strip_accents(pos_lemma_conv.lower())
            matched = False
            for c in candidates:
                if c.lemma == pos_lemma_conv:
                    results[i] = c.lemma
                    matched = True
                    break
            if not matched:
                # Try accent-stripped comparison
                for c in candidates:
                    if strip_accents(c.lemma.lower()) == pos_stripped:
                        results[i] = c.lemma
                        matched = True
                        break
            # If no match among candidates, keep the baseline result

        return results

    def lemmatize_verbose(self, word: str) -> list[LemmaCandidate]:
        """Return all candidate lemmas with metadata.

        Unlike lemmatize(), this returns multiple candidates for
        ambiguous forms, tagged with language, proper noun status,
        and source. Useful for downstream tools that can use context
        to disambiguate.

        Examples:
            lemmatize_verbose("ἔριδι")
            -> [LemmaCandidate(lemma="Ἔρις", lang="grc", proper=True, ...),
                LemmaCandidate(lemma="ἔρις", lang="grc", proper=False, ...)]

            lemmatize_verbose("πόλεμο")
            -> [LemmaCandidate(lemma="πόλεμος", lang="el", ...),
                LemmaCandidate(lemma="πόλεμος", lang="grc", ...)]

            lemmatize_verbose("ἀλλ̓")
            -> [LemmaCandidate(lemma="ἀλλά", lang="grc", source="elision", via="elision:ά")]
        """
        candidates = []
        seen = set()  # track (lemma_lower, lang) to avoid exact dupes

        def _add(lemma, lang="", source="", via="", score=1.0):
            key = (lemma, lang)
            if key not in seen:
                seen.add(key)
                candidates.append(LemmaCandidate(
                    lemma=lemma,
                    lang=lang or self._lang_of(lemma),
                    proper=self._is_proper(lemma),
                    source=source,
                    score=score,
                    via=via,
                ))

        # 1. Article/pronoun
        if self._resolve_articles:
            closed = self._resolve_closed_class(word)
            if closed is not None:
                _add(closed, source="article")
                return candidates

        # 2. Crasis
        from crasis import resolve_crasis
        cr = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
        if cr:
            _add(cr, source="crasis")
            return candidates

        # 3. Elision expansion — collect ALL valid expansions (before
        #    lookup, since elided forms false-match letter headwords)
        elision_results = self._expand_elision_all(word)
        for expanded, lemma, vowel in elision_results:
            lang = self._lang_of(expanded) or self._lang_of(lemma)
            _add(lemma, lang=lang, source="elision", via=f"elision:{vowel}")

        # 4. Lookup — collect from ALL language tables
        lower = word.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        variants = [
            (word, "exact"), (lower, "lower"),
            (mono, "mono"), (stripped, "stripped"),
        ]

        for table, lang in [(self._mg_lookup, "el"),
                            (self._med_lookup, "med"),
                            (self._ag_lookup, "grc")]:
            for variant, via in variants:
                lemma = table.get(variant)
                if lemma:
                    # Skip trivial short self-mappings (accent artifacts)
                    if len(variant) <= 2 and lemma == variant and via in ("mono", "stripped"):
                        continue
                    _add(lemma, lang=lang, source="lookup", via=via)
                    # Also check if the OTHER case variant is a headword
                    # (Ἔρις the goddess vs ἔρις strife)
                    if lemma[0].isupper():
                        alt = lemma[0].lower() + lemma[1:]
                    else:
                        alt = lemma[0].upper() + lemma[1:]
                    # The alt must be a self-mapping headword (not just
                    # a form that maps elsewhere)
                    alt_lemma = table.get(alt)
                    if alt_lemma == alt:
                        _add(alt, lang=lang, source="lookup",
                             via=via + "+case_alt")
                    break  # first matching variant wins per language

        # 5. Normalizer: try orthographic variants against all tables
        if not candidates and self._normalizer:
            for norm_candidate in self._normalizer.normalize(word):
                norm_lower = norm_candidate.lower()
                norm_mono = to_monotonic(norm_lower)
                norm_stripped = strip_accents(norm_lower)
                norm_variants = [
                    (norm_candidate, "normalize"),
                    (norm_lower, "normalize+lower"),
                    (norm_mono, "normalize+mono"),
                    (norm_stripped, "normalize+stripped"),
                ]
                for table, lang in [(self._mg_lookup, "el"),
                                    (self._med_lookup, "med"),
                                    (self._ag_lookup, "grc")]:
                    for variant, via in norm_variants:
                        lemma = table.get(variant)
                        if lemma:
                            if len(variant) <= 2 and lemma == variant and via.endswith(("mono", "stripped")):
                                continue
                            _add(lemma, lang=lang, source="normalize", via=via)
                            break

        # 6. Model fallback (if no candidates yet)
        model_identity = False
        if not candidates:
            try:
                self._load_model()
                pred = self._predict([word])[0]
                if strip_accents(pred.lower()) != strip_accents(word.lower()):
                    _add(pred, source="model", score=0.5)
                else:
                    model_identity = True
            except (FileNotFoundError, RuntimeError):
                model_identity = True

        # 7. Compound decomposition (only when model returned identity)
        if model_identity:
            for compound, base_lemma, prefix in self._decompose_compound_all(word):
                _add(compound, source="compound",
                     via=f"{prefix}+{base_lemma}", score=0.7)

        # If still nothing, return the word itself
        if not candidates:
            _add(word, source="identity", score=0.0)

        # Sort: non-proper before proper, then by score descending
        candidates.sort(key=lambda c: (c.proper, -c.score))

        # Apply convention remapping
        if self._convention_map:
            seen_remapped = set()
            remapped = []
            for c in candidates:
                c.lemma = self._apply_convention(c.lemma)
                key = (c.lemma, c.lang)
                if key not in seen_remapped:
                    seen_remapped.add(key)
                    remapped.append(c)
            candidates = remapped

        return candidates

    def lemmatize_batch(self, words: list[str]) -> list[str]:
        """Lemmatize a batch of words. Uses model only for unknowns.

        If a convention is set, all output lemmas are remapped accordingly.
        """
        results = []
        model_indices = []
        model_words = []

        for i, word in enumerate(words):
            # Article/pronoun resolution
            closed = self._resolve_closed_class(word)
            if closed is not None:
                results.append(closed)
                continue

            # Crasis
            from crasis import resolve_crasis
            cr = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
            if cr:
                results.append(cr)
                continue

            # Elision expansion (before lookup — elided forms can
            # false-match letter headwords)
            elision_lemma = self._expand_elision(word)
            if elision_lemma:
                results.append(elision_lemma)
                continue

            # Lookup
            lemma = self._lookup_word(word)
            if lemma:
                results.append(lemma)
                continue

            # Normalizer: try orthographic variants against lookup
            if self._normalizer:
                norm_hit = None
                for candidate in self._normalizer.normalize(word):
                    norm_hit = self._lookup_word(candidate)
                    if norm_hit:
                        break
                if norm_hit:
                    results.append(norm_hit)
                    continue

            results.append(None)
            model_indices.append(i)
            model_words.append(word)

        if model_words:
            self._load_model()
            predictions = self._predict(model_words)
            for idx, word, pred in zip(model_indices, model_words, predictions):
                # Compound decomposition: only when model returns identity
                if strip_accents(pred.lower()) == strip_accents(word.lower()):
                    compound = self._decompose_compound(word)
                    if compound:
                        pred = compound
                results[idx] = pred

        # Apply convention remapping to all results
        if self._convention_map:
            results = [self._apply_convention(r) if r else r for r in results]

        return results

    # ---- Spelling correction ----

    # Greek lowercase letters for ED1 candidate generation
    _GREEK_LETTERS = "αβγδεζηθικλμνξοπρσςτυφχψω"

    def _build_spell_index(self):
        """Build the accent-stripped index for spelling correction.

        With SQLite backend: no-op (queries use the indexed 'stripped' column).
        With JSON fallback: builds in-memory norm map from dict.
        """
        if hasattr(self, "_spell_norm_map"):
            return
        if self._using_db:
            # SQLite path: no in-memory index needed
            self._spell_norm_map = None
            self._spell_norm_set = None
            return
        norm_map: dict[str, set[str]] = {}
        for form in self._lookup:
            stripped = strip_accents(form.lower())
            if stripped not in norm_map:
                norm_map[stripped] = set()
            norm_map[stripped].add(form)
        self._spell_norm_map = norm_map
        self._spell_norm_set = set(norm_map.keys())

    @staticmethod
    def _edits1(word: str) -> set[str]:
        """Generate all strings within edit distance 1 of word.

        Operations: deletes, transposes, replaces, inserts.
        Uses Greek lowercase alphabet for replacements and insertions.
        """
        letters = Dilemma._GREEK_LETTERS
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def suggest_spelling(self, word: str, max_distance: int = 2,
                         ocr_mode: bool = False
                         ) -> list[tuple[str, int | float]]:
        """Suggest spelling corrections for an unknown Greek word.

        Returns a list of (correct_form, edit_distance) tuples, sorted
        by edit distance then alphabetically. Uses a two-layer approach:

        1. Strip diacritics from the input and the dictionary, reducing
           8-11M entries to ~1-3M unique base forms
        2. Find ED0/ED1/ED2 matches on the stripped forms
        3. Return the original polytonic forms, ranked by actual
           Levenshtein distance to the input

        This means diacritic errors (wrong accent, missing breathing)
        cost 0 in the first layer and are corrected for free, while
        letter-level errors (θ/δ, ρ/ν) use standard edit distance.

        Args:
            word: The possibly-misspelled Greek word.
            max_distance: Maximum edit distance (1 or 2). Default 2.
            ocr_mode: If True, use weighted Levenshtein distance that
                gives lower cost to OCR-common confusions (Greek/Latin
                script mixing, Cyrillic contamination, θ/δ, ο/σ).
                This produces better rankings for OCR post-correction.

        Returns:
            List of (corrected_form, distance) tuples. Empty if no
            suggestions found within max_distance.
        """
        self._build_spell_index()
        query_stripped = strip_accents(word.lower())

        if self._using_db:
            return self._suggest_spelling_db(word, query_stripped,
                                             max_distance, ocr_mode)
        return self._suggest_spelling_mem(word, query_stripped,
                                          max_distance, ocr_mode)

    def _suggest_spelling_db(self, word: str, query_stripped: str,
                             max_distance: int, ocr_mode: bool = False
                             ) -> list[tuple[str, int | float]]:
        """SQLite-backed spelling suggestion. No in-memory index needed.

        Generates ED1/ED2 candidate strings, then batch-queries the
        indexed 'stripped' column. ~1000 candidates for ED1, checked
        in one SQL query.
        """
        # For AG mode, filter to AG-sourced forms only (src='grc')
        # to avoid suggesting monotonic MG forms
        src_filter = 'grc' if self.lang == 'grc' else None

        # Collect candidate stripped forms at each distance level
        candidates: set[str] = set()

        # ED0: just the query itself
        candidates.add(query_stripped)

        # ED1: all edits of the stripped query
        if max_distance >= 1:
            ed1 = self._edits1(query_stripped)
            candidates.update(ed1)

        # Look up which candidates actually exist in the DB
        hits = self._lookup.spell_lookup_stripped(candidates,
                                                  src_filter=src_filter)

        # ED2: if few hits so far, expand
        if max_distance >= 2 and len(hits) < 3:
            ed2_candidates: set[str] = set()
            for e1 in ed1:
                ed2_candidates.update(self._edits1(e1))
            # Remove already-checked candidates
            ed2_candidates -= candidates
            ed2_hits = self._lookup.spell_lookup_stripped(ed2_candidates,
                                                          src_filter=src_filter)
            hits.update(ed2_hits)

        if not hits:
            return []

        return self._rank_spell_results(word, query_stripped, hits, ocr_mode)

    def _suggest_spelling_mem(self, word: str, query_stripped: str,
                              max_distance: int, ocr_mode: bool = False
                              ) -> list[tuple[str, int | float]]:
        """In-memory spelling suggestion (JSON fallback)."""
        norm_hits: set[str] = set()

        if query_stripped in self._spell_norm_set:
            norm_hits.add(query_stripped)

        if not norm_hits or max_distance >= 1:
            for candidate in self._edits1(query_stripped):
                if candidate in self._spell_norm_set:
                    norm_hits.add(candidate)

        if max_distance >= 2 and len(norm_hits) < 3:
            for e1 in self._edits1(query_stripped):
                for candidate in self._edits1(e1):
                    if candidate in self._spell_norm_set:
                        norm_hits.add(candidate)

        if not norm_hits:
            return []

        # Convert to {stripped: [original_forms]} format
        hits = {n: list(self._spell_norm_map[n]) for n in norm_hits}
        return self._rank_spell_results(word, query_stripped, hits, ocr_mode)

    @staticmethod
    def _has_breathing(s: str) -> bool:
        """Check if a string contains Greek breathing marks (polytonic)."""
        nfd = unicodedata.normalize("NFD", s)
        return "\u0313" in nfd or "\u0314" in nfd

    def _get_frequency(self, headword: str) -> int:
        """Get reference count for a headword from lsj9 frequency data.

        Lazy-loads lsj9_frequency.json on first call. Returns 0 for
        unknown headwords.
        """
        if self._hw_frequency is None:
            self._hw_frequency = {}
            if LSJ9_FREQUENCY_PATH.exists():
                with open(LSJ9_FREQUENCY_PATH, encoding="utf-8") as f:
                    self._hw_frequency = json.load(f)
        return self._hw_frequency.get(headword, 0)

    def _rank_spell_results(self, word: str, query_stripped: str,
                            hits: dict[str, list[str]],
                            ocr_mode: bool = False
                            ) -> list[tuple[str, int | float]]:
        """Rank spelling suggestions by edit distance.

        Sorts by (stripped_dist, full_dist, -frequency, form) so that
        at the same stripped distance, forms closer to the original
        polytonic input are preferred, then more common headwords win.
        Polytonic forms (with breathing marks) are preferred over
        monotonic when the input is polytonic or when using AG mode.

        In ocr_mode, uses weighted Levenshtein that gives lower cost
        to OCR-common character confusions (Greek/Latin script mixing,
        Cyrillic contamination, theta/delta, omicron/sigma).
        """
        prefer_polytonic = self.lang == "grc" or self._has_breathing(word)
        dist_fn = _weighted_levenshtein if ocr_mode else _levenshtein

        results: list[tuple[str, float, float]] = []
        for norm, originals in hits.items():
            stripped_dist = _levenshtein(query_stripped, norm)
            for original in originals:
                full_dist = dist_fn(word.lower(), original.lower())
                results.append((original, stripped_dist, full_dist))

        # Deduplicate: keep best (lowest stripped_dist, then full_dist)
        best: dict[str, tuple[float, float]] = {}
        for form, sd, fd in results:
            if form not in best or (sd, fd) < best[form]:
                best[form] = (sd, fd)

        if prefer_polytonic:
            by_sd: dict[float, list[str]] = {}
            for form, (sd, fd) in best.items():
                by_sd.setdefault(sd, []).append(form)
            for sd, forms in by_sd.items():
                poly = [f for f in forms if self._has_breathing(f)]
                if poly:
                    for f in forms:
                        if not self._has_breathing(f):
                            del best[f]

        # Sort by: stripped_dist, full_dist, then prefer common headwords
        return sorted(
            [(form, sd) for form, (sd, fd) in best.items()],
            key=lambda x: (x[1], best[x[0]][1],
                           -self._get_frequency(x[0]), x[0]),
        )

    def _predict(self, words: list[str], num_beams=4) -> list[str]:
        """Run model inference with beam search + headword filtering.

        Generates multiple candidates via beam search. Picks the
        highest-scoring candidate that is a known headword in the
        lookup table. If no candidate is a headword, returns the
        input word unchanged (better than a confidently wrong answer).

        Works with both PyTorch and ONNX backends transparently.
        """
        if not words:
            return []

        # Build headword set on first use (Wiktionary self-maps + LSJ + Cunliffe)
        if not hasattr(self, "_headwords") or self._headwords is None:
            self._headwords = {k for k, v in self._lookup.items() if k == v}
            if LSJ_HEADWORDS_PATH.exists():
                with open(LSJ_HEADWORDS_PATH, encoding="utf-8") as f:
                    self._headwords |= set(json.load(f))
            if CUNLIFFE_HEADWORDS_PATH.exists():
                with open(CUNLIFFE_HEADWORDS_PATH, encoding="utf-8") as f:
                    self._headwords |= set(json.load(f))

        max_len = max(len(w) for w in words) + 1
        src_ids = []
        for w in words:
            ids = self._vocab.encode(w)
            ids = ids + [0] * (max_len - len(ids))
            src_ids.append(ids)

        if getattr(self, '_use_onnx', False):
            import numpy as np
            # ONNX MHA reshapes require consistent sequence lengths.
            # Pad all inputs to a fixed max to avoid shape mismatches.
            ONNX_MAX_LEN = 48
            padded = []
            for ids in src_ids:
                if len(ids) < ONNX_MAX_LEN:
                    ids = ids + [0] * (ONNX_MAX_LEN - len(ids))
                padded.append(ids[:ONNX_MAX_LEN])
            src = np.array(padded, dtype=np.int64)
            src_pad_mask = (src == 0)
            beam_results = self._model.generate(
                src, src_key_padding_mask=src_pad_mask, num_beams=num_beams)
        else:
            import torch
            src = torch.tensor(src_ids, dtype=torch.long, device=self._device)
            src_pad_mask = (src == 0)
            with torch.no_grad():
                beam_results = self._model.generate(
                    src, src_key_padding_mask=src_pad_mask, num_beams=num_beams)

        results = []
        for i, candidates in enumerate(beam_results):
            decoded = [self._vocab.decode(ids) for ids, score in candidates]
            chosen = None
            for d in decoded:
                # Check headword with normalization cascade
                if any(v in self._headwords for v in (
                    d, d.lower(), to_monotonic(d), to_monotonic(d).lower(),
                    d[0].upper() + d[1:] if d else d,
                ) if v):
                    chosen = d
                    break
            if chosen is None:
                chosen = words[i]
            results.append(chosen)

        return results
