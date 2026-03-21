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
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "model"
LOOKUP_PATH = Path(__file__).parent / "data" / "mg_lookup.json"
AG_LOOKUP_PATH = Path(__file__).parent / "data" / "ag_lookup.json"
MED_LOOKUP_PATH = Path(__file__).parent / "data" / "med_lookup.json"
MG_POS_LOOKUP_PATH = Path(__file__).parent / "data" / "mg_pos_lookup.json"
AG_POS_LOOKUP_PATH = Path(__file__).parent / "data" / "ag_pos_lookup.json"
TREEBANK_POS_LOOKUP_PATH = Path(__file__).parent / "data" / "treebank_pos_lookup.json"
LSJ_HEADWORDS_PATH = Path(__file__).parent / "data" / "lsj_headwords.json"
CUNLIFFE_HEADWORDS_PATH = Path(__file__).parent / "data" / "cunliffe_headwords.json"


_POLYTONIC_STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
_POLYTONIC_TO_ACUTE = {0x0300, 0x0342}

# Elision mark: U+0313 COMBINING COMMA ABOVE (repurposed as apostrophe
# in polytonic Greek text). Also handle right single quote U+2019 and
# modifier letter apostrophe U+02BC.
_ELISION_MARKS = {"\u0313", "\u2019", "\u02BC", "'", "\u1FBD"}

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


class Dilemma:
    def __init__(self, lang="all", device=None, scale=None,
                 resolve_articles=False, normalize=False, period=None):
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
        """
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

        # Per-language lookup tables (always loaded separately for verbose mode)
        self._mg_lookup: dict[str, str] = {}
        self._med_lookup: dict[str, str] = {}
        self._ag_lookup: dict[str, str] = {}

        # Combined lookup (respects lang priority)
        self._lookup: dict[str, str] = {}

        # POS-indexed disambiguation table: {form: {upos: lemma}}
        self._pos_lookup: dict[str, dict[str, str]] = {}

        self._load_lookups()
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
        """Load lookup tables, keeping per-language copies for verbose mode."""
        if LOOKUP_PATH.exists():
            with open(LOOKUP_PATH, encoding="utf-8") as f:
                self._mg_lookup = json.load(f)
        if MED_LOOKUP_PATH.exists():
            with open(MED_LOOKUP_PATH, encoding="utf-8") as f:
                self._med_lookup = json.load(f)
        if AG_LOOKUP_PATH.exists():
            with open(AG_LOOKUP_PATH, encoding="utf-8") as f:
                self._ag_lookup = json.load(f)

        # Build combined lookup respecting lang priority.
        # Real mappings (form != lemma) override self-mappings (form == lemma)
        # regardless of language priority. Self-maps just mean "this is a
        # headword" — a weaker signal than "this is an inflected form of X".
        # E.g. MG τούτου→τούτου (self-map) should NOT block AG τούτου→οὗτος.
        # Also treat "polytonic -> monotonic of same word" as a notation
        # variant (e.g. MG ἐν→εν), not a real mapping. When both languages
        # have notation variants, prefer the one that preserves the input form.
        def _is_self_map(form, lemma):
            return (form == lemma
                    or strip_accents(form.lower()) == strip_accents(lemma.lower()))

        if self.lang == "all":
            for data in [self._mg_lookup, self._med_lookup, self._ag_lookup]:
                for k, v in data.items():
                    if k not in self._lookup:
                        self._lookup[k] = v
                    elif _is_self_map(k, self._lookup[k]) and not _is_self_map(k, v):
                        # Existing is a self-map, new is a real mapping - override
                        self._lookup[k] = v
                    elif (_is_self_map(k, self._lookup[k])
                          and _is_self_map(k, v) and v == k
                          and self._lookup[k] != k):
                        # Both are notation variants, but new one preserves
                        # the exact form (true self-map) - prefer it
                        # E.g. MG ἐν→εν replaced by AG ἐν→ἐν
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

        # Load POS disambiguation tables.
        # Priority: treebank (gold) > MG Wiktionary > AG Wiktionary.
        # Lower-priority sources only fill in form+upos slots not already set.

        # 1. Treebank POS lookup (gold-annotated, highest priority for AG)
        if self.lang in ("all", "grc") and TREEBANK_POS_LOOKUP_PATH.exists():
            with open(TREEBANK_POS_LOOKUP_PATH, encoding="utf-8") as f:
                tb_pos = json.load(f)
            for form, upos_lemmas in tb_pos.items():
                if form not in self._pos_lookup:
                    self._pos_lookup[form] = {}
                self._pos_lookup[form].update(upos_lemmas)

        # 2. MG POS lookup (Wiktionary-derived)
        if self.lang in ("all", "el") and MG_POS_LOOKUP_PATH.exists():
            with open(MG_POS_LOOKUP_PATH, encoding="utf-8") as f:
                mg_pos = json.load(f)
            for form, upos_lemmas in mg_pos.items():
                if form not in self._pos_lookup:
                    self._pos_lookup[form] = {}
                for upos, lemma in upos_lemmas.items():
                    if upos not in self._pos_lookup[form]:
                        self._pos_lookup[form][upos] = lemma

        # 3. AG Wiktionary POS lookup (fills remaining gaps)
        if self.lang in ("all", "grc") and AG_POS_LOOKUP_PATH.exists():
            with open(AG_POS_LOOKUP_PATH, encoding="utf-8") as f:
                ag_pos = json.load(f)
            for form, upos_lemmas in ag_pos.items():
                if form not in self._pos_lookup:
                    self._pos_lookup[form] = {}
                for upos, lemma in upos_lemmas.items():
                    if upos not in self._pos_lookup[form]:
                        self._pos_lookup[form][upos] = lemma

    def _find_model_dir(self):
        """Find the best available model directory."""
        lang_dir = {"el": "el", "grc": "grc", "all": "combined"}[self.lang]
        if self._scale is not None:
            return MODEL_DIR / f"{lang_dir}-s{self._scale}"
        for s in [3, 2, 1]:
            candidate = MODEL_DIR / f"{lang_dir}-s{s}"
            if ((candidate / "encoder.onnx").exists()
                    or (candidate / "model.pt").exists()):
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
        """Load PyTorch model (original path)."""
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
        self._model = LemmaTransformer(**cfg)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()
        self._use_onnx = False

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

        Skips mono/stripped matches that are trivially short (1-2 chars)
        and map to themselves — these are usually false positives from
        accent stripping on elided or particle forms.
        """
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

        # Sort by vowel frequency (same ranking as _expand_elision)
        _VOWEL_RANK = {v: i for i, v in enumerate("εαοιηυω")}
        _ACC_VOWEL_RANK = {"ά": 1, "έ": 0, "ί": 4, "ό": 2,
                           "ή": 3, "ύ": 5, "ώ": 6}

        def _rank(item):
            expanded, lemma, vowel = item
            vrank = _ACC_VOWEL_RANK.get(vowel, _VOWEL_RANK.get(vowel, 10))
            return (vrank, len(lemma))

        results.sort(key=_rank)
        return results

    def _lang_of(self, form: str) -> str:
        """Determine which language table a form comes from."""
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

    def _is_proper(self, lemma: str) -> bool:
        """Check if a lemma is a proper noun (capitalized headword)."""
        return bool(lemma) and lemma[0].isupper()

    def lemmatize(self, word: str) -> str:
        """Lemmatize a single Greek word.

        Resolution order:
          1. Article/pronoun resolution (if resolve_articles=True)
          2. Crasis table (small, hand-curated)
          3. Lookup table (instant, 5M+ forms)
          4. Elision expansion (strip mark, try vowels against lookup)
          5. Model with beam search + headword filter
        """
        # Resolve articles/pronouns to canonical lemma
        closed = self._resolve_closed_class(word)
        if closed is not None:
            return closed

        # Check crasis first (before lookup, since crasis forms are
        # Wiktionary headwords that self-map in the lookup)
        from crasis import resolve_crasis
        crasis_result = resolve_crasis(word) or resolve_crasis(to_monotonic(word))
        if crasis_result is not None:
            return crasis_result

        # Lookup: exact -> lowercase -> monotonic -> accent-stripped
        lemma = self._lookup_word(word)
        if lemma:
            return lemma

        # Elision expansion (after lookup, so known words like εἰ/οὐ
        # aren't falsely caught by smooth-breathing-as-elision)
        elision_lemma = self._expand_elision(word)
        if elision_lemma:
            return elision_lemma

        # Normalizer: try orthographic variants against lookup
        if self._normalizer:
            for candidate in self._normalizer.normalize(word):
                lemma = self._lookup_word(candidate)
                if lemma:
                    return lemma

        # Fall back to model
        self._load_model()
        return self._predict([word])[0]

    def lemmatize_pos(self, word: str, upos: str) -> str:
        """Lemmatize with POS-aware disambiguation.

        If the form is in the POS lookup table and the given UPOS tag
        matches a known disambiguation, returns the POS-specific lemma.
        Otherwise falls back to regular lemmatize().

        Args:
            word: Greek word form.
            upos: Universal POS tag (NOUN, VERB, ADJ, etc.).

        Returns:
            The lemma string.
        """
        # Closed class first (articles, pronouns)
        closed = self._resolve_closed_class(word)
        if closed is not None:
            return closed

        # Check POS lookup with cascade: exact, lower, grave_to_acute, monotonic, stripped
        lower = word.lower()
        acute = grave_to_acute(lower)
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        for variant in (word, lower, acute, mono, stripped):
            pos_entry = self._pos_lookup.get(variant)
            if pos_entry and upos in pos_entry:
                return pos_entry[upos]

        # Fall back to regular lemmatize
        return self.lemmatize(word)

    def lemmatize_batch_pos(self, words: list[str], upos_tags: list[str]) -> list[str]:
        """Lemmatize a batch of words with POS-aware disambiguation.

        For each word, tries POS-specific lookup first, then falls back
        to regular batch lemmatization for unresolved words.

        Args:
            words: List of Greek word forms.
            upos_tags: List of UPOS tags, one per word.

        Returns:
            List of lemma strings.
        """
        assert len(words) == len(upos_tags), (
            f"words and upos_tags must have same length: {len(words)} vs {len(upos_tags)}"
        )

        results = [None] * len(words)
        fallback_indices = []
        fallback_words = []

        for i, (word, upos) in enumerate(zip(words, upos_tags)):
            # Closed class first
            closed = self._resolve_closed_class(word)
            if closed is not None:
                results[i] = closed
                continue

            # POS lookup cascade
            lower = word.lower()
            mono = to_monotonic(lower)
            stripped = strip_accents(lower)
            pos_hit = None
            for variant in (word, lower, mono, stripped):
                pos_entry = self._pos_lookup.get(variant)
                if pos_entry and upos in pos_entry:
                    pos_hit = pos_entry[upos]
                    break

            if pos_hit is not None:
                results[i] = pos_hit
            else:
                fallback_indices.append(i)
                fallback_words.append(word)

        # Batch-lemmatize the fallbacks
        if fallback_words:
            fallback_lemmas = self.lemmatize_batch(fallback_words)
            for idx, lemma in zip(fallback_indices, fallback_lemmas):
                results[idx] = lemma

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
        if not candidates:
            try:
                self._load_model()
                pred = self._predict([word])[0]
                if pred != word:
                    _add(pred, source="model", score=0.5)
            except (FileNotFoundError, RuntimeError):
                pass

        # If still nothing, return the word itself
        if not candidates:
            _add(word, source="identity", score=0.0)

        # Sort: non-proper before proper, then by score descending
        candidates.sort(key=lambda c: (c.proper, -c.score))

        return candidates

    def lemmatize_batch(self, words: list[str]) -> list[str]:
        """Lemmatize a batch of words. Uses model only for unknowns."""
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
            for idx, pred in zip(model_indices, predictions):
                results[idx] = pred

        return results

    # ---- Spelling correction ----

    # Greek lowercase letters for ED1 candidate generation
    _GREEK_LETTERS = "αβγδεζηθικλμνξοπρσςτυφχψω"

    def _build_spell_index(self):
        """Build the accent-stripped index for spelling correction.

        Maps each accent-stripped form to all its original polytonic
        variants in the lookup table. This collapses the 8-11M entry
        lookup into a ~1-3M normalized set, making ED1 candidate
        generation fast.
        """
        if hasattr(self, "_spell_norm_map"):
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

    def suggest_spelling(self, word: str, max_distance: int = 2
                         ) -> list[tuple[str, int]]:
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

        Returns:
            List of (corrected_form, distance) tuples. Empty if no
            suggestions found within max_distance.
        """
        self._build_spell_index()

        query_stripped = strip_accents(word.lower())

        # Collect normalized matches at each distance level
        norm_hits: set[str] = set()

        # ED0: exact match on stripped form (handles pure diacritic errors)
        if query_stripped in self._spell_norm_set:
            norm_hits.add(query_stripped)

        # ED1 on stripped forms
        if not norm_hits or max_distance >= 1:
            for candidate in self._edits1(query_stripped):
                if candidate in self._spell_norm_set:
                    norm_hits.add(candidate)

        # ED2: edits of edits (only if requested and ED0/ED1 found few results)
        if max_distance >= 2 and len(norm_hits) < 3:
            for e1 in self._edits1(query_stripped):
                for candidate in self._edits1(e1):
                    if candidate in self._spell_norm_set:
                        norm_hits.add(candidate)

        if not norm_hits:
            return []

        # Expand normalized hits back to original polytonic forms.
        # Rank by edit distance on stripped forms (so diacritic
        # differences don't inflate the distance), but also compute
        # the full-form distance for the returned value.
        results: list[tuple[str, int, int]] = []  # (form, stripped_dist, full_dist)
        for norm in norm_hits:
            stripped_dist = _levenshtein(query_stripped, norm)
            for original in self._spell_norm_map[norm]:
                full_dist = _levenshtein(word.lower(), original.lower())
                results.append((original, stripped_dist, full_dist))

        # Deduplicate preserving best stripped distance
        best: dict[str, tuple[int, int]] = {}
        for form, sd, fd in results:
            if form not in best or sd < best[form][0]:
                best[form] = (sd, fd)

        return sorted(
            [(form, sd) for form, (sd, fd) in best.items()],
            key=lambda x: (x[1], x[0]),
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
