"""Dilemma - Greek lemmatizer.

Fast lookup table for known forms, custom transformer model for unknown forms.

Usage:
    from dilemma import Dilemma

    m = Dilemma()                        # loads lookup table + model
    m.lemmatize("πάθης")                # -> "παθαίνω"
    m.lemmatize("πολεμούσαν")           # -> "πολεμώ"
    m.lemmatize_batch(["δώση", "σκότωσε"])  # -> ["δίνω", "σκοτώνω"]
"""

import json
import unicodedata
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "model"
LOOKUP_PATH = Path(__file__).parent / "data" / "mg_lookup.json"
AG_LOOKUP_PATH = Path(__file__).parent / "data" / "ag_lookup.json"
MED_LOOKUP_PATH = Path(__file__).parent / "data" / "med_lookup.json"


_POLYTONIC_STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
_POLYTONIC_TO_ACUTE = {0x0300, 0x0342}

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


def strip_accents(s: str) -> str:
    """Strip all accents for fuzzy matching."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


class Dilemma:
    def __init__(self, lang="all", device=None, scale=None,
                 resolve_articles=False):
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
        """
        if lang == "both":
            lang = "all"
        self.lang = lang
        self._scale = scale
        self._resolve_articles = resolve_articles
        self._model = None
        self._vocab = None
        self._device = device
        self._lookup: dict[str, str] = {}

        # Load lookup table(s)
        # Medieval Greek is folded into MG — it's an earlier stage of
        # the same language, not a separate period.
        if lang == "all":
            # MG + Medieval first, then AG fills gaps
            for path in [LOOKUP_PATH, MED_LOOKUP_PATH, AG_LOOKUP_PATH]:
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    for k, v in data.items():
                        if k not in self._lookup:
                            self._lookup[k] = v
        elif lang == "el":
            # MG + Medieval
            for path in [LOOKUP_PATH, MED_LOOKUP_PATH]:
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    for k, v in data.items():
                        if k not in self._lookup:
                            self._lookup[k] = v
        else:
            lookup_path = {"grc": AG_LOOKUP_PATH}[lang]
            if lookup_path.exists():
                with open(lookup_path, encoding="utf-8") as f:
                    self._lookup = json.load(f)

    def _load_model(self):
        """Lazy-load the transformer model on first use."""
        if self._model is not None:
            return

        import torch
        from model import CharVocab, LemmaTransformer

        lang_dir = {"el": "el", "grc": "grc", "all": "combined"}[self.lang]

        # Find model: try scale-specific dir first, then auto-detect best available
        if self._scale is not None:
            model_path = MODEL_DIR / f"{lang_dir}-s{self._scale}"
        else:
            # Auto-detect: pick highest available scale
            model_path = None
            for s in [3, 2, 1]:
                candidate = MODEL_DIR / f"{lang_dir}-s{s}"
                if (candidate / "model.pt").exists():
                    model_path = candidate
                    break
            # Fallback to unscaled dir
            if model_path is None:
                model_path = MODEL_DIR / lang_dir

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

    def _resolve_closed_class(self, word: str) -> str | None:
        """Resolve articles/pronouns to canonical lemma if enabled."""
        if not self._resolve_articles:
            return None
        if (word in _ARTICLE_FORMS
                or to_monotonic(word) in _ARTICLE_FORMS
                or strip_accents(word.lower()) in _ARTICLE_FORMS):
            return _ARTICLE_LEMMA
        if word in _PRONOUN_LEMMAS:
            return _PRONOUN_LEMMAS[word]
        mono = to_monotonic(word)
        if mono in _PRONOUN_LEMMAS:
            return _PRONOUN_LEMMAS[mono]
        return None

    def lemmatize(self, word: str) -> str:
        """Lemmatize a single Greek word.

        Resolution order:
          1. Article/pronoun resolution (if resolve_articles=True)
          2. Crasis table (small, hand-curated)
          3. Lookup table (instant, 5M+ forms)
          4. Model with beam search + headword filter
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
        lower = word.lower()
        mono = to_monotonic(lower)
        stripped = strip_accents(lower)
        lemma = (self._lookup.get(word)
                 or self._lookup.get(lower)
                 or self._lookup.get(mono)
                 or self._lookup.get(stripped))
        if lemma:
            return lemma

        # Fall back to model
        self._load_model()
        return self._predict([word])[0]

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

            # Lookup
            lower = word.lower()
            mono = to_monotonic(lower)
            stripped = strip_accents(lower)
            lemma = (self._lookup.get(word)
                     or self._lookup.get(lower)
                     or self._lookup.get(mono)
                     or self._lookup.get(stripped))
            if lemma:
                results.append(lemma)
            else:
                    results.append(None)
                    model_indices.append(i)
                    model_words.append(word)

        if model_words:
            self._load_model()
            predictions = self._predict(model_words)
            for idx, pred in zip(model_indices, predictions):
                results[idx] = pred

        return results

    def _predict(self, words: list[str], num_beams=4) -> list[str]:
        """Run model inference with beam search + headword filtering.

        Generates multiple candidates via beam search. Picks the
        highest-scoring candidate that is a known headword in the
        lookup table. If no candidate is a headword, returns the
        input word unchanged (better than a confidently wrong answer).
        """
        if not words:
            return []

        import torch

        # Build headword set on first use (forms that map to themselves)
        if not hasattr(self, "_headwords"):
            self._headwords = {k for k, v in self._lookup.items() if k == v}

        max_len = max(len(w) for w in words) + 1
        src_ids = []
        for w in words:
            ids = self._vocab.encode(w)
            ids = ids + [0] * (max_len - len(ids))
            src_ids.append(ids)

        src = torch.tensor(src_ids, dtype=torch.long, device=self._device)
        src_pad_mask = (src == 0)

        with torch.no_grad():
            beam_results = self._model.generate(
                src, src_key_padding_mask=src_pad_mask, num_beams=num_beams)

        results = []
        for i, candidates in enumerate(beam_results):
            decoded = [self._vocab.decode(ids) for ids, score in candidates]
            # Pick first candidate that's a known headword
            chosen = None
            for d in decoded:
                if d in self._headwords or d.lower() in self._headwords:
                    chosen = d
                    break
            if chosen is None:
                # No candidate is a headword; return input unchanged
                chosen = words[i]
            results.append(chosen)

        return results
