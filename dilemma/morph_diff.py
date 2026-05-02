"""Stem-change / morphological-diff annotator.

Given a (lemma, form, lang) triple, classify each character of `form`
as STEM, ENDING, AUGMENT, REDUPLICATION, IRREGULAR_STEM, or UNCHANGED.
A consumer (e.g. a flashcard UI) can use the per-character roles to
highlight irregular root changes (augment, reduplication, vowel
gradation, suppletion) while leaving regular endings unstyled.

Usage:

    from dilemma.morph_diff import diff_form, diff_paradigm, MorphDiff, Role

    diff_form("γράφω", "ἔγραψα", lang="grc")
    # MorphDiff(form="ἔγραψα", roles=[AUGMENT, STEM, STEM, STEM,
    #          IRREGULAR_STEM, ENDING], irregular_indices=[0, 4],
    #          stem_change=True, ending_start=5)

The annotator is pure: it does not query the lookup DB, load any model,
or touch torch / onnxruntime. It is intentionally simple so a downstream
pipeline can call it on every (lemma, form) pair without paying any IO
or model-load cost. Heuristic rules cover the common AG/MG cases of
augment, reduplication, and vowel gradation; what they cannot explain
falls into IRREGULAR_STEM, which the UI can style as it sees fit.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum


__all__ = [
    "Role",
    "MorphDiff",
    "diff_form",
    "diff_paradigm",
]


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """Per-character classification of a form against its lemma stem."""

    UNCHANGED = "UNCHANGED"
    STEM = "STEM"
    ENDING = "ENDING"
    AUGMENT = "AUGMENT"
    REDUPLICATION = "REDUPLICATION"
    IRREGULAR_STEM = "IRREGULAR_STEM"


_IRREGULAR_ROLES = frozenset({
    Role.AUGMENT,
    Role.REDUPLICATION,
    Role.IRREGULAR_STEM,
})


@dataclass
class MorphDiff:
    """Result of diff_form: per-character role classification of `form`."""

    form: str
    roles: list[Role] = field(default_factory=list)
    irregular_indices: list[int] = field(default_factory=list)
    stem_change: bool = False
    ending_start: int | None = None


# ---------------------------------------------------------------------------
# Greek vowel / accent helpers
# ---------------------------------------------------------------------------


# Ancient Greek augment behaviour: a stem starting with a consonant gets
# a syllabic ε- prepended (with smooth breathing in classical orthography:
# ἐ-). A stem starting with a short vowel takes a "temporal augment", i.e.
# vowel lengthening: α/ε -> η, ο -> ω, ι -> ι (long), υ -> υ (long), αι -> ῃ,
# etc. We only model the most common cases.
_AUGMENT_VOWEL_LENGTHENINGS = {
    "α": "η",
    "ε": "η",
    "ο": "ω",
    "αι": "ῃ",
    "ει": "ῃ",
    "οι": "ῳ",
    "αυ": "ηυ",
    "ευ": "ηυ",
}

# A more permissive set: any character whose stripped-and-lowered base form
# is one of these is considered an "augment-flavoured" leading vowel, used
# when scanning the start of a past-tense form.
_AUGMENT_VOWEL_BASES = frozenset({"η", "ω", "ε"})

# Greek base vowel letters (no diacritics, lowercased).
_VOWEL_BASES = frozenset("αεηιουω")

_AG_VERB_LEMMA_RE = re.compile(r"(?:ω|ώ|μι|μαι|μην)$")
_MG_VERB_LEMMA_RE = re.compile(r"(?:ω|ώ|ομαι|άμαι|έμαι|ούμαι|ιέμαι)$")

# Common Greek noun / adjective lemma endings, longest first so the regex
# greedily matches the longest plausible ending.
_NOUN_ADJ_ENDINGS = (
    "ευς", "εως",
    "ος", "ον", "ης", "ας", "ις", "υς", "ως",
    "ός", "όν", "ής", "άς", "ίς", "ύς", "ώς",
    "α", "η", "ε", "ι", "ο", "ω", "υ",
    "ά", "ή", "έ", "ί", "ό", "ώ", "ύ",
)


# ---------------------------------------------------------------------------
# Normalization helpers (intentionally local; we deliberately keep this
# module decoupled from the rest of the dilemma package so callers do not
# pay any of dilemma's import-time cost).
# ---------------------------------------------------------------------------


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _strip_diacritics(s: str) -> str:
    """Strip every combining mark, return a lower-case base-letter string."""
    nfd = unicodedata.normalize("NFD", s or "")
    base = "".join(c for c in nfd if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", base).lower()


def _is_vowel(ch: str) -> bool:
    if not ch:
        return False
    return _strip_diacritics(ch) in _VOWEL_BASES


def _is_consonant_letter(ch: str) -> bool:
    """True if ch is a Greek letter and not a vowel."""
    if not ch:
        return False
    base = _strip_diacritics(ch)
    if not base:
        return False
    if base in _VOWEL_BASES:
        return False
    return base.isalpha()


# ---------------------------------------------------------------------------
# Stem stripping
# ---------------------------------------------------------------------------


def _strip_lemma_ending(lemma: str, lang: str) -> str:
    """Heuristically strip the inflectional ending off a citation form.

    This is a crude rule: it covers the common AG / MG verb and noun /
    adjective endings. Anything we cannot identify is returned unchanged
    (treated as an opaque stem). The stem is computed on the *accent-
    normalized* lemma so we can compare it against an accent-normalized
    form in the LCS step. Callers that need the original NFC lemma should
    keep their own copy.
    """
    if not lemma:
        return lemma

    # Work on the diacritic-stripped lower-case lemma.
    base = _strip_diacritics(lemma)

    # Verbs: classical -ω, MG -ω/-ώ, mediopassive -μαι, athematic -μι, etc.
    if lang in ("grc", "el"):
        if lang == "grc":
            m = _AG_VERB_LEMMA_RE.search(base)
        else:
            m = _MG_VERB_LEMMA_RE.search(base)
        if m:
            return base[: m.start()]

    # Nouns / adjectives.
    for end in _NOUN_ADJ_ENDINGS:
        if base.endswith(end) and len(base) > len(end):
            return base[: -len(end)]

    return base


# ---------------------------------------------------------------------------
# Augment / reduplication detection
# ---------------------------------------------------------------------------


# Modern Greek compound forms ride a leading tense particle (θα / να /
# ας) and / or a perfect auxiliary (έχω / έχεις / … / είχα / … / έχε)
# that's regular for the language but not part of the verb's lemma
# stem. Diffing γράφω against `θα γράψω` would otherwise flag θ + α +
# space as IRREGULAR_STEM because none of those characters appear in
# the lemma, even though they're a tense marker shared by every MG
# verb. We strip the prefix before alignment and shift the resulting
# indices back to the original form's coordinate frame.
import re as _re  # local alias so the module-top imports stay tidy

_MG_LEADING_PREFIX_RE = _re.compile(
    r'^(?:(?:θα|να|ας)\s+)?'
    r'(?:(?:έχω|έχεις|έχει|έχουμε|έχομε|έχετε|έχουν|έχουνε|'
    r'είχα|είχες|είχε|είχαμε|είχατε|είχαν|είχανε|έχε)\s+)?'
)


def _strip_mg_compound_prefix(form: str, lang: str) -> int:
    """Return the length of the MG compound leading prefix, or 0.

    Only fires for `lang == "el"`. Matches an optional tense particle
    (θα / να / ας) and an optional perfect-tense auxiliary, in either
    order or alone. Returns the character count consumed by the
    matched prefix (callable side strips `form[:n]` and shifts indices
    by `n`).
    """
    if lang != "el" or not form:
        return 0
    m = _MG_LEADING_PREFIX_RE.match(form)
    return m.end() if m else 0


def _detect_syllabic_augment(form_base: str, lemma_base: str) -> int:
    """Return the number of leading characters of form_base that look like
    a syllabic augment (ε- / ἐ-).

    Triggered when:
      * form_base starts with ε
      * AND the corresponding lemma_base does NOT start with ε
        (otherwise a verb like ἐθέλω wouldn't be a stem-change)
    """
    if not form_base or not lemma_base:
        return 0
    if form_base[0] != "ε":
        return 0
    if lemma_base.startswith("ε"):
        # ε in form is just the stem itself, not an augment.
        return 0
    return 1


def _detect_temporal_augment(form_base: str, lemma_base: str) -> int:
    """Detect a temporal (lengthening) augment, e.g. ἀκούω -> ἤκουσα.

    Returns 1 if the leading vowel of form_base is the lengthened version
    of the leading vowel of lemma_base, else 0.
    """
    if not form_base or not lemma_base:
        return 0
    f0 = form_base[0]
    l0 = lemma_base[0]
    if not _is_vowel(f0):
        return 0
    if not _is_vowel(l0):
        return 0
    expected = _AUGMENT_VOWEL_LENGTHENINGS.get(l0)
    if expected and form_base.startswith(expected) and not lemma_base.startswith(expected):
        return len(expected)
    # Diphthong case: lemma starts with αι / ει / οι and form starts with ῃ-
    # (handled via _AUGMENT_VOWEL_LENGTHENINGS with 2-char keys).
    two = lemma_base[:2]
    expected2 = _AUGMENT_VOWEL_LENGTHENINGS.get(two)
    if expected2 and form_base.startswith(expected2):
        return len(expected2)
    # Generic fallback: form starts with η or ω where lemma starts with α/ε/ο.
    if f0 in _AUGMENT_VOWEL_BASES and l0 in {"α", "ε", "ο"} and f0 != l0:
        return 1
    return 0


def _detect_reduplication(form_base: str, lemma_base: str) -> int:
    """Detect Cε- reduplication of a consonant-initial stem.

    Returns 2 if form_base starts with `<C>ε<lemma_stem>...` where <C>
    is the lemma's initial consonant (or the de-aspirated equivalent
    for φ/θ/χ -> π/τ/κ), else 0.

    Crucially, the remainder after `<C>ε` must continue with the lemma
    stem — otherwise plain present-tense forms like φέρε / φέρει /
    λέει look like reduplication just because their first two letters
    happen to be `<lemma_initial_consonant>ε`. Real reduplication is
    γέγραφα / λέλυκα / πέφευγα where the third character onward
    recovers the lemma stem.
    """
    if len(form_base) < 3 or not lemma_base:
        return 0
    c1 = form_base[0]
    e = form_base[1]
    if e != "ε":
        return 0
    if not _is_consonant_letter(c1):
        return 0
    l0 = lemma_base[0]
    if not _is_consonant_letter(l0):
        return 0
    # Exact match (e.g. γέγραφα <- γράφω: γ matches γ).
    deaspirate = {"φ": "π", "θ": "τ", "χ": "κ"}
    is_match = (c1 == l0) or (deaspirate.get(l0) == c1)
    if not is_match:
        return 0
    # Verify the form continues with the lemma stem after the Cε
    # prefix. We allow either an exact stem match or a prefix match
    # (the form may carry an internal vowel change after reduplication
    # but should at least share the lemma's initial consonant cluster).
    rest = form_base[2:]
    if rest.startswith(lemma_base):
        return 2
    # Short-stem fallback: accept when at least the lemma's initial
    # consonant cluster reappears (e.g. λέλυκα -> rest "λυκα" starts
    # with "λυ" which is the full lemma_base for λύω). Already covered
    # above. We additionally accept a single-char match for very short
    # stems to keep coverage on contracted verbs, but only when the
    # lemma_base is at least 2 chars long — single-letter "stems" are
    # too noisy.
    if len(lemma_base) >= 2 and rest.startswith(lemma_base[:2]):
        return 2
    return 0


# ---------------------------------------------------------------------------
# LCS-based alignment
# ---------------------------------------------------------------------------


def _lcs_table(a: str, b: str) -> list[list[int]]:
    """Length table for the longest common subsequence of a and b."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp


def _lcs_align(stem: str, body: str) -> list[bool]:
    """Greedy LCS alignment.

    Returns a list of bools the same length as `body`; True means that
    position in `body` lies on the LCS path with `stem`.
    """
    if not stem or not body:
        return [False] * len(body)
    dp = _lcs_table(stem, body)
    on_lcs = [False] * len(body)
    i, j = 0, 0
    n, m = len(stem), len(body)
    while i < n and j < m:
        if stem[i] == body[j]:
            on_lcs[j] = True
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return on_lcs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diff_form(lemma: str, form: str, lang: str = "grc") -> MorphDiff:
    """Annotate `form` against `lemma`, classifying each char by Role.

    The annotator strips the lemma down to a "stem", scans for an augment
    or reduplication prefix on the form, then aligns the rest of the form
    against the stem via LCS to mark which characters are STEM vs.
    IRREGULAR_STEM. Anything left at the tail of the form is the ENDING.

    Args:
        lemma: Citation form (e.g. "γράφω").
        form:  Inflected form (e.g. "ἔγραψα").
        lang:  ISO 639 code; "grc" for Ancient Greek, "el" for Modern Greek.

    Returns:
        MorphDiff with `roles` aligned 1:1 with the NFC code points of
        `form`. Empty inputs yield an empty MorphDiff cleanly.
    """
    form_nfc = _nfc(form)
    lemma_nfc = _nfc(lemma)
    if not form_nfc:
        return MorphDiff(form=form_nfc)

    n = len(form_nfc)
    roles: list[Role] = [Role.STEM] * n

    # Empty or trivially-equal lemma: everything is STEM (or UNCHANGED if
    # form == lemma).
    if not lemma_nfc:
        return MorphDiff(form=form_nfc, roles=roles, stem_change=False)

    if lemma_nfc == form_nfc:
        roles = [Role.UNCHANGED] * n
        return MorphDiff(form=form_nfc, roles=roles, stem_change=False)

    # MG compound prefix: strip θα/να/ας + optional perfect auxiliary
    # before alignment. The prefix characters are regular tense markers,
    # not part of the verb's lemma stem, so they should never end up
    # flagged irregular. We diff the stripped form, then shift the
    # resulting roles back to absolute form indices and label the prefix
    # itself as STEM.
    mg_prefix_len = _strip_mg_compound_prefix(form_nfc, lang)
    if mg_prefix_len >= n:
        # The "form" is just a particle / auxiliary with nothing after.
        # Treat it as fully UNCHANGED so the caller doesn't see spurious
        # irregularity for an effectively empty body.
        return MorphDiff(form=form_nfc, roles=[Role.STEM] * n, stem_change=False)

    form_diff_target = form_nfc[mg_prefix_len:]
    form_base = _strip_diacritics(form_diff_target)
    lemma_base = _strip_diacritics(lemma_nfc)
    stem_base = _strip_lemma_ending(lemma_nfc, lang)

    # ------------------------------------------------------------------
    # Pass 1: prefix detection (augment / reduplication).
    # ------------------------------------------------------------------
    prefix_role: Role | None = None
    prefix_len = _detect_reduplication(form_base, lemma_base)
    if prefix_len:
        prefix_role = Role.REDUPLICATION
    if not prefix_len:
        prefix_len = _detect_syllabic_augment(form_base, lemma_base)
        if prefix_len:
            prefix_role = Role.AUGMENT
    if not prefix_len:
        prefix_len = _detect_temporal_augment(form_base, lemma_base)
        if prefix_len:
            prefix_role = Role.AUGMENT

    # Cap prefix_len: if marking it as a prefix would leave nothing for
    # the stem to match against, back off.
    if prefix_len >= n:
        prefix_len = 0
        prefix_role = None

    # ------------------------------------------------------------------
    # Pass 2: align the post-prefix body against the lemma stem via LCS.
    # ------------------------------------------------------------------
    body_start = prefix_len
    body = form_base[body_start:]

    on_lcs = _lcs_align(stem_base, body) if stem_base else [False] * len(body)

    # Find the last LCS-matched position in the body. Everything after the
    # stem region is the ENDING; everything in the stem region is STEM
    # (if on the LCS path) or IRREGULAR_STEM (if not).
    last_match = -1
    for idx, hit in enumerate(on_lcs):
        if hit:
            last_match = idx

    # Where does the inflectional ending start (in body coordinates)?
    # Two competing signals:
    #   * last_match + 1: every LCS hit must be inside the stem region.
    #   * len(stem_base): the stem region of the form should be roughly
    #     as long as the lemma stem (this is what catches "γρα-ψ-α" where
    #     ψ is stem-region IRR rather than ending).
    # We take the larger of the two but cap to len(body) - 1 so the form
    # always has at least one ENDING character (when it has any chars).
    body_len = len(body)
    if body_len == 0:
        ending_start_in_body = 0
    else:
        target = min(len(stem_base), body_len - 1) if stem_base else 0
        ending_start_in_body = max(last_match + 1, target)
        # Never claim more of the body than exists.
        if ending_start_in_body > body_len:
            ending_start_in_body = body_len

    for idx in range(body_len):
        absolute = mg_prefix_len + body_start + idx
        if idx >= ending_start_in_body:
            roles[absolute] = Role.ENDING
        elif on_lcs[idx]:
            roles[absolute] = Role.STEM
        else:
            roles[absolute] = Role.IRREGULAR_STEM

    # Apply the augment / reduplication prefix role last so it survives
    # the body assignment loop. Indices are relative to the form past the
    # MG compound prefix (θα / να / έχω / ...), which we leave as STEM.
    for idx in range(prefix_len):
        roles[mg_prefix_len + idx] = prefix_role  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Compute summary fields.
    # ------------------------------------------------------------------
    irregular_indices = [
        i for i, r in enumerate(roles) if r in _IRREGULAR_ROLES
    ]
    stem_change = bool(irregular_indices)

    ending_start: int | None = None
    for i, r in enumerate(roles):
        if r == Role.ENDING:
            ending_start = i
            break

    return MorphDiff(
        form=form_nfc,
        roles=roles,
        irregular_indices=irregular_indices,
        stem_change=stem_change,
        ending_start=ending_start,
    )


def diff_paradigm(
    lemma: str,
    forms: dict[str, str],
    lang: str = "grc",
) -> dict[str, MorphDiff]:
    """Annotate every (key -> form) entry in a paradigm against `lemma`."""
    return {
        key: diff_form(lemma, form, lang=lang)
        for key, form in forms.items()
    }
