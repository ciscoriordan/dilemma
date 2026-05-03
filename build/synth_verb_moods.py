#!/usr/bin/env python3
"""Procedural synthesis of finite verb mood forms from principal parts.

Given an Ancient Greek verb lemma and the principal-parts dict produced
by ``build/lsj_principal_parts.parse_principal_parts``, this module
generates the missing finite-mood paradigm cells (subjunctive, optative,
imperative, aorist infinitive) by stem-templating with regular endings.

Scope (v2):
  - Active, middle, and passive voices.
  - Plain thematic -ω verbs only. Contracts (-άω/-έω/-όω), athematic
    (-μι/-μαι), and lemmas not ending in ω are skipped.
  - Tense / mood combinations covered:
      * present subjunctive    (active / middle; mp shared in present)
      * present optative       (active / middle)
      * present imperative     (active / middle; 2sg, 3sg, 2pl, 3pl, duals
                                in active)
      * aorist subjunctive     (active / middle / passive)
      * aorist optative        (active / middle / passive; passive carries
                                duals to mirror jtauber)
      * aorist imperative      (active / middle / passive; sigmatic only)
      * aorist infinitive      (active / middle / passive)
      * present infinitive     (active / middle)
  - The function only produces forms; the caller decides which slots
    to write (and whether to overwrite or only fill empty cells).
  - Accent: the synthesis preserves the lemma / fut-stem accent on stem
    syllables, but does NOT compute fresh accent placement on
    enclitic-final cells (3sg/3pl imperative). Those endings carry an
    inherent acute, but the stem-accent in the synthesised form remains
    where the input has it. Result: 3sg/3pl imperatives may be
    accent-imperfect on multi-syllable stems, but the segmentation is
    always correct so the caller's training pipeline still benefits.
  - jtauber emits one ``middle_present_*`` column for the present
    mediopassive (middle and passive share the same form in the present
    system). We mirror that: this module emits ``middle_present_*`` keys
    only -- no ``passive_present_*``. In the aorist system middle and
    passive split into distinct stems (sigmatic σ-stem vs θη-stem), so
    ``middle_aorist_*`` and ``passive_aorist_*`` get separate cells.

What this module does NOT do (deferred to v3+):
  - Aor-2 (thematic ``ἔλιπον``-style) imperative: ending pattern is
    different (2sg ``λίπε`` vs sigmatic ``λῦσον``).
  - Contract verbs: contract-future / contract-aorist forms have
    distinct vowel-stem rules.
  - Perfect-system synthesis (perfect subj/opt is rare and almost
    always periphrastic).
  - Future-system finite moods (no future subjunctive in Greek; future
    optative exists but is mostly indirect-discourse).
  - Participles: full case×gender×number declension lives in
    ``build/synth_verb_participles.py``.

Pure-Python; no I/O, no DB, no external deps. Mirrors the shape of
``dilemma.morph_diff`` and ``build.lsj_principal_parts``.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, Optional


__all__ = [
    "synthesize_active_moods",
    "synthesize_mp_moods",
    "is_thematic_omega",
]


# ---------------------------------------------------------------------------
# Lemma classification
# ---------------------------------------------------------------------------


_GREEK_VOWELS = set("αεηιουωᾰᾱ")


def _strip_accents_lower(s: str) -> str:
    """Diacritic-stripped, lowercased version of a Greek string."""
    nfd = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in nfd if not unicodedata.combining(c)).lower()


def is_thematic_omega(lemma: str) -> bool:
    """True iff ``lemma`` is a plain thematic -ω verb suitable for
    template-based mood synthesis.

    Filters out contract (-άω/-έω/-όω), athematic (-μι/-μαι), and any
    lemma that doesn't end in plain ω. Mirrors
    ``dilemma.paradigm._is_regular_thematic_omega`` but accepts both
    bare -ω and accented -ώ (ποιῶ as an alternate citation form, etc.).
    """
    if not lemma:
        return False
    base = _strip_accents_lower(lemma)
    if not base.endswith("ω"):
        return False
    if base.endswith(("αω", "εω", "οω")):
        return False
    if base.endswith(("μι", "μαι")):
        return False
    if len(base) < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Stem extraction
# ---------------------------------------------------------------------------


def _strip_final_omega(form: str) -> Optional[str]:
    """Drop trailing ω from a form, preserving accents on the stem.

    Returns None if the final character is not ω (or accented ώ).
    """
    if not form:
        return None
    nfc = unicodedata.normalize("NFC", form)
    if nfc.endswith("ω") or nfc.endswith("ώ"):
        return nfc[:-1]
    return None


def _aor_terminal_cluster(aor_form: str) -> Optional[str]:
    """Identify the sigmatic terminal cluster (σ / ψ / ξ) of an aor 1sg
    active form. Returns ``None`` for non-sigmatic / second-aorist
    patterns.

    ``ἔλυσα``     -> ``σ``
    ``ἔγραψα``    -> ``ψ``
    ``ἤγαγον``    -> None (aor-2)
    ``ἐπαίδευσα`` -> ``σ``
    """
    if not aor_form:
        return None
    plain = _strip_accents_lower(aor_form)
    if not plain or len(plain) < 2:
        return None
    if not (plain.endswith("α") or plain.endswith("ᾰ")):
        return None
    cluster = plain[-2]
    if cluster in ("σ", "ψ", "ξ"):
        return cluster
    return None


def _aorist_stem_from_lemma_and_aor(
    lemma: str, aor_form: str
) -> Optional[str]:
    """Build a sigmatic aorist stem by grafting the aor form's
    terminal consonant cluster onto the lemma's present stem.

    The present stem keeps the lemma's accent. The aor form's σ/ψ/ξ
    replaces the present stem's final consonant when the swap is
    consistent with the standard future-stem composition rule:

        π/β/φ + σ -> ψ
        κ/γ/χ + σ -> ξ
        τ/δ/θ + σ -> σ  (dental drops before σ)
        ν + σ     -> σ
        vowel + σ -> σ

    For verbs whose aorist stem differs from the present stem in a
    way these rules don't predict (suppletion, ablaut, second-aorist
    formed off a different root), this returns ``None`` and the caller
    skips synthesis.
    """
    if not lemma or not aor_form:
        return None
    cluster = _aor_terminal_cluster(aor_form)
    if cluster is None:
        return None
    pres_stem = _present_stem(lemma)
    if not pres_stem:
        return None
    plain_pres = _strip_accents_lower(pres_stem)
    if not plain_pres:
        return None
    last = plain_pres[-1]
    expected_cluster: Optional[str] = None
    if last in ("π", "β", "φ"):
        expected_cluster = "ψ"
    elif last in ("κ", "γ", "χ"):
        expected_cluster = "ξ"
    elif last in ("τ", "δ", "θ", "ν", "ζ"):
        expected_cluster = "σ"
    elif last in _GREEK_VOWELS:
        expected_cluster = "σ"
    if expected_cluster is None or expected_cluster != cluster:
        return None
    # Splice the cluster onto the lemma stem, replacing the final
    # consonant unless the stem ended in a vowel (then we just append
    # the σ).
    if last in _GREEK_VOWELS:
        return pres_stem + cluster
    # Drop the final NFC code point and append the cluster. Diacritics
    # ride on the previous letter (vowel), so this preserves the accent.
    return pres_stem[:-1] + cluster


def _aorist_stem_from_fut(fut_form: str) -> Optional[str]:
    """Extract the sigmatic aorist stem from a fut 1sg active form.

    ``λύσω`` -> ``λύσ`` -- preserves accent.
    Returns None if the form doesn't end in ω.

    Validity check: the future stem ends in σ or ψ or ξ for sigmatic
    futures. Verbs with deponent / liquid futures (-εῖ, -οῦμαι) get
    skipped here so we don't templatise non-sigmatic patterns.
    """
    stem = _strip_final_omega(fut_form)
    if stem is None:
        return None
    plain = _strip_accents_lower(stem)
    # Sigmatic future stems end in σ/ψ/ξ (also -ησ for some, -ασ).
    # Reject contract / liquid futures (φανῶ, μενῶ) and middle deponents.
    if not plain or plain[-1] not in ("σ", "ψ", "ξ"):
        return None
    return stem


def _present_stem(lemma: str) -> Optional[str]:
    """Strip the trailing ω/ώ from a thematic lemma."""
    return _strip_final_omega(lemma)


# Tonal accents we strip for ending-accented cells (3sg/3pl imperative).
# Macron / breve quantitative marks (U+0304 / U+0306) are preserved.
_TONAL_ACCENTS = {0x0301, 0x0300, 0x0342}  # acute, grave, circumflex


def _strip_tonal_accents(form: str) -> str:
    """Remove acute/grave/circumflex while keeping macron, breve, and
    breathing marks. Used to neutralise the stem accent on cells whose
    ending carries an inherent acute (3sg / 3pl imperative).
    """
    nfd = unicodedata.normalize("NFD", form or "")
    return unicodedata.normalize(
        "NFC",
        "".join(c for c in nfd if ord(c) not in _TONAL_ACCENTS),
    )


# ---------------------------------------------------------------------------
# Ending tables (active voice)
# ---------------------------------------------------------------------------


# Present-active thematic endings. Used for present subj / opt / imp
# when the cell is empty.
_PRES_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ω",
    ("2", "sg"): "ῃς",
    ("3", "sg"): "ῃ",
    ("1", "pl"): "ωμεν",
    ("2", "pl"): "ητε",
    ("3", "pl"): "ωσι(ν)",
    ("2", "du"): "ητον",
    ("3", "du"): "ητον",
}

_PRES_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οιμι",
    ("2", "sg"): "οις",
    ("3", "sg"): "οι",
    ("1", "pl"): "οιμεν",
    ("2", "pl"): "οιτε",
    ("3", "pl"): "οιεν",
    ("2", "du"): "οιτον",
    ("3", "du"): "οίτην",
}

_PRES_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ε",
    ("3", "sg"): "έτω",
    ("2", "pl"): "ετε",
    ("3", "pl"): "όντων",
    ("2", "du"): "ετον",
    ("3", "du"): "έτων",
}


# Aorist-active sigmatic endings. The stem already carries the σ
# (e.g. ``λύσ-`` from ``λύσω``), so the endings are vowel-initial.
_AOR_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ω",
    ("2", "sg"): "ῃς",
    ("3", "sg"): "ῃ",
    ("1", "pl"): "ωμεν",
    ("2", "pl"): "ητε",
    ("3", "pl"): "ωσι(ν)",
}

_AOR_OPT: Dict[tuple, str] = {
    ("1", "sg"): "αιμι",
    ("2", "sg"): "αις",
    ("3", "sg"): "αι",
    ("1", "pl"): "αιμεν",
    ("2", "pl"): "αιτε",
    ("3", "pl"): "αιεν",
}

_AOR_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ον",
    ("3", "sg"): "άτω",
    ("2", "pl"): "ατε",
    ("3", "pl"): "άντων",
}

_AOR_INF = "αι"


# ---------------------------------------------------------------------------
# Middle / passive stem helpers
# ---------------------------------------------------------------------------


def _strip_augment(form: str) -> Optional[str]:
    """Strip a syllabic augment ε- (with breathing) from a 1sg form.

    ``ἐλύθην`` -> ``λύθην``.  Returns ``None`` if the form doesn't start
    with an augment we can safely strip (e.g. lengthened-vowel temporal
    augments like ``ἤγαγον`` -- those signal aor-2 anyway).

    Mirrors ``synth_verb_participles._strip_augment``.
    """
    if not form:
        return None
    nfd = unicodedata.normalize("NFD", form)
    chars = list(nfd)
    if not chars:
        return None
    if chars[0] != "ε":
        return None
    i = 1
    while i < len(chars) and unicodedata.combining(chars[i]):
        i += 1
    rest = "".join(chars[i:])
    return unicodedata.normalize("NFC", rest)


def _aor_passive_stem(aor_p_form: str) -> Optional[str]:
    """Extract the aorist passive stem from a 1sg form like ``ἐλύθην``.

    Strips the augment, then strips the trailing ``ην``. Validates that
    the resulting stem ends in θ (sigmatic θη-aorist). Stem-accents are
    preserved -- the caller decides whether to strip them when splicing
    accent-bearing endings.

    Mirrors ``synth_verb_participles._aor_passive_stem`` but does NOT
    pre-strip stem accents: callers here splice both stem-accented
    (athematic-style ``ώ``, ``ῶ``) and ending-accented (``θητι``)
    endings, so the strip happens at emit time.
    """
    if not aor_p_form:
        return None
    body = _strip_augment(aor_p_form)
    if body is None:
        return None
    plain_body = _strip_accents_lower(body)
    if not plain_body.endswith("ην"):
        return None
    nfc = unicodedata.normalize("NFC", body)
    if len(nfc) < 2:
        return None
    stem = nfc[:-2]
    plain_stem = _strip_accents_lower(stem)
    if not plain_stem or plain_stem[-1] != "θ":
        return None
    return stem


# ---------------------------------------------------------------------------
# Ending tables (middle voice)
# ---------------------------------------------------------------------------


# Present-middle thematic endings. jtauber labels these as ``middle_*``
# even though middle and passive share these forms in the present.
_PRES_MID_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ωμαι",
    ("2", "sg"): "ῃ",
    ("3", "sg"): "ηται",
    ("1", "pl"): "ώμεθα",
    ("2", "pl"): "ησθε",
    ("3", "pl"): "ωνται",
}


_PRES_MID_OPT: Dict[tuple, str] = {
    ("1", "sg"): "οίμην",
    ("2", "sg"): "οιο",
    ("3", "sg"): "οιτο",
    ("1", "pl"): "οίμεθα",
    ("2", "pl"): "οισθε",
    ("3", "pl"): "οιντο",
}


# Present-middle imperative.
#  2sg ου comes from -εο contracted, lemma-stem accent rides through.
#  3sg/2pl/3pl carry their own accent on the ending.
_PRES_MID_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ου",
    ("3", "sg"): "έσθω",
    ("2", "pl"): "εσθε",
    ("3", "pl"): "έσθων",
}


_PRES_MID_INF = "εσθαι"


# Aorist-middle endings (sigmatic σ-stem). Stem already carries σ.
_AOR_MID_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ωμαι",
    ("2", "sg"): "ῃ",
    ("3", "sg"): "ηται",
    ("1", "pl"): "ώμεθα",
    ("2", "pl"): "ησθε",
    ("3", "pl"): "ωνται",
}


_AOR_MID_OPT: Dict[tuple, str] = {
    ("1", "sg"): "αίμην",
    ("2", "sg"): "αιο",
    ("3", "sg"): "αιτο",
    ("1", "pl"): "αίμεθα",
    ("2", "pl"): "αισθε",
    ("3", "pl"): "αιντο",
}


# Aorist-middle imperative. 2sg ``-αι`` (e.g. λῦσαι) -- circumflex on
# the stem (jtauber preserves it). 3sg/2pl/3pl have ending-accent.
_AOR_MID_IMP: Dict[tuple, str] = {
    ("2", "sg"): "αι",
    ("3", "sg"): "άσθω",
    ("2", "pl"): "ασθε",
    ("3", "pl"): "άσθων",
}


_AOR_MID_INF = "ασθαι"


# ---------------------------------------------------------------------------
# Ending tables (passive aorist)
# ---------------------------------------------------------------------------


# Aorist-passive endings (athematic θη-aorist). Note: subj/opt take
# active-style endings on a passive stem -- this is correct: the aor
# passive is morphologically active.  The contraction with the θη/θε-
# linker creates circumflex/acute on the joined vowel.
#
# subj 1sg: λυθῶ <- λυθή-ω with ε+ω contraction. Endings as listed are
# what gets appended after the θ stem.
_AOR_PASS_SUBJ: Dict[tuple, str] = {
    ("1", "sg"): "ῶ",
    ("2", "sg"): "ῇς",
    ("3", "sg"): "ῇ",
    ("1", "pl"): "ῶμεν",
    ("2", "pl"): "ῆτε",
    ("3", "pl"): "ῶσι(ν)",
}


# opt 1sg: λυθείην <- θη + ιη + ν. Athematic optative ι-η-marker fuses
# with the η stem-vowel to ει. jtauber emits both pl forms for 1pl and
# the duals (2du / 3du).
_AOR_PASS_OPT: Dict[tuple, str] = {
    ("1", "sg"): "είην",
    ("2", "sg"): "είης",
    ("3", "sg"): "είη",
    ("1", "pl"): "εῖμεν",
    ("2", "pl"): "εῖτε",
    ("3", "pl"): "εῖεν",
    ("2", "du"): "εῖτον",
    ("3", "du"): "είτην",
}


# imp 2sg: -ητι (or -ηθι after stems where dissimilation gives -θητι ->
#                 -θηθι, but for regular θη-aorists -θητι is the form).
# imp 3sg: -ήτω -- ending-accented.
# imp 2pl: -ητε -- stem-accented.
# imp 3pl: -έντων -- 3rd-decl participle form, ending-accented; the η
#                    of the stem drops before the participle linker -ε-.
_AOR_PASS_IMP: Dict[tuple, str] = {
    ("2", "sg"): "ητι",
    ("3", "sg"): "ήτω",
    ("2", "pl"): "ητε",
    ("3", "pl"): "έντων",
}


# Set of (person, number) keys whose passive-aorist imperative ending
# carries its own accent and so requires the stem accent to be stripped
# first, AND additionally drops the stem-final η before splicing.
_AOR_PASS_IMP_3PL_DROPS_ETA = {("3", "pl")}


_AOR_PASS_INF = "ῆναι"


# Set of mp/passive cells whose ending carries an inherent accent.
# Stem-accents on these get neutralised before splicing so we don't
# produce double-accent forms.
_MP_END_ACCENTED_KEYS = {
    "middle_present_optative_1sg",   # οίμην
    "middle_present_optative_1pl",   # οίμεθα
    "middle_present_subjunctive_1pl",  # ώμεθα
    "middle_present_imperative_3sg",   # έσθω
    "middle_present_imperative_3pl",   # έσθων
    "middle_aorist_optative_1sg",   # αίμην
    "middle_aorist_optative_1pl",   # αίμεθα
    "middle_aorist_subjunctive_1pl",  # ώμεθα
    "middle_aorist_imperative_3sg",   # άσθω
    "middle_aorist_imperative_3pl",   # άσθων
    # Passive aorist subjunctive: ALL cells are ending-accented (the
    # θη + ω contraction produces ω/ῇς/ῇ etc.). Stem accent is dropped.
    "passive_aorist_subjunctive_1sg",
    "passive_aorist_subjunctive_2sg",
    "passive_aorist_subjunctive_3sg",
    "passive_aorist_subjunctive_1pl",
    "passive_aorist_subjunctive_2pl",
    "passive_aorist_subjunctive_3pl",
    # Passive aorist optative: same -- ει contraction is ending-accented.
    "passive_aorist_optative_1sg",
    "passive_aorist_optative_2sg",
    "passive_aorist_optative_3sg",
    "passive_aorist_optative_1pl",
    "passive_aorist_optative_2pl",
    "passive_aorist_optative_3pl",
    "passive_aorist_optative_2du",
    "passive_aorist_optative_3du",
    # Passive aorist imperative 3sg / 3pl carry their own accent.
    "passive_aorist_imperative_3sg",
    "passive_aorist_imperative_3pl",
    # Passive aorist infinitive: ending-accented (-ῆναι).
    "passive_aorist_infinitive",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def synthesize_active_moods(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise active subjunctive / optative / imperative / aorist
    infinitive forms for a thematic -ω verb.

    Returns a dict ``{paradigm_key: form}`` with whatever cells the
    templating succeeded for. Returns an empty dict when the lemma is
    not a plain thematic -ω verb.

    The caller is responsible for merging this into the existing
    paradigm and *only* writing into empty slots; this function never
    decides whether a real cell already exists.

    Parameters
    ----------
    lemma : str
        The verb's citation form. Must be NFC.
    principal_parts : dict, optional
        Output of ``parse_principal_parts``. When None or empty, only
        the present-tense moods (which need only the lemma) are
        synthesised.
    """
    if not lemma or not is_thematic_omega(lemma):
        return {}
    parts = principal_parts or {}

    out: Dict[str, str] = {}

    # Cells whose ending carries an inherent acute (έτω / όντων / έτων /
    # άτω / άντων / οίτην). Stem accent is dropped before appending so we
    # don't produce double-accented forms like ``γράψάτω``.
    end_accented_keys = {
        "active_present_imperative_3sg",
        "active_present_imperative_3pl",
        "active_present_imperative_3du",
        "active_present_optative_3du",
        "active_aorist_imperative_3sg",
        "active_aorist_imperative_3pl",
    }

    def _emit(key: str, stem: str, ending: str) -> None:
        if key in end_accented_keys:
            out[key] = _strip_tonal_accents(stem) + ending
        else:
            out[key] = stem + ending

    # ---- Present-system moods (stem from lemma) ----
    pres_stem = _present_stem(lemma)
    if pres_stem:
        for (p, n), end in _PRES_SUBJ.items():
            _emit(f"active_present_subjunctive_{p}{n}", pres_stem, end)
        for (p, n), end in _PRES_OPT.items():
            _emit(f"active_present_optative_{p}{n}", pres_stem, end)
        for (p, n), end in _PRES_IMP.items():
            _emit(f"active_present_imperative_{p}{n}", pres_stem, end)
        # Present infinitive is also derivable but already covered by the
        # paradigm.py template fallback elsewhere; we add it for
        # completeness so a fresh paradigm without that fallback gets it.
        out["active_present_infinitive"] = pres_stem + "ειν"

    # ---- Aorist-system moods (sigmatic only) ----
    # Two paths to the sigmatic aorist stem:
    #   1. Future principal part: strip ω, validate σ/ψ/ξ ending.
    #      Carries the original accent without an augment.
    #   2. Aorist principal part + lemma: identify the σ/ψ/ξ cluster on
    #      the aor form, then graft it onto the lemma's present stem
    #      (which carries the accent). Skips when the present-final
    #      consonant doesn't predict the aorist cluster (suppletion,
    #      ablaut, irregular).
    aor_stem: Optional[str] = None
    if "fut" in parts:
        aor_stem = _aorist_stem_from_fut(parts["fut"])
    if aor_stem is None and "aor" in parts:
        aor_stem = _aorist_stem_from_lemma_and_aor(lemma, parts["aor"])
    # Final sanity: stem must look sigmatic.
    if aor_stem is not None:
        plain = _strip_accents_lower(aor_stem)
        if not plain or plain[-1] not in ("σ", "ψ", "ξ"):
            aor_stem = None

    if aor_stem:
        for (p, n), end in _AOR_SUBJ.items():
            _emit(f"active_aorist_subjunctive_{p}{n}", aor_stem, end)
        for (p, n), end in _AOR_OPT.items():
            _emit(f"active_aorist_optative_{p}{n}", aor_stem, end)
        for (p, n), end in _AOR_IMP.items():
            _emit(f"active_aorist_imperative_{p}{n}", aor_stem, end)
        out["active_aorist_infinitive"] = aor_stem + _AOR_INF

    return out


def _resolve_aor_stem(
    parts: Dict[str, str], lemma: str
) -> Optional[str]:
    """Resolve the sigmatic σ/ψ/ξ-stem from principal parts.

    Tries the future principal part first (carries the original accent
    without an augment), then derives from aor + lemma. Mirrors the
    in-line resolution at the top of ``synthesize_active_moods``.
    """
    aor_stem: Optional[str] = None
    if "fut" in parts:
        aor_stem = _aorist_stem_from_fut(parts["fut"])
    if aor_stem is None and "aor" in parts:
        aor_stem = _aorist_stem_from_lemma_and_aor(lemma, parts["aor"])
    if aor_stem is not None:
        plain = _strip_accents_lower(aor_stem)
        if not plain or plain[-1] not in ("σ", "ψ", "ξ"):
            return None
    return aor_stem


def synthesize_mp_moods(
    lemma: str,
    principal_parts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Synthesise middle / passive subj / opt / imp / inf forms for a
    thematic -ω verb.

    Returns a dict ``{paradigm_key: form}`` keyed in the jtauber-style
    flat shape:
      ``middle_present_subjunctive_<persnum>``
      ``middle_present_optative_<persnum>``
      ``middle_present_imperative_<persnum>``
      ``middle_present_infinitive``
      ``middle_aorist_subjunctive_<persnum>``
      ``middle_aorist_optative_<persnum>``
      ``middle_aorist_imperative_<persnum>``
      ``middle_aorist_infinitive``
      ``passive_aorist_subjunctive_<persnum>``
      ``passive_aorist_optative_<persnum>`` (incl. 2du / 3du)
      ``passive_aorist_imperative_<persnum>``
      ``passive_aorist_infinitive``

    Important schema notes (matched to jtauber):
      - Present middle and passive share form, so we emit only
        ``middle_present_*`` keys -- no ``passive_present_*``.
      - Aorist middle uses the σ/ψ/ξ-stem (sigmatic).
      - Aorist passive uses the θη-stem (athematic-style endings).
      - When the σ-stem is not derivable (aor-2, liquid future) the
        aorist-middle cells are simply skipped.
      - When the aor passive principal part is missing (no ``aor_p`` in
        the dict, e.g. γράφω which has 2nd-aor passive ἐγράφην that
        doesn't fit the θη pattern) the passive-aorist cells are
        skipped.

    Returns an empty dict when the lemma is not a plain thematic -ω
    verb. The caller is responsible for merging this into the existing
    paradigm and *only* writing into empty slots.
    """
    if not lemma or not is_thematic_omega(lemma):
        return {}
    parts = principal_parts or {}

    out: Dict[str, str] = {}

    def _emit_mp(key: str, stem: str, ending: str) -> None:
        if key in _MP_END_ACCENTED_KEYS:
            out[key] = _strip_tonal_accents(stem) + ending
        else:
            out[key] = stem + ending

    # ---- Present middle (= present mediopassive in jtauber) ----
    pres_stem = _present_stem(lemma)
    if pres_stem:
        for (p, n), end in _PRES_MID_SUBJ.items():
            _emit_mp(f"middle_present_subjunctive_{p}{n}", pres_stem, end)
        for (p, n), end in _PRES_MID_OPT.items():
            _emit_mp(f"middle_present_optative_{p}{n}", pres_stem, end)
        for (p, n), end in _PRES_MID_IMP.items():
            _emit_mp(f"middle_present_imperative_{p}{n}", pres_stem, end)
        out["middle_present_infinitive"] = pres_stem + _PRES_MID_INF

    # ---- Aorist middle (sigmatic σ-stem) ----
    aor_stem = _resolve_aor_stem(parts, lemma)
    if aor_stem:
        for (p, n), end in _AOR_MID_SUBJ.items():
            _emit_mp(f"middle_aorist_subjunctive_{p}{n}", aor_stem, end)
        for (p, n), end in _AOR_MID_OPT.items():
            _emit_mp(f"middle_aorist_optative_{p}{n}", aor_stem, end)
        for (p, n), end in _AOR_MID_IMP.items():
            _emit_mp(f"middle_aorist_imperative_{p}{n}", aor_stem, end)
        out["middle_aorist_infinitive"] = aor_stem + _AOR_MID_INF

    # ---- Aorist passive (θη-stem) ----
    pass_stem: Optional[str] = None
    if "aor_p" in parts:
        pass_stem = _aor_passive_stem(parts["aor_p"])
    if pass_stem:
        # All passive-aorist subjunctive / optative cells are
        # ending-accented (the θη-vowel contracts with the ending),
        # so the stem accent is dropped on the entire subj/opt grid.
        # The passive aor imperative needs special handling for the
        # 3pl form (-έντων), which drops the stem-final η before
        # splicing the participle-style ending.
        for (p, n), end in _AOR_PASS_SUBJ.items():
            _emit_mp(f"passive_aorist_subjunctive_{p}{n}", pass_stem, end)
        for (p, n), end in _AOR_PASS_OPT.items():
            _emit_mp(f"passive_aorist_optative_{p}{n}", pass_stem, end)
        for (p, n), end in _AOR_PASS_IMP.items():
            key = f"passive_aorist_imperative_{p}{n}"
            if (p, n) in _AOR_PASS_IMP_3PL_DROPS_ETA:
                # 3pl ``λυθέντων``: drop the stem accent AND splice the
                # ending after the bare θ (no η). That means the stem
                # we pass is already θ-only, not θη-... but our pass_stem
                # ends in θ already (we cut ``ην`` off), so we just need
                # to strip the stem accent and use the -έντων ending.
                out[key] = _strip_tonal_accents(pass_stem) + end
            else:
                _emit_mp(key, pass_stem, end)
        out["passive_aorist_infinitive"] = (
            _strip_tonal_accents(pass_stem) + _AOR_PASS_INF
        )

    return out


if __name__ == "__main__":
    # Smoke test for a few canonical verbs.
    samples = [
        ("λύω", {"fut": "λύσω", "aor": "ἔλυσα", "aor_p": "ἐλύθην"}),
        ("παιδεύω", {"aor": "ἐπαίδευσα", "aor_p": "ἐπαιδεύθην"}),
        ("γράφω", {"aor": "ἔγραψα", "fut_med": "γράψομαι"}),  # no aor_p
        ("πείθω", {"fut": "πείσω", "aor": "ἔπεισα", "aor_p": "ἐπείσθην"}),
        ("τιμάω", {"fut": "τιμήσω"}),  # contract; should be skipped
        ("τίθημι", {}),                # athematic; should be skipped
        ("μένω", {"fut": "μενῶ"}),     # liquid future; aorist skipped
    ]
    for lemma, parts in samples:
        out_act = synthesize_active_moods(lemma, parts)
        out_mp = synthesize_mp_moods(lemma, parts)
        print(f"=== {lemma} (parts={parts}) ===")
        print(f"  active: {len(out_act)} cells, mp: {len(out_mp)} cells")
        for k in sorted(out_mp):
            print(f"  {k} = {out_mp[k]}")
        print()
