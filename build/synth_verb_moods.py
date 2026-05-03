#!/usr/bin/env python3
"""Procedural synthesis of finite verb mood forms from principal parts.

Given an Ancient Greek verb lemma and the principal-parts dict produced
by ``build/lsj_principal_parts.parse_principal_parts``, this module
generates the missing finite-mood paradigm cells (subjunctive, optative,
imperative, aorist infinitive) by stem-templating with regular endings.

Scope (v1):
  - Active voice only.
  - Plain thematic -ω verbs only. Contracts (-άω/-έω/-όω), athematic
    (-μι/-μαι), and lemmas not ending in ω are skipped.
  - Tense / mood combinations covered:
      * present subjunctive   (stem from lemma)
      * present optative      (stem from lemma)
      * present imperative    (stem from lemma; 2sg, 3sg, 2pl, 3pl, duals)
      * aorist subjunctive    (stem from fut. or aor.)
      * aorist optative       (stem from fut. or aor.)
      * aorist imperative     (stem from fut. or aor.; sigmatic only)
      * aorist infinitive     (stem from fut. or aor.)
  - The function only produces forms; the caller decides which slots
    to write (and whether to overwrite or only fill empty cells).
  - Accent: the synthesis preserves the lemma / fut-stem accent on stem
    syllables, but does NOT compute fresh accent placement on
    enclitic-final cells (3sg/3pl imperative). Those endings carry an
    inherent acute, but the stem-accent in the synthesised form remains
    where the input has it. Result: 3sg/3pl imperatives may be
    accent-imperfect on multi-syllable stems, but the segmentation is
    always correct so the caller's training pipeline still benefits.

What this module does NOT do (deferred to v2+):
  - Middle / passive voice. The mp-stem requires reverse-assimilation
    (handled in lsj_principal_parts.derive_grc_conj_args but not yet
    plumbed through here).
  - Aor-2 (thematic ``ἔλιπον``-style) imperative: ending pattern is
    different (2sg ``λίπε`` vs sigmatic ``λῦσον``).
  - Contract verbs: contract-future / contract-aorist forms have
    distinct vowel-stem rules.
  - Perfect-system synthesis (perfect subj/opt is rare and almost
    always periphrastic).
  - Participles: full case×gender×number declension is its own module.

Pure-Python; no I/O, no DB, no external deps. Mirrors the shape of
``dilemma.morph_diff`` and ``build.lsj_principal_parts``.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, Optional


__all__ = [
    "synthesize_active_moods",
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


if __name__ == "__main__":
    # Smoke test for a few canonical verbs.
    samples = [
        ("λύω", {"fut": "λύσω", "aor": "ἔλυσα"}),
        ("παιδεύω", {"aor": "ἐπαίδευσα"}),
        ("γράφω", {"aor": "ἔγραψα", "fut_med": "γράψομαι"}),
        ("τιμάω", {"fut": "τιμήσω"}),  # contract; should be skipped
        ("τίθημι", {}),                # athematic; should be skipped
        ("μένω", {"fut": "μενῶ"}),     # liquid future; aorist skipped
    ]
    for lemma, parts in samples:
        out = synthesize_active_moods(lemma, parts)
        print(f"=== {lemma} (parts={parts}) ===")
        for k in sorted(out):
            print(f"  {k} = {out[k]}")
        print()
