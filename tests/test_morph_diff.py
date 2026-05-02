#!/usr/bin/env python3
"""Tests for the morph_diff annotator.

Run with:
    python -m pytest tests/test_morph_diff.py -x -v
"""

import unicodedata

import pytest

from dilemma.morph_diff import diff_form, diff_paradigm, MorphDiff, Role


# ---------------------------------------------------------------------------
# Small helpers for asserting against a roles array.
# ---------------------------------------------------------------------------


def role_at(diff: MorphDiff, ch: str) -> list[Role]:
    """Return the roles assigned to every NFC code point of diff.form whose
    diacritic-stripped lower-case base is `ch`."""
    nfd_target = unicodedata.normalize("NFD", ch)
    target_base = "".join(c for c in nfd_target
                          if not unicodedata.combining(c)).lower()
    out = []
    for i, c in enumerate(diff.form):
        nfd = unicodedata.normalize("NFD", c)
        base = "".join(x for x in nfd if not unicodedata.combining(x)).lower()
        if base == target_base:
            out.append(diff.roles[i])
    return out


def has_role(diff: MorphDiff, role: Role) -> bool:
    return any(r == role for r in diff.roles)


def count_role(diff: MorphDiff, role: Role) -> int:
    return sum(1 for r in diff.roles if r == role)


# ---------------------------------------------------------------------------
# Ancient Greek tests
# ---------------------------------------------------------------------------


class TestAncientGreek:
    """Verify expected role assignments for canonical AG paradigm forms."""

    def test_grafw_aorist_augment_and_irregular_stem(self):
        """γράφω → ἔγραψα: ε- augment + φ -> ψ stem allomorphy."""
        d = diff_form("γράφω", "ἔγραψα", lang="grc")
        assert d.form == "ἔγραψα"
        assert len(d.roles) == 6
        assert d.roles[0] == Role.AUGMENT
        # γρα should be STEM
        assert d.roles[1] == Role.STEM
        assert d.roles[2] == Role.STEM
        assert d.roles[3] == Role.STEM
        # ψ is the labial+s coalescence of φ+σ - irregular relative to lemma stem γραφ-
        assert d.roles[4] == Role.IRREGULAR_STEM
        # α is the ending
        assert d.roles[5] == Role.ENDING
        assert d.stem_change is True
        assert 0 in d.irregular_indices
        assert 4 in d.irregular_indices
        # ENDING is never in irregular_indices
        assert 5 not in d.irregular_indices
        assert d.ending_start == 5

    def test_grafw_perfect_reduplication(self):
        """γράφω → γέγραφα: γε- reduplication."""
        d = diff_form("γράφω", "γέγραφα", lang="grc")
        assert d.form == "γέγραφα"
        assert len(d.roles) == 7
        # γε is reduplication
        assert d.roles[0] == Role.REDUPLICATION
        assert d.roles[1] == Role.REDUPLICATION
        # γραφ should be STEM
        for i in (2, 3, 4, 5):
            assert d.roles[i] == Role.STEM, f"index {i} expected STEM, got {d.roles[i]}"
        # α is ending
        assert d.roles[6] == Role.ENDING
        assert d.stem_change is True
        # Reduplication is "irregular" for highlighting purposes.
        assert 0 in d.irregular_indices
        assert 1 in d.irregular_indices

    def test_ferw_present_imperative_no_false_reduplication(self):
        """φέρω → φέρε: present-tense form starts with `Cε` but isn't
        reduplication — the rest of the form ('ρε') doesn't recover the
        lemma stem 'φερ'. Regression: an earlier reduplication detector
        only checked the leading `Cε` pair and falsely flagged every
        consonant-initial `Cε…` form as reduplicated."""
        d = diff_form("φέρω", "φέρε", lang="grc")
        assert d.irregular_indices == []
        assert d.roles[0] == Role.STEM
        assert d.roles[1] == Role.STEM

    def test_lew_mg_present_no_false_reduplication(self):
        """λέω → λέμε (MG): regular present 1pl. Lemma λέω has stem
        'λε', form λέμε also starts with 'λε' followed by 'με' which
        is the ending. No reduplication, no irregular flagging."""
        d = diff_form("λέω", "λέμε", lang="el")
        assert d.irregular_indices == []

    def test_luw_present_no_stem_change(self):
        """λύω → λύεις: regular present indicative, no stem change."""
        d = diff_form("λύω", "λύεις", lang="grc")
        assert d.form == "λύεις"
        # λύ is the stem (2 chars), -εις is the ending.
        assert d.roles[0] == Role.STEM
        assert d.roles[1] == Role.STEM
        for i in range(2, 5):
            assert d.roles[i] == Role.ENDING, f"index {i} expected ENDING, got {d.roles[i]}"
        assert d.stem_change is False
        assert d.irregular_indices == []
        assert d.ending_start == 2

    def test_luw_imperfect_augment(self):
        """λύω → ἔλυον: ε- augment, otherwise regular."""
        d = diff_form("λύω", "ἔλυον", lang="grc")
        assert d.form == "ἔλυον"
        assert len(d.roles) == 5
        assert d.roles[0] == Role.AUGMENT
        # λυ stem
        assert d.roles[1] == Role.STEM
        assert d.roles[2] == Role.STEM
        # ον ending
        assert d.roles[3] == Role.ENDING
        assert d.roles[4] == Role.ENDING
        assert d.stem_change is True
        # The only irregular bit is the augment.
        assert d.irregular_indices == [0]

    def test_erkhomai_supplete_aorist(self):
        """ἔρχομαι → ἦλθον: highly irregular suppletion (root ἐλθ-)."""
        d = diff_form("ἔρχομαι", "ἦλθον", lang="grc")
        assert d.form == "ἦλθον"
        assert d.stem_change is True
        # Most of the form should be IRREGULAR_STEM since there's almost
        # no overlap with the present stem ἐρχ-.
        irregular_count = count_role(d, Role.IRREGULAR_STEM)
        assert irregular_count >= 2, (
            f"expected at least 2 IRREGULAR_STEM chars, got roles={d.roles}"
        )

    def test_ferw_supplete_oisw(self):
        """φέρω → οἴσω: future suppletion (root οἰσ-, totally different)."""
        d = diff_form("φέρω", "οἴσω", lang="grc")
        assert d.form == "οἴσω"
        assert d.stem_change is True
        # No characters of the form match the stem φερ-, so most should
        # be IRREGULAR_STEM (with possibly the trailing -ω as ENDING).
        irregular_count = count_role(d, Role.IRREGULAR_STEM)
        assert irregular_count >= 2

    def test_ferw_supplete_enegka(self):
        """φέρω → ἤνεγκα: another suppletion (aorist root ἐνεγκ-)."""
        d = diff_form("φέρω", "ἤνεγκα", lang="grc")
        assert d.form == "ἤνεγκα"
        assert d.stem_change is True
        irregular_count = count_role(d, Role.IRREGULAR_STEM)
        # The leading η could be flagged as AUGMENT (temporal) which is
        # also "irregular"; either way at least 1 char should be irregular.
        assert (irregular_count + count_role(d, Role.AUGMENT)) >= 2


# ---------------------------------------------------------------------------
# Modern Greek tests
# ---------------------------------------------------------------------------


class TestModernGreek:
    """Modern Greek augment, stem allomorphy, suppletion."""

    def test_grafw_mg_augment(self):
        """γράφω → έγραψα: ε- augment in MG aorist."""
        d = diff_form("γράφω", "έγραψα", lang="el")
        assert d.form == "έγραψα"
        assert d.roles[0] == Role.AUGMENT
        assert d.stem_change is True
        # The φ -> ψ allomorphy should also be flagged.
        assert count_role(d, Role.IRREGULAR_STEM) >= 1

    def test_agapaw_stem_allomorphy(self):
        """αγαπάω → αγάπησε: stem ends in -ησ in past, lemma in -α."""
        d = diff_form("αγαπάω", "αγάπησε", lang="el")
        assert d.form == "αγάπησε"
        # The lemma stem (αγαπ-) has its three letters present, but the
        # extra -ησ- before the ending is irregular.
        assert d.stem_change is True
        assert count_role(d, Role.IRREGULAR_STEM) >= 1

    def test_lew_supplete(self):
        """λέω → είπα: classic Greek suppletion (different roots)."""
        d = diff_form("λέω", "είπα", lang="el")
        assert d.form == "είπα"
        assert d.stem_change is True

    def test_pairnw_stem_allomorphy(self):
        """παίρνω → πήρα: stem allomorphy (παιρν- -> πηρ-)."""
        d = diff_form("παίρνω", "πήρα", lang="el")
        assert d.form == "πήρα"
        assert d.stem_change is True
        # At least one character should be flagged as a stem change since
        # the form doesn't share much surface material with παιρν-.
        irregular_count = (
            count_role(d, Role.IRREGULAR_STEM)
            + count_role(d, Role.AUGMENT)
        )
        assert irregular_count >= 1


# ---------------------------------------------------------------------------
# Negative / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_form(self):
        d = diff_form("γράφω", "", lang="grc")
        assert d.form == ""
        assert d.roles == []
        assert d.irregular_indices == []
        assert d.stem_change is False
        assert d.ending_start is None

    def test_empty_lemma(self):
        d = diff_form("", "λύεις", lang="grc")
        assert d.form == "λύεις"
        # Without a lemma to compare against, every char is just STEM.
        assert all(r == Role.STEM for r in d.roles)
        assert d.stem_change is False
        assert d.irregular_indices == []

    def test_both_empty(self):
        d = diff_form("", "", lang="grc")
        assert d.form == ""
        assert d.roles == []
        assert d.stem_change is False
        assert d.ending_start is None

    def test_form_equals_lemma(self):
        """A form identical to the lemma is UNCHANGED everywhere."""
        d = diff_form("λύω", "λύω", lang="grc")
        assert d.form == "λύω"
        assert all(r == Role.UNCHANGED for r in d.roles)
        assert d.stem_change is False

    def test_none_inputs(self):
        """None lemma/form should be handled gracefully (treated as ''')."""
        d = diff_form(None, None, lang="grc")  # type: ignore[arg-type]
        assert d.form == ""
        assert d.roles == []

    def test_nfc_length_invariant(self):
        """roles must match the NFC code-point length of form."""
        for lemma, form in [
            ("γράφω", "ἔγραψα"),
            ("γράφω", "γέγραφα"),
            ("λύω", "λύεις"),
            ("ἔρχομαι", "ἦλθον"),
            ("φέρω", "οἴσω"),
        ]:
            d = diff_form(lemma, form, lang="grc")
            nfc_form = unicodedata.normalize("NFC", form)
            assert len(d.roles) == len(nfc_form)
            assert d.form == nfc_form

    def test_irregular_indices_excludes_ending(self):
        """ENDING characters must never appear in irregular_indices."""
        for lemma, form, lang in [
            ("γράφω", "ἔγραψα", "grc"),
            ("γράφω", "γέγραφα", "grc"),
            ("λύω", "ἔλυον", "grc"),
            ("γράφω", "έγραψα", "el"),
        ]:
            d = diff_form(lemma, form, lang=lang)
            for idx in d.irregular_indices:
                assert d.roles[idx] != Role.ENDING


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


class TestParadigm:
    def test_diff_paradigm_basic(self):
        forms = {
            "1sg.pres":    "λύω",
            "2sg.pres":    "λύεις",
            "1sg.impf":    "ἔλυον",
            "1sg.aor.act": "ἔλυσα",
        }
        out = diff_paradigm("λύω", forms, lang="grc")
        assert set(out.keys()) == set(forms.keys())
        # The present-tense 1sg has no stem change.
        assert out["1sg.pres"].stem_change is False
        # Imperfect has an augment.
        assert out["1sg.impf"].roles[0] == Role.AUGMENT
        # Aorist has both an augment and a sigmatic ending.
        assert out["1sg.aor.act"].roles[0] == Role.AUGMENT
