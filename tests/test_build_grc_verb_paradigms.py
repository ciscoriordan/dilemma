"""Tests for ``build/build_grc_verb_paradigms.py``.

Focus: the augment-preference rule in :func:`pick_best_form` for past-
indicative cells (aorist / imperfect / pluperfect indicative). Multiple
corpora regularly attest both Homeric un-augmented forms (``λῦσε``,
``λῦσαν``) and Attic augmented forms (``ἔλυσε``, ``ἔλυσαν``) for the same
cell. Without the augment preference, the ``-len(f)`` tie-breaker silently
picks the shorter, un-augmented variant for the canonical Attic slice -
which is wrong for every classical / koine / modern reader expecting
augmented past indicatives.

Run with:

    python -m pytest tests/test_build_grc_verb_paradigms.py -x -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "build" / "build_grc_verb_paradigms.py"


@pytest.fixture(scope="module")
def b():
    """Load build/build_grc_verb_paradigms.py as a module without
    package install (mirrors the pattern in test_synth_verb_moods.py)."""
    spec = importlib.util.spec_from_file_location(
        "build_grc_verb_paradigms", MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# is_past_indicative_key
# ---------------------------------------------------------------------------


class TestIsPastIndicativeKey:
    """The augment-preference rule only fires on past-tense indicative
    cells. Non-indicative moods and non-past tenses must be left alone."""

    @pytest.mark.parametrize("key", [
        "active_aorist_indicative_1sg",
        "active_aorist_indicative_2sg",
        "active_aorist_indicative_3sg",
        "active_aorist_indicative_1pl",
        "active_aorist_indicative_2pl",
        "active_aorist_indicative_3pl",
        "middle_aorist_indicative_3sg",
        "passive_aorist_indicative_1pl",
        "active_imperfect_indicative_3sg",
        "active_imperfect_indicative_3pl",
        "middle_imperfect_indicative_2sg",
        "active_pluperfect_indicative_1sg",
        "middle_pluperfect_indicative_3pl",
    ])
    def test_past_indicative_keys_match(self, b, key):
        assert b.is_past_indicative_key(key)

    @pytest.mark.parametrize("key", [
        "active_present_indicative_3sg",
        "active_present_indicative_3pl",
        "middle_perfect_indicative_3sg",
        "active_future_indicative_1sg",
        "middle_future_indicative_3pl",
        "active_aorist_subjunctive_3sg",
        "active_aorist_optative_3pl",
        "active_aorist_imperative_2sg",
        "active_aorist_infinitive",
        "active_aorist_participle_nom_m_sg",
        "middle_imperfect_subjunctive_1sg",  # nonsense, but mood test
        "",
    ])
    def test_non_past_or_non_indicative_skipped(self, b, key):
        assert not b.is_past_indicative_key(key)


# ---------------------------------------------------------------------------
# has_augment
# ---------------------------------------------------------------------------


class TestHasAugment:
    """Augment detection wraps the dilemma.morph_diff helpers."""

    @pytest.mark.parametrize("form,lemma", [
        ("ἔλυσα", "λύω"),
        ("ἔλυσας", "λύω"),
        ("ἔλυσε", "λύω"),
        ("ἐλύσαμεν", "λύω"),
        ("ἐλύσατε", "λύω"),
        ("ἔλυσαν", "λύω"),
        ("ἔπαυσε", "παύω"),
        ("ἔπαυσαν", "παύω"),
        ("ἔγραψε", "γράφω"),
        ("ἔπεισε", "πείθω"),
        ("ἔλυε", "λύω"),
        ("ἔλυον", "λύω"),
        # Temporal augment for vowel-initial stems
        ("ἤκουσα", "ἀκούω"),
        ("ἤθελον", "ἐθέλω"),  # ε -> η
    ])
    def test_augmented_forms_detected(self, b, form, lemma):
        assert b.has_augment(form, lemma), f"expected augment in {form} <- {lemma}"

    @pytest.mark.parametrize("form,lemma", [
        # Un-augmented past-indicative variants (Homeric / poetic)
        ("λῦσε", "λύω"),
        ("λῦσαν", "λύω"),
        ("λῦε", "λύω"),
        ("λύον", "λύω"),
        ("παῦσε", "παύω"),
        ("παῦσαν", "παύω"),
        # Present forms (no augment by definition)
        ("λύω", "λύω"),
        ("λύει", "λύω"),
        ("ἀκούω", "ἀκούω"),
        ("ἐθέλω", "ἐθέλω"),
    ])
    def test_unaugmented_forms_detected(self, b, form, lemma):
        assert not b.has_augment(form, lemma), \
            f"unexpected augment in {form} <- {lemma}"

    def test_empty_inputs_safe(self, b):
        assert not b.has_augment("", "λύω")
        assert not b.has_augment("ἔλυσε", "")
        assert not b.has_augment("", "")


# ---------------------------------------------------------------------------
# pick_best_form: augment preference for past-indicative cells
# ---------------------------------------------------------------------------


class TestPickBestFormAugmentPreference:
    """The original bug: glaux corpus emits both ``λῦσε`` and ``ἔλυσε``
    once each for λύω aor-act-ind 3sg, and the un-augmented ``λῦσε``
    won via the ``-len(f)`` tie-breaker. The fix: when ``key`` names a
    past-indicative cell and ``lemma`` is provided, augment-bearing
    variants outrank un-augmented variants regardless of length."""

    # --- λύω aor-act-ind: full bug regression suite ------------------------

    def test_lyo_aor_act_ind_1sg(self, b):
        forms = ["ἔλυσα", "ἔλυσά"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_1sg", lemma="λύω")
        assert got == "ἔλυσα"

    def test_lyo_aor_act_ind_2sg(self, b):
        forms = ["ἔλυσας", "ἔλυσάς"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_2sg", lemma="λύω")
        assert got in {"ἔλυσας", "ἔλυσάς"}

    def test_lyo_aor_act_ind_3sg_augmented_wins(self, b):
        # The reported bug: un-augmented λῦσε used to win here.
        forms = ["ἔλυσ’", "λῦσεν", "ἔλυσεν", "λῦσε", "ἔλυσε",
                 "ἔλυσέ", "ἔλυσέν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="λύω")
        assert b.has_augment(got, "λύω"), \
            f"3sg pick {got} lacks augment"
        # Specifically must not be λῦσε / λῦσεν.
        assert got not in {"λῦσε", "λῦσεν"}

    def test_lyo_aor_act_ind_3sg_no_augment_in_pool_ok(self, b):
        # If only un-augmented variants are attested, we still pick
        # something rather than returning None.
        forms = ["λῦσε", "λῦσεν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="λύω")
        assert got in {"λῦσε", "λῦσεν"}

    def test_lyo_aor_act_ind_1pl(self, b):
        forms = ["ἐλύσαμεν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_1pl", lemma="λύω")
        assert got == "ἐλύσαμεν"

    def test_lyo_aor_act_ind_2pl(self, b):
        forms = ["ἐλύσατε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_2pl", lemma="λύω")
        assert got == "ἐλύσατε"

    def test_lyo_aor_act_ind_3pl_augmented_wins(self, b):
        # The reported bug: λῦσαν used to win here.
        forms = ["ἔλυσαν", "λῦσαν", "ἔλυσάν", "λύσαν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3pl", lemma="λύω")
        assert b.has_augment(got, "λύω"), \
            f"3pl pick {got} lacks augment"
        assert got not in {"λῦσαν", "λύσαν"}

    # --- λύω aor-mid-ind: should already be augmented ---------------------

    def test_lyo_aor_mid_ind_3sg(self, b):
        forms = ["ἐλύσατο", "λύσατο"]
        got = b.pick_best_form(
            forms, key="middle_aorist_indicative_3sg", lemma="λύω")
        assert b.has_augment(got, "λύω")

    def test_lyo_aor_mid_ind_3pl(self, b):
        forms = ["ἐλύσαντο", "λύσαντο"]
        got = b.pick_best_form(
            forms, key="middle_aorist_indicative_3pl", lemma="λύω")
        assert b.has_augment(got, "λύω")

    # --- λύω imperfect-ind: same bug class --------------------------------

    def test_lyo_imperfect_act_ind_3sg(self, b):
        # λῦε / λύε / λύεν all un-augmented; ἔλυεν / ἔλυέν are right.
        forms = ["λύεσκε", "λῦε", "ἔλυεν", "λύε", "ἔλυέν", "λύεν"]
        got = b.pick_best_form(
            forms, key="active_imperfect_indicative_3sg", lemma="λύω")
        assert b.has_augment(got, "λύω")
        assert got not in {"λῦε", "λύε", "λύεν"}

    def test_lyo_imperfect_act_ind_3pl(self, b):
        forms = ["λύον", "ἔλυον"]
        got = b.pick_best_form(
            forms, key="active_imperfect_indicative_3pl", lemma="λύω")
        assert got == "ἔλυον"

    # --- παύω aor-act-ind: same bug class ---------------------------------

    def test_pauo_aor_act_ind_3sg_augmented_wins(self, b):
        forms = ["παῦσε", "ἔπαυσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="παύω")
        assert got == "ἔπαυσε"

    def test_pauo_aor_act_ind_3pl_augmented_wins(self, b):
        forms = ["παῦσαν", "ἔπαυσαν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3pl", lemma="παύω")
        assert got == "ἔπαυσαν"

    # --- πείθω aor-act-ind: bug-adjacent (3sg sometimes correct) ----------

    def test_peitho_aor_act_ind_3sg_augmented_wins(self, b):
        forms = ["πεῖσε", "ἔπεισε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="πείθω")
        assert got == "ἔπεισε"

    # --- γράφω aor-act-ind: bug-adjacent ---------------------------------

    def test_grapho_aor_act_ind_3sg_augmented_wins(self, b):
        forms = ["γράψε", "ἔγραψε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="γράφω")
        assert got == "ἔγραψε"

    def test_grapho_aor_act_ind_3pl_augmented_wins(self, b):
        forms = ["γράψαν", "ἔγραψαν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3pl", lemma="γράφω")
        assert got == "ἔγραψαν"

    # --- λούω aor-act-ind: same bug class ---------------------------------

    def test_louo_aor_act_ind_3sg_augmented_wins(self, b):
        forms = ["λοῦσε", "ἔλουσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="λούω")
        assert got == "ἔλουσε"

    def test_louo_aor_act_ind_3pl_augmented_wins(self, b):
        forms = ["λοῦσαν", "ἔλουσαν"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3pl", lemma="λούω")
        assert got == "ἔλουσαν"

    # --- Vowel-initial stem: temporal augment ----------------------------

    def test_akouo_aor_act_ind_3sg_temporal_augment(self, b):
        # ἀκούω -> ἤκουσε (temporal augment α -> η).
        forms = ["ἄκουσε", "ἤκουσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="ἀκούω")
        assert got == "ἤκουσε"


# ---------------------------------------------------------------------------
# pick_best_form: behaviour outside past-indicative is unchanged
# ---------------------------------------------------------------------------


class TestPickBestFormPreservesNonPast:
    """The augment preference must NOT activate for non-past or non-
    indicative cells. Subjunctive / optative / imperative / infinitive /
    participle cells never carry augment, so we keep the original
    polytonic / length / alphabetical tie-breaker chain."""

    def test_present_indicative_unchanged(self, b):
        # No augment-preference: shortest variant still wins on tie.
        forms = ["λύει", "λύεις"]
        got = b.pick_best_form(
            forms, key="active_present_indicative_3sg", lemma="λύω")
        # Before the fix this would pick by polytonic / length /
        # alphabetical. After the fix we did not change this branch.
        assert got in forms

    def test_aorist_subjunctive_unchanged(self, b):
        # Subjunctive never has augment.
        forms = ["λύσῃ", "λύσῃς"]
        got = b.pick_best_form(
            forms, key="active_aorist_subjunctive_3sg", lemma="λύω")
        assert got in forms

    def test_aorist_imperative_unchanged(self, b):
        forms = ["λῦσον"]
        got = b.pick_best_form(
            forms, key="active_aorist_imperative_2sg", lemma="λύω")
        assert got == "λῦσον"

    def test_aorist_infinitive_unchanged(self, b):
        forms = ["λῦσαι"]
        got = b.pick_best_form(
            forms, key="active_aorist_infinitive", lemma="λύω")
        assert got == "λῦσαι"

    def test_aorist_participle_unchanged(self, b):
        forms = ["λύσας"]
        got = b.pick_best_form(
            forms, key="active_aorist_participle_nom_m_sg", lemma="λύω")
        assert got == "λύσας"

    def test_no_lemma_falls_back_to_length(self, b):
        # When the caller doesn't pass a lemma (e.g. dialect slice),
        # the augment-preference rule is disabled and we fall back to
        # the original `-len(f)` tie-breaker. λῦσε wins over ἔλυσε on
        # length alone.
        forms = ["λῦσε", "ἔλυσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg")
        assert got == "λῦσε"

    def test_no_key_falls_back_to_length(self, b):
        forms = ["λῦσε", "ἔλυσε"]
        got = b.pick_best_form(forms)
        assert got == "λῦσε"

    def test_count_still_dominates(self, b):
        # If one variant is attested twice and the other once, the
        # higher-count variant wins regardless of augment status. Real
        # corpus quality > our augment heuristic.
        forms = ["λῦσε", "λῦσε", "ἔλυσε"]
        got = b.pick_best_form(
            forms, key="active_aorist_indicative_3sg", lemma="λύω")
        assert got == "λῦσε"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """The old call signature ``pick_best_form(forms)`` must still work
    so any in-repo / out-of-repo callers that haven't migrated still
    behave identically to the pre-fix code path."""

    def test_no_kwargs_old_signature(self, b):
        forms = ["λύει", "λύεις"]
        got = b.pick_best_form(forms)
        assert got in forms

    def test_set_input(self, b):
        forms = {"ἔλυσα", "ἔλυσά"}
        got = b.pick_best_form(forms, key="active_aorist_indicative_1sg",
                               lemma="λύω")
        assert got in forms

    def test_empty_returns_none(self, b):
        assert b.pick_best_form([]) is None
        assert b.pick_best_form(set()) is None
