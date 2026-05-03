"""Tests for ``build/synth_verb_moods.py`` and its integration in
``build/build_grc_verb_paradigms.py``.

The synth module is loaded directly as a script-style module via
``importlib`` (the ``build/`` directory is not an installable package),
mirroring the pattern in ``test_athematic_verbs.py``.

Run with:

    python -m pytest tests/test_synth_verb_moods.py -x -v
"""

from __future__ import annotations

import importlib.util
import unicodedata
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_PATH = REPO_ROOT / "build" / "synth_verb_moods.py"


@pytest.fixture(scope="module")
def synth():
    """Load build/synth_verb_moods.py as a module without package install."""
    spec = importlib.util.spec_from_file_location(
        "synth_verb_moods", SYNTH_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Lemma classification
# ---------------------------------------------------------------------------


class TestIsThematicOmega:
    def test_plain_omega_verb(self, synth):
        assert synth.is_thematic_omega("λύω")
        assert synth.is_thematic_omega("γράφω")
        assert synth.is_thematic_omega("παύω")
        assert synth.is_thematic_omega("παιδεύω")

    def test_alpha_contract_rejected(self, synth):
        assert not synth.is_thematic_omega("τιμάω")

    def test_epsilon_contract_rejected(self, synth):
        assert not synth.is_thematic_omega("φιλέω")
        assert not synth.is_thematic_omega("ποιέω")

    def test_omicron_contract_rejected(self, synth):
        assert not synth.is_thematic_omega("δηλόω")

    def test_athematic_mi_rejected(self, synth):
        assert not synth.is_thematic_omega("τίθημι")
        assert not synth.is_thematic_omega("δίδωμι")
        assert not synth.is_thematic_omega("εἰμί")

    def test_deponent_mai_rejected(self, synth):
        assert not synth.is_thematic_omega("ἔρχομαι")
        assert not synth.is_thematic_omega("γίγνομαι")

    def test_empty_or_garbage_rejected(self, synth):
        assert not synth.is_thematic_omega("")
        assert not synth.is_thematic_omega(None)
        # No -ω lemma
        assert not synth.is_thematic_omega("ἀνήρ")


# ---------------------------------------------------------------------------
# Present-system synthesis (uses lemma stem only, no principal_parts needed)
# ---------------------------------------------------------------------------


class TestPresentSynthesis:
    """Present subj/opt/imp synthesis works off the lemma alone -- no
    principal_parts data required."""

    def test_lyo_present_subjunctive(self, synth):
        out = synth.synthesize_active_moods("λύω", {})
        assert out["active_present_subjunctive_1sg"] == "λύω"
        assert out["active_present_subjunctive_2sg"] == "λύῃς"
        assert out["active_present_subjunctive_3sg"] == "λύῃ"
        assert out["active_present_subjunctive_1pl"] == "λύωμεν"
        assert out["active_present_subjunctive_2pl"] == "λύητε"
        assert out["active_present_subjunctive_3pl"] == "λύωσι(ν)"

    def test_lyo_present_optative(self, synth):
        out = synth.synthesize_active_moods("λύω", {})
        assert out["active_present_optative_1sg"] == "λύοιμι"
        assert out["active_present_optative_2sg"] == "λύοις"
        assert out["active_present_optative_3sg"] == "λύοι"
        assert out["active_present_optative_1pl"] == "λύοιμεν"
        assert out["active_present_optative_2pl"] == "λύοιτε"
        assert out["active_present_optative_3pl"] == "λύοιεν"

    def test_lyo_present_imperative(self, synth):
        out = synth.synthesize_active_moods("λύω", {})
        assert out["active_present_imperative_2sg"] == "λύε"
        assert out["active_present_imperative_2pl"] == "λύετε"
        # 3sg/3pl have ending-stress and lose the lemma's stem accent.
        assert out["active_present_imperative_3sg"] == "λυέτω"
        assert out["active_present_imperative_3pl"] == "λυόντων"

    def test_lyo_present_infinitive(self, synth):
        out = synth.synthesize_active_moods("λύω", {})
        assert out["active_present_infinitive"] == "λύειν"

    def test_grapho_keeps_stem_accent(self, synth):
        """γράφω's stem accent on γρά- should ride through to the
        synthesised forms (γράφω -> γράφω-subj, γράφοιμι-opt)."""
        out = synth.synthesize_active_moods("γράφω", {})
        assert out["active_present_subjunctive_1sg"] == "γράφω"
        assert out["active_present_optative_1sg"] == "γράφοιμι"
        # 3sg imperative drops stem accent (ending carries it).
        assert out["active_present_imperative_3sg"] == "γραφέτω"

    def test_present_only_when_no_pp_provided(self, synth):
        """Without any principal_parts, only present-system cells are
        produced -- no aorist."""
        out = synth.synthesize_active_moods("λύω", None)
        assert "active_present_subjunctive_1sg" in out
        assert "active_aorist_subjunctive_1sg" not in out


# ---------------------------------------------------------------------------
# Aorist synthesis from principal parts
# ---------------------------------------------------------------------------


class TestAoristSynthesisFromFut:
    """When the future principal part is available, the aorist stem
    derives from it (preserves accent without an augment)."""

    def test_lyo_with_fut(self, synth):
        out = synth.synthesize_active_moods("λύω", {"fut": "λύσω"})
        assert out["active_aorist_subjunctive_1sg"] == "λύσω"
        assert out["active_aorist_subjunctive_2sg"] == "λύσῃς"
        assert out["active_aorist_subjunctive_3pl"] == "λύσωσι(ν)"
        assert out["active_aorist_optative_1sg"] == "λύσαιμι"
        assert out["active_aorist_optative_2sg"] == "λύσαις"
        assert out["active_aorist_optative_3pl"] == "λύσαιεν"
        assert out["active_aorist_imperative_2sg"] == "λύσον"
        assert out["active_aorist_imperative_2pl"] == "λύσατε"
        # 3sg/3pl drop stem accent (ending is accented).
        assert out["active_aorist_imperative_3sg"] == "λυσάτω"
        assert out["active_aorist_imperative_3pl"] == "λυσάντων"
        assert out["active_aorist_infinitive"] == "λύσαι"

    def test_paideuo_with_fut(self, synth):
        out = synth.synthesize_active_moods(
            "παιδεύω", {"fut": "παιδεύσω"}
        )
        assert out["active_aorist_subjunctive_1sg"] == "παιδεύσω"
        assert out["active_aorist_optative_1sg"] == "παιδεύσαιμι"
        assert out["active_aorist_imperative_2sg"] == "παιδεύσον"
        assert out["active_aorist_imperative_3sg"] == "παιδευσάτω"

    def test_pempo_with_fut_psi_cluster(self, synth):
        """π-stem verb πέμπω -> πέμψω: aorist endings hang off πέμψ-."""
        out = synth.synthesize_active_moods("πέμπω", {"fut": "πέμψω"})
        assert out["active_aorist_subjunctive_1sg"] == "πέμψω"
        assert out["active_aorist_optative_1sg"] == "πέμψαιμι"
        assert out["active_aorist_imperative_2sg"] == "πέμψον"
        assert out["active_aorist_imperative_3sg"] == "πεμψάτω"


class TestAoristSynthesisFromAor:
    """When only the aorist (not the future) is given, we splice the
    aor's terminal cluster onto the lemma's present stem to keep the
    accent."""

    def test_grapho_with_aor_only(self, synth):
        """γράφω + aor ἔγραψα: present stem γράφ- -> swap φ for ψ -> γράψ-."""
        out = synth.synthesize_active_moods("γράφω", {"aor": "ἔγραψα"})
        assert out["active_aorist_subjunctive_1sg"] == "γράψω"
        assert out["active_aorist_optative_1sg"] == "γράψαιμι"
        assert out["active_aorist_imperative_2sg"] == "γράψον"
        assert out["active_aorist_imperative_3sg"] == "γραψάτω"
        assert out["active_aorist_infinitive"] == "γράψαι"

    def test_paideuo_with_aor_only(self, synth):
        out = synth.synthesize_active_moods(
            "παιδεύω", {"aor": "ἐπαίδευσα"}
        )
        assert out["active_aorist_subjunctive_1sg"] == "παιδεύσω"
        assert out["active_aorist_optative_1sg"] == "παιδεύσαιμι"
        assert out["active_aorist_imperative_2sg"] == "παιδεύσον"


class TestAoristSkippedForIrregular:
    """Verbs whose aorist is not derivable from the present stem +
    sigma graft should NOT get any synthesised aorist forms (the synth
    is conservative; better to leave a cell empty than fill it wrong)."""

    def test_pipto_aor_2_skipped(self, synth):
        """πίπτω has thematic aor-2 ἔπεσον (no σ); synthesis must skip."""
        out = synth.synthesize_active_moods(
            "πίπτω", {"aor": "ἔπεσον"}
        )
        # No aorist active forms produced
        assert "active_aorist_subjunctive_1sg" not in out
        assert "active_aorist_imperative_2sg" not in out
        # Present forms still synthesised from the lemma
        assert "active_present_subjunctive_1sg" in out

    def test_lipo_aor_2_skipped(self, synth):
        """λείπω has aor-2 ἔλιπον (root vowel ablaut, not sigmatic)."""
        out = synth.synthesize_active_moods(
            "λείπω", {"aor": "ἔλιπον"}
        )
        assert "active_aorist_subjunctive_1sg" not in out
        assert "active_aorist_imperative_2sg" not in out
        assert "active_present_subjunctive_1sg" in out

    def test_meno_liquid_future_skipped(self, synth):
        """μένω -> contract future μενῶ (no σ); no sigmatic stem."""
        out = synth.synthesize_active_moods("μένω", {"fut": "μενῶ"})
        # Aorist branch should not fire from the future alone.
        assert "active_aorist_subjunctive_1sg" not in out
        # Present moods still come through.
        assert out["active_present_subjunctive_1sg"] == "μένω"


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


class TestNoOpCases:
    def test_contract_verb_returns_empty(self, synth):
        """ε-contract φιλέω is rejected at the classifier; nothing
        synthesised."""
        out = synth.synthesize_active_moods(
            "φιλέω", {"aor": "ἐφίλησα", "fut": "φιλήσω"}
        )
        assert out == {}

    def test_athematic_returns_empty(self, synth):
        """μι-verb δίδωμι: no synthesis (athematic)."""
        out = synth.synthesize_active_moods(
            "δίδωμι", {"aor": "ἔδωκα"}
        )
        assert out == {}

    def test_missing_principal_parts_no_error(self, synth):
        """No principal_parts dict (None) must not error: only present
        moods get synthesised."""
        out = synth.synthesize_active_moods("λύω", None)
        assert isinstance(out, dict)
        assert "active_present_subjunctive_1sg" in out
        assert "active_aorist_subjunctive_1sg" not in out

    def test_empty_principal_parts_no_error(self, synth):
        """Empty dict is also fine (caller had a head text but parser
        returned nothing)."""
        out = synth.synthesize_active_moods("λύω", {})
        assert isinstance(out, dict)
        assert "active_present_subjunctive_1sg" in out
        assert "active_aorist_subjunctive_1sg" not in out

    def test_none_lemma_returns_empty(self, synth):
        out = synth.synthesize_active_moods(None, {"fut": "λύσω"})
        assert out == {}

    def test_non_omega_lemma_returns_empty(self, synth):
        out = synth.synthesize_active_moods(
            "ἀνήρ", {"fut": "whatever"}
        )
        assert out == {}


# ---------------------------------------------------------------------------
# Accent neutralisation on ending-accented cells
# ---------------------------------------------------------------------------


class TestAccentHandling:
    """3sg/3pl imperative endings carry their own accent; the synthesis
    must drop the stem accent to avoid producing double-accented forms
    like ``γράψάτω``."""

    def test_no_double_accent_on_imp_3sg(self, synth):
        out = synth.synthesize_active_moods("γράφω", {"aor": "ἔγραψα"})
        form = out["active_aorist_imperative_3sg"]
        # Only one acute should be present (on the ending vowel).
        nfd = unicodedata.normalize("NFD", form)
        n_acute = nfd.count("́")
        assert n_acute == 1, f"got {n_acute} acutes in {form!r}"

    def test_no_double_accent_on_imp_3pl(self, synth):
        out = synth.synthesize_active_moods("παιδεύω", {"fut": "παιδεύσω"})
        form = out["active_aorist_imperative_3pl"]
        nfd = unicodedata.normalize("NFD", form)
        n_acute = nfd.count("́")
        assert n_acute == 1, f"got {n_acute} acutes in {form!r}"

    def test_2sg_2pl_keep_stem_accent(self, synth):
        """2sg/2pl imperatives have un-accented endings; stem accent
        should be preserved."""
        out = synth.synthesize_active_moods("παιδεύω", {"fut": "παιδεύσω"})
        # παιδεύσον has acute on ευ (the diphthong)
        assert out["active_aorist_imperative_2sg"] == "παιδεύσον"
        assert out["active_aorist_imperative_2pl"] == "παιδεύσατε"
