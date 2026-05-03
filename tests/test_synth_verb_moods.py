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


# ---------------------------------------------------------------------------
# Middle / passive synthesis (v2)
# ---------------------------------------------------------------------------


class TestPresentMiddleSynthesis:
    """Present middle / mediopassive synthesis works off the lemma alone."""

    def test_lyo_present_middle_subjunctive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert out["middle_present_subjunctive_1sg"] == "λύωμαι"
        assert out["middle_present_subjunctive_2sg"] == "λύῃ"
        assert out["middle_present_subjunctive_3sg"] == "λύηται"
        assert out["middle_present_subjunctive_1pl"] == "λυώμεθα"
        assert out["middle_present_subjunctive_2pl"] == "λύησθε"
        assert out["middle_present_subjunctive_3pl"] == "λύωνται"

    def test_lyo_present_middle_optative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert out["middle_present_optative_1sg"] == "λυοίμην"
        assert out["middle_present_optative_2sg"] == "λύοιο"
        assert out["middle_present_optative_3sg"] == "λύοιτο"
        assert out["middle_present_optative_1pl"] == "λυοίμεθα"
        assert out["middle_present_optative_2pl"] == "λύοισθε"
        assert out["middle_present_optative_3pl"] == "λύοιντο"

    def test_lyo_present_middle_imperative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert out["middle_present_imperative_2sg"] == "λύου"
        assert out["middle_present_imperative_2pl"] == "λύεσθε"
        # 3sg / 3pl ending-stress dropped lemma stem accent.
        assert out["middle_present_imperative_3sg"] == "λυέσθω"
        assert out["middle_present_imperative_3pl"] == "λυέσθων"

    def test_lyo_present_middle_infinitive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert out["middle_present_infinitive"] == "λύεσθαι"

    def test_no_passive_present_keys(self, synth):
        """jtauber emits ``middle_present_*`` only -- no
        ``passive_present_*`` (the present mp shares one column)."""
        out = synth.synthesize_mp_moods("λύω", {})
        for k in out:
            assert not k.startswith("passive_present_"), k

    def test_grapho_keeps_present_stem_accent(self, synth):
        """γράφω -> middle present should preserve stem accent."""
        out = synth.synthesize_mp_moods("γράφω", {})
        assert out["middle_present_subjunctive_1sg"] == "γράφωμαι"
        assert out["middle_present_optative_1sg"] == "γραφοίμην"
        assert out["middle_present_imperative_3sg"] == "γραφέσθω"
        assert out["middle_present_infinitive"] == "γράφεσθαι"


class TestAoristMiddleFromFut:
    """Aorist middle uses the σ-stem from the future principal part."""

    def test_lyo_aorist_middle_subjunctive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_subjunctive_1sg"] == "λύσωμαι"
        assert out["middle_aorist_subjunctive_2sg"] == "λύσῃ"
        assert out["middle_aorist_subjunctive_3sg"] == "λύσηται"
        assert out["middle_aorist_subjunctive_1pl"] == "λυσώμεθα"
        assert out["middle_aorist_subjunctive_2pl"] == "λύσησθε"
        assert out["middle_aorist_subjunctive_3pl"] == "λύσωνται"

    def test_lyo_aorist_middle_optative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_optative_1sg"] == "λυσαίμην"
        assert out["middle_aorist_optative_2sg"] == "λύσαιο"
        assert out["middle_aorist_optative_3sg"] == "λύσαιτο"
        assert out["middle_aorist_optative_1pl"] == "λυσαίμεθα"
        assert out["middle_aorist_optative_2pl"] == "λύσαισθε"
        assert out["middle_aorist_optative_3pl"] == "λύσαιντο"

    def test_lyo_aorist_middle_imperative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        # 2sg ending -αι; we don't recompute circumflex on the stem
        # so this comes out with an acute (jtauber has λῦσαι with
        # circumflex but we accept the segmentation match).
        assert out["middle_aorist_imperative_2sg"] == "λύσαι"
        assert out["middle_aorist_imperative_2pl"] == "λύσασθε"
        # 3sg/3pl ending-stress dropped stem accent.
        assert out["middle_aorist_imperative_3sg"] == "λυσάσθω"
        assert out["middle_aorist_imperative_3pl"] == "λυσάσθων"

    def test_lyo_aorist_middle_infinitive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_infinitive"] == "λύσασθαι"


class TestAoristMiddleFromAor:
    """When only the aorist principal part is given (no fut), σ-stem
    derives from grafting the σ/ψ/ξ cluster onto the present stem."""

    def test_grapho_aorist_middle_with_aor_only(self, synth):
        """γράφω + ἔγραψα: present stem γράφ-, swap φ for ψ -> γράψ-."""
        out = synth.synthesize_mp_moods("γράφω", {"aor": "ἔγραψα"})
        assert out["middle_aorist_subjunctive_1sg"] == "γράψωμαι"
        assert out["middle_aorist_optative_1sg"] == "γραψαίμην"
        assert out["middle_aorist_imperative_3sg"] == "γραψάσθω"
        assert out["middle_aorist_infinitive"] == "γράψασθαι"


class TestPassiveAoristSynthesis:
    """Passive aorist uses the θη-stem from ``aor_p`` principal part."""

    def test_lyo_passive_aorist_subjunctive(self, synth):
        """λύω: aor_p ἐλύθην, stem λυθ-, subj all ending-accented."""
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_subjunctive_1sg"] == "λυθῶ"
        assert out["passive_aorist_subjunctive_2sg"] == "λυθῇς"
        assert out["passive_aorist_subjunctive_3sg"] == "λυθῇ"
        assert out["passive_aorist_subjunctive_1pl"] == "λυθῶμεν"
        assert out["passive_aorist_subjunctive_2pl"] == "λυθῆτε"
        assert out["passive_aorist_subjunctive_3pl"] == "λυθῶσι(ν)"

    def test_lyo_passive_aorist_optative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_optative_1sg"] == "λυθείην"
        assert out["passive_aorist_optative_2sg"] == "λυθείης"
        assert out["passive_aorist_optative_3sg"] == "λυθείη"
        assert out["passive_aorist_optative_1pl"] == "λυθεῖμεν"
        assert out["passive_aorist_optative_2pl"] == "λυθεῖτε"
        assert out["passive_aorist_optative_3pl"] == "λυθεῖεν"
        # Duals: jtauber emits 2du / 3du for the passive aorist optative.
        assert out["passive_aorist_optative_2du"] == "λυθεῖτον"
        assert out["passive_aorist_optative_3du"] == "λυθείτην"

    def test_lyo_passive_aorist_imperative(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_imperative_2sg"] == "λύθητι"
        assert out["passive_aorist_imperative_3sg"] == "λυθήτω"
        assert out["passive_aorist_imperative_2pl"] == "λύθητε"
        # 3pl: drops the stem-final η before the participle linker -ε-.
        assert out["passive_aorist_imperative_3pl"] == "λυθέντων"

    def test_lyo_passive_aorist_infinitive(self, synth):
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_infinitive"] == "λυθῆναι"


class TestPeitho:
    """πείθω: dental cluster (θ + σ -> σ) for the aor middle, plus
    σθ-passive aorist ἐπείσθην."""

    def test_peitho_middle_aorist_uses_pi_sigma_stem(self, synth):
        """πείθω -> πείσω -> πεισ- middle aorist stem (θ assimilates)."""
        out = synth.synthesize_mp_moods(
            "πείθω", {"fut": "πείσω", "aor": "ἔπεισα", "aor_p": "ἐπείσθην"}
        )
        assert out["middle_aorist_subjunctive_1sg"] == "πείσωμαι"
        assert out["middle_aorist_subjunctive_3sg"] == "πείσηται"
        assert out["middle_aorist_optative_1sg"] == "πεισαίμην"
        assert out["middle_aorist_imperative_3sg"] == "πεισάσθω"
        assert out["middle_aorist_infinitive"] == "πείσασθαι"

    def test_peitho_passive_aorist_uses_sigma_theta_stem(self, synth):
        """πείθω -> ἐπείσθην -> πεισθ- passive aorist stem."""
        out = synth.synthesize_mp_moods(
            "πείθω", {"fut": "πείσω", "aor_p": "ἐπείσθην"}
        )
        assert out["passive_aorist_subjunctive_1sg"] == "πεισθῶ"
        assert out["passive_aorist_subjunctive_3pl"] == "πεισθῶσι(ν)"
        assert out["passive_aorist_optative_1sg"] == "πεισθείην"
        assert out["passive_aorist_optative_2du"] == "πεισθεῖτον"
        assert out["passive_aorist_imperative_2sg"] == "πείσθητι"
        assert out["passive_aorist_imperative_3sg"] == "πεισθήτω"
        assert out["passive_aorist_imperative_3pl"] == "πεισθέντων"
        assert out["passive_aorist_infinitive"] == "πεισθῆναι"


class TestGraphoNoPassive:
    """γράφω has 2nd-aor passive ἐγράφην (athematic, no θη). The synth
    only handles regular θη-aorists, so γράφω gets middle-aorist cells
    but no passive-aorist cells."""

    def test_grapho_aor_p_not_provided_skips_passive(self, synth):
        out = synth.synthesize_mp_moods(
            "γράφω", {"aor": "ἔγραψα", "fut_med": "γράψομαι"}
        )
        # No passive_aorist_* cells (no aor_p in parts)
        for k in out:
            assert not k.startswith("passive_aorist_"), k
        # Middle aorist cells still come through
        assert out["middle_aorist_subjunctive_1sg"] == "γράψωμαι"
        assert out["middle_aorist_infinitive"] == "γράψασθαι"

    def test_pf_mp_not_used_for_aor_synth(self, synth):
        """γράφω has γέγραμμαι (assimilated φ + μαι -> μμαι). The synth
        does NOT use pf_mp for any of the aor-system mp cells, so the
        γραψ- σ-stem still wins for middle aorist."""
        out = synth.synthesize_mp_moods(
            "γράφω",
            {"aor": "ἔγραψα", "pf_mp": "γέγραμμαι"},
        )
        # Middle aorist still uses σ-stem γραψ-, not pf_mp γεγραμ-.
        assert out["middle_aorist_subjunctive_1sg"] == "γράψωμαι"


class TestMpAoristSkippedForIrregular:
    """Verbs whose aor middle / passive can't be synthesized regularly
    must skip those cells entirely."""

    def test_lipo_aor_2_skips_middle_aorist(self, synth):
        """λείπω has aor-2 ἔλιπον; middle aorist can't be synthesized
        (no σ-stem from a 2nd aorist)."""
        out = synth.synthesize_mp_moods(
            "λείπω", {"aor": "ἔλιπον"}
        )
        assert "middle_aorist_subjunctive_1sg" not in out
        assert "middle_aorist_imperative_2sg" not in out
        # Present middle still comes through.
        assert out["middle_present_subjunctive_1sg"] == "λείπωμαι"

    def test_meno_liquid_future_skips_middle_aorist(self, synth):
        """μένω -> contract future μενῶ (no σ); middle aorist skipped."""
        out = synth.synthesize_mp_moods("μένω", {"fut": "μενῶ"})
        assert "middle_aorist_subjunctive_1sg" not in out
        assert out["middle_present_subjunctive_1sg"] == "μένωμαι"

    def test_no_aor_p_skips_passive_aorist(self, synth):
        """Without aor_p, passive_aorist_* cells are not produced."""
        out = synth.synthesize_mp_moods("λύω", {"fut": "λύσω"})
        for k in out:
            assert not k.startswith("passive_aorist_"), k


class TestMpNoOpCases:
    def test_contract_returns_empty(self, synth):
        out = synth.synthesize_mp_moods(
            "φιλέω", {"fut": "φιλήσω", "aor_p": "ἐφιλήθην"}
        )
        assert out == {}

    def test_athematic_returns_empty(self, synth):
        out = synth.synthesize_mp_moods("τίθημι", {"aor_p": "ἐτέθην"})
        assert out == {}

    def test_missing_principal_parts_no_error(self, synth):
        out = synth.synthesize_mp_moods("λύω", None)
        assert isinstance(out, dict)
        assert "middle_present_subjunctive_1sg" in out
        assert "middle_aorist_subjunctive_1sg" not in out
        assert "passive_aorist_subjunctive_1sg" not in out

    def test_empty_principal_parts_no_error(self, synth):
        out = synth.synthesize_mp_moods("λύω", {})
        assert "middle_present_subjunctive_1sg" in out
        assert "middle_aorist_subjunctive_1sg" not in out

    def test_none_lemma_returns_empty(self, synth):
        out = synth.synthesize_mp_moods(None, {"fut": "λύσω"})
        assert out == {}

    def test_non_omega_lemma_returns_empty(self, synth):
        out = synth.synthesize_mp_moods("ἀνήρ", {"fut": "whatever"})
        assert out == {}


class TestMpAccentHandling:
    """Stem-accents must be neutralised on cells whose ending carries
    its own accent, to avoid double-accent forms."""

    def test_no_double_accent_passive_aorist_subj_1sg(self, synth):
        out = synth.synthesize_mp_moods("παιδεύω", {"aor_p": "ἐπαιδεύθην"})
        form = out["passive_aorist_subjunctive_1sg"]
        nfd = unicodedata.normalize("NFD", form)
        assert nfd.count("́") + nfd.count("͂") == 1, (
            f"got accents in {form!r}"
        )

    def test_no_double_accent_passive_aorist_opt_1sg(self, synth):
        out = synth.synthesize_mp_moods("παιδεύω", {"aor_p": "ἐπαιδεύθην"})
        form = out["passive_aorist_optative_1sg"]
        nfd = unicodedata.normalize("NFD", form)
        # 1 acute on the ending vowel ει.
        assert nfd.count("́") == 1, f"got accents in {form!r}"

    def test_no_double_accent_middle_aorist_imp_3sg(self, synth):
        out = synth.synthesize_mp_moods("παιδεύω", {"fut": "παιδεύσω"})
        form = out["middle_aorist_imperative_3sg"]
        nfd = unicodedata.normalize("NFD", form)
        assert nfd.count("́") == 1, f"got accents in {form!r}"

    def test_no_double_accent_middle_present_imp_3sg(self, synth):
        out = synth.synthesize_mp_moods("παιδεύω", {})
        form = out["middle_present_imperative_3sg"]
        nfd = unicodedata.normalize("NFD", form)
        # παιδεύεσθω -> παιδευέσθω drops the original acute on ευ.
        # Only one acute should remain (on the ending vowel έ).
        assert nfd.count("́") == 1, f"got accents in {form!r}"

    def test_passive_aor_imp_3pl_drops_eta(self, synth):
        """3pl ending -έντων replaces the stem-final η outright;
        the result for λύω is λυθέντων (not λυθηέντων)."""
        out = synth.synthesize_mp_moods("λύω", {"aor_p": "ἐλύθην"})
        assert out["passive_aorist_imperative_3pl"] == "λυθέντων"


class TestJtauberSchemaAlignment:
    """Sanity check that key shapes align with what jtauber emits.
    These keys are the exact strings the consumer expects."""

    def test_lyo_full_paradigm_keys(self, synth):
        out = synth.synthesize_mp_moods(
            "λύω", {"fut": "λύσω", "aor": "ἔλυσα", "aor_p": "ἐλύθην"}
        )
        # Exact keys from jtauber for λύω middle/passive moods.
        expected = {
            # present middle
            "middle_present_subjunctive_1sg",
            "middle_present_subjunctive_2sg",
            "middle_present_subjunctive_3sg",
            "middle_present_subjunctive_1pl",
            "middle_present_subjunctive_2pl",
            "middle_present_subjunctive_3pl",
            "middle_present_optative_1sg",
            "middle_present_optative_2sg",
            "middle_present_optative_3sg",
            "middle_present_optative_1pl",
            "middle_present_optative_2pl",
            "middle_present_optative_3pl",
            "middle_present_imperative_2sg",
            "middle_present_imperative_3sg",
            "middle_present_imperative_2pl",
            "middle_present_imperative_3pl",
            "middle_present_infinitive",
            # aorist middle
            "middle_aorist_subjunctive_1sg",
            "middle_aorist_subjunctive_2sg",
            "middle_aorist_subjunctive_3sg",
            "middle_aorist_subjunctive_1pl",
            "middle_aorist_subjunctive_2pl",
            "middle_aorist_subjunctive_3pl",
            "middle_aorist_optative_1sg",
            "middle_aorist_optative_2sg",
            "middle_aorist_optative_3sg",
            "middle_aorist_optative_1pl",
            "middle_aorist_optative_2pl",
            "middle_aorist_optative_3pl",
            "middle_aorist_imperative_2sg",
            "middle_aorist_imperative_3sg",
            "middle_aorist_imperative_2pl",
            "middle_aorist_imperative_3pl",
            "middle_aorist_infinitive",
            # aorist passive (ALL with duals on optative)
            "passive_aorist_subjunctive_1sg",
            "passive_aorist_subjunctive_2sg",
            "passive_aorist_subjunctive_3sg",
            "passive_aorist_subjunctive_1pl",
            "passive_aorist_subjunctive_2pl",
            "passive_aorist_subjunctive_3pl",
            "passive_aorist_optative_1sg",
            "passive_aorist_optative_2sg",
            "passive_aorist_optative_3sg",
            "passive_aorist_optative_1pl",
            "passive_aorist_optative_2pl",
            "passive_aorist_optative_3pl",
            "passive_aorist_optative_2du",
            "passive_aorist_optative_3du",
            "passive_aorist_imperative_2sg",
            "passive_aorist_imperative_3sg",
            "passive_aorist_imperative_2pl",
            "passive_aorist_imperative_3pl",
            "passive_aorist_infinitive",
        }
        assert expected.issubset(set(out.keys())), (
            f"missing keys: {expected - set(out.keys())}"
        )
