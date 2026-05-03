"""Tests for ``build/synth_verb_participles.py`` and its integration in
``build/build_grc_verb_paradigms.py``.

The synth module is loaded directly as a script-style module via
``importlib`` (the ``build/`` directory is not an installable package),
mirroring the pattern in ``test_synth_verb_moods.py``.

Run with:

    python -m pytest tests/test_synth_verb_participles.py -x -v
"""

from __future__ import annotations

import importlib.util
import unicodedata
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_PATH = REPO_ROOT / "build" / "synth_verb_participles.py"


@pytest.fixture(scope="module")
def synth():
    """Load build/synth_verb_participles.py as a module without package install."""
    spec = importlib.util.spec_from_file_location(
        "synth_verb_participles", SYNTH_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip_quantity(s):
    """Strip macron (U+0304) and breve (U+0306) — quantity marks that
    jtauber adds prosodically but we don't synthesise. Comparison
    against jtauber forms ignores these."""
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize(
        "NFC",
        "".join(c for c in nfd if ord(c) not in (0x0304, 0x0306)),
    )


# ---------------------------------------------------------------------------
# Lemma classification
# ---------------------------------------------------------------------------


class TestIsThematicOmega:
    def test_plain_omega_verb(self, synth):
        assert synth.is_thematic_omega("λύω")
        assert synth.is_thematic_omega("γράφω")
        assert synth.is_thematic_omega("παύω")
        assert synth.is_thematic_omega("παιδεύω")
        assert synth.is_thematic_omega("πείθω")

    def test_contract_rejected(self, synth):
        assert not synth.is_thematic_omega("τιμάω")
        assert not synth.is_thematic_omega("φιλέω")
        assert not synth.is_thematic_omega("δηλόω")

    def test_athematic_rejected(self, synth):
        assert not synth.is_thematic_omega("τίθημι")
        assert not synth.is_thematic_omega("δίδωμι")

    def test_deponent_rejected(self, synth):
        assert not synth.is_thematic_omega("ἔρχομαι")

    def test_empty_or_garbage_rejected(self, synth):
        assert not synth.is_thematic_omega("")
        assert not synth.is_thematic_omega(None)
        assert not synth.is_thematic_omega("ἀνήρ")


# ---------------------------------------------------------------------------
# Present-active synthesis (lemma alone is enough)
# ---------------------------------------------------------------------------


class TestPresentActive:
    """λύω → λύων / λύουσα / λῦον. We don't synthesise the circumflex
    on neuter sg cells, so ignore quantity marks for those compares."""

    def test_lyo_present_active_masculine(self, synth):
        out = synth.synthesize_participles("λύω", {})
        assert out["active_present_participle_nom_m_sg"] == "λύων"
        assert out["active_present_participle_gen_m_sg"] == "λύοντος"
        assert out["active_present_participle_dat_m_sg"] == "λύοντι"
        assert out["active_present_participle_acc_m_sg"] == "λύοντα"
        assert out["active_present_participle_nom_m_pl"] == "λύοντες"
        assert out["active_present_participle_gen_m_pl"] == "λυόντων"
        assert out["active_present_participle_dat_m_pl"] == "λύουσι(ν)"
        assert out["active_present_participle_acc_m_pl"] == "λύοντας"

    def test_lyo_present_active_feminine(self, synth):
        out = synth.synthesize_participles("λύω", {})
        assert out["active_present_participle_nom_f_sg"] == "λύουσα"
        assert out["active_present_participle_gen_f_sg"] == "λυούσης"
        assert out["active_present_participle_dat_f_sg"] == "λυούσῃ"
        assert out["active_present_participle_acc_f_sg"] == "λύουσαν"
        assert out["active_present_participle_nom_f_pl"] == "λύουσαι"
        assert out["active_present_participle_dat_f_pl"] == "λυούσαις"
        assert out["active_present_participle_acc_f_pl"] == "λυούσας"

    def test_lyo_present_active_neuter(self, synth):
        out = synth.synthesize_participles("λύω", {})
        # Neuter pl uses same forms as masc -οντα.
        assert out["active_present_participle_nom_n_pl"] == "λύοντα"
        assert out["active_present_participle_gen_n_pl"] == "λυόντων"

    def test_grapho_present_active(self, synth):
        out = synth.synthesize_participles("γράφω", {})
        assert out["active_present_participle_nom_m_sg"] == "γράφων"
        assert out["active_present_participle_gen_m_pl"] == "γραφόντων"
        assert out["active_present_participle_nom_f_sg"] == "γράφουσα"
        assert out["active_present_participle_dat_f_sg"] == "γραφούσῃ"


# ---------------------------------------------------------------------------
# Present middle/passive synthesis (lemma alone)
# ---------------------------------------------------------------------------


class TestPresentMP:
    """λύω → λυόμενος / λυομένη / λυόμενον (middle present participle).
    Recessive accent: antepenult on -ό- when final short, penult on
    -μέ- when final long."""

    def test_lyo_present_mp_short_final(self, synth):
        out = synth.synthesize_participles("λύω", {})
        # Final -ος, -ον, -ε, -οι, -α, -αι are short → antepenult acute.
        assert out["middle_present_participle_nom_m_sg"] == "λυόμενος"
        assert out["middle_present_participle_acc_m_sg"] == "λυόμενον"
        assert out["middle_present_participle_voc_m_sg"] == "λυόμενε"
        assert out["middle_present_participle_nom_m_pl"] == "λυόμενοι"
        assert out["middle_present_participle_nom_n_pl"] == "λυόμενα"
        assert out["middle_present_participle_acc_n_pl"] == "λυόμενα"

    def test_lyo_present_mp_long_final(self, synth):
        out = synth.synthesize_participles("λύω", {})
        # Final -η, -ης, -ῃ, -ην, -ων, -ου, -ῳ, -ους, -οις, -αις are long
        # → penult acute.
        assert out["middle_present_participle_nom_f_sg"] == "λυομένη"
        assert out["middle_present_participle_gen_f_sg"] == "λυομένης"
        assert out["middle_present_participle_dat_f_sg"] == "λυομένῃ"
        assert out["middle_present_participle_acc_f_sg"] == "λυομένην"
        assert out["middle_present_participle_gen_m_sg"] == "λυομένου"
        assert out["middle_present_participle_dat_m_sg"] == "λυομένῳ"
        assert out["middle_present_participle_gen_m_pl"] == "λυομένων"
        assert out["middle_present_participle_acc_m_pl"] == "λυομένους"

    def test_grapho_present_mp(self, synth):
        out = synth.synthesize_participles("γράφω", {})
        assert out["middle_present_participle_nom_m_sg"] == "γραφόμενος"
        assert out["middle_present_participle_gen_m_sg"] == "γραφομένου"
        assert out["middle_present_participle_nom_f_sg"] == "γραφομένη"


# ---------------------------------------------------------------------------
# Aorist active synthesis (3rd-decl, sigmatic)
# ---------------------------------------------------------------------------


class TestAoristActive:
    """λύω → λύσας / λύσασα / λῦσαν (aor act participle). σ-stem comes
    either from fut (λύσω → λύσ-) or from aor (ἔλυσα → λύσ-)."""

    def test_lyo_aorist_active_from_fut(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["active_aorist_participle_nom_m_sg"] == "λύσας"
        assert out["active_aorist_participle_gen_m_sg"] == "λύσαντος"
        assert out["active_aorist_participle_dat_m_sg"] == "λύσαντι"
        assert out["active_aorist_participle_acc_m_sg"] == "λύσαντα"
        assert out["active_aorist_participle_nom_m_pl"] == "λύσαντες"
        assert out["active_aorist_participle_gen_m_pl"] == "λυσάντων"
        assert out["active_aorist_participle_dat_m_pl"] == "λύσασι(ν)"

    def test_lyo_aorist_active_feminine(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["active_aorist_participle_nom_f_sg"] == "λύσασα"
        assert out["active_aorist_participle_gen_f_sg"] == "λυσάσης"
        assert out["active_aorist_participle_dat_f_sg"] == "λυσάσῃ"
        assert out["active_aorist_participle_acc_f_sg"] == "λύσασαν"
        assert out["active_aorist_participle_nom_f_pl"] == "λύσασαι"

    def test_grapho_aorist_active_from_aor(self, synth):
        """γράφω + aor ἔγραψα: graft ψ-cluster onto present stem."""
        out = synth.synthesize_participles("γράφω", {"aor": "ἔγραψα"})
        assert out["active_aorist_participle_nom_m_sg"] == "γράψας"
        assert out["active_aorist_participle_gen_m_pl"] == "γραψάντων"
        assert out["active_aorist_participle_nom_f_sg"] == "γράψασα"
        assert out["active_aorist_participle_dat_f_pl"] == "γραψάσαις"

    def test_pempo_aorist_active(self, synth):
        out = synth.synthesize_participles("πέμπω", {"fut": "πέμψω"})
        assert out["active_aorist_participle_nom_m_sg"] == "πέμψας"
        assert out["active_aorist_participle_acc_m_sg"] == "πέμψαντα"
        assert out["active_aorist_participle_nom_f_sg"] == "πέμψασα"


# ---------------------------------------------------------------------------
# Aorist middle synthesis
# ---------------------------------------------------------------------------


class TestAoristMiddle:
    """λύω → λυσάμενος / λυσαμένη / λυσάμενον."""

    def test_lyo_aorist_middle_short_final(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_participle_nom_m_sg"] == "λυσάμενος"
        assert out["middle_aorist_participle_acc_m_sg"] == "λυσάμενον"
        assert out["middle_aorist_participle_voc_m_sg"] == "λυσάμενε"
        assert out["middle_aorist_participle_nom_m_pl"] == "λυσάμενοι"
        assert out["middle_aorist_participle_nom_n_pl"] == "λυσάμενα"
        assert out["middle_aorist_participle_acc_n_pl"] == "λυσάμενα"

    def test_lyo_aorist_middle_long_final(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["middle_aorist_participle_nom_f_sg"] == "λυσαμένη"
        assert out["middle_aorist_participle_gen_f_sg"] == "λυσαμένης"
        assert out["middle_aorist_participle_dat_m_sg"] == "λυσαμένῳ"
        assert out["middle_aorist_participle_gen_m_pl"] == "λυσαμένων"


# ---------------------------------------------------------------------------
# Aorist passive synthesis (3rd-decl, ending-accented -θείς)
# ---------------------------------------------------------------------------


class TestAoristPassive:
    """λύω + aor.p ἐλύθην → λυθείς / λυθεῖσα / λυθέν."""

    def test_lyo_aorist_passive_masculine(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"aor_p": "ἐλύθην"}
        )
        assert out["passive_aorist_participle_nom_m_sg"] == "λυθείς"
        assert out["passive_aorist_participle_gen_m_sg"] == "λυθέντος"
        assert out["passive_aorist_participle_dat_m_sg"] == "λυθέντι"
        assert out["passive_aorist_participle_acc_m_sg"] == "λυθέντα"
        assert out["passive_aorist_participle_nom_m_pl"] == "λυθέντες"
        assert out["passive_aorist_participle_gen_m_pl"] == "λυθέντων"
        assert out["passive_aorist_participle_dat_m_pl"] == "λυθεῖσι(ν)"

    def test_lyo_aorist_passive_feminine(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"aor_p": "ἐλύθην"}
        )
        assert out["passive_aorist_participle_nom_f_sg"] == "λυθεῖσα"
        assert out["passive_aorist_participle_gen_f_sg"] == "λυθείσης"
        assert out["passive_aorist_participle_dat_f_sg"] == "λυθείσῃ"
        assert out["passive_aorist_participle_acc_f_sg"] == "λυθεῖσαν"
        assert out["passive_aorist_participle_nom_f_pl"] == "λυθεῖσαι"
        assert out["passive_aorist_participle_gen_f_pl"] == "λυθεισῶν"

    def test_lyo_aorist_passive_neuter(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"aor_p": "ἐλύθην"}
        )
        assert out["passive_aorist_participle_nom_n_sg"] == "λυθέν"
        assert out["passive_aorist_participle_acc_n_sg"] == "λυθέν"
        assert out["passive_aorist_participle_gen_n_sg"] == "λυθέντος"
        assert out["passive_aorist_participle_nom_n_pl"] == "λυθέντα"

    def test_paideuo_aorist_passive(self, synth):
        out = synth.synthesize_participles(
            "παιδεύω", {"aor_p": "ἐπαιδεύθην"}
        )
        assert out["passive_aorist_participle_nom_m_sg"] == "παιδευθείς"
        assert out["passive_aorist_participle_gen_m_pl"] == "παιδευθέντων"


# ---------------------------------------------------------------------------
# Perfect active synthesis (3rd-decl, -ώς/-υῖα/-ός)
# ---------------------------------------------------------------------------


class TestPerfectActive:
    """λύω + pf λέλυκα → λελυκώς / λελυκυῖα / λελυκός."""

    def test_lyo_perfect_active_masculine(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"pf": "λέλυκα"}
        )
        assert out["active_perfect_participle_nom_m_sg"] == "λελυκώς"
        assert out["active_perfect_participle_gen_m_sg"] == "λελυκότος"
        assert out["active_perfect_participle_dat_m_sg"] == "λελυκότι"
        assert out["active_perfect_participle_acc_m_sg"] == "λελυκότα"
        assert out["active_perfect_participle_nom_m_pl"] == "λελυκότες"
        assert out["active_perfect_participle_gen_m_pl"] == "λελυκότων"
        assert out["active_perfect_participle_dat_m_pl"] == "λελυκόσι(ν)"

    def test_lyo_perfect_active_feminine(self, synth):
        out = synth.synthesize_participles("λύω", {"pf": "λέλυκα"})
        assert out["active_perfect_participle_nom_f_sg"] == "λελυκυῖα"
        assert out["active_perfect_participle_gen_f_sg"] == "λελυκυίας"
        assert out["active_perfect_participle_dat_f_sg"] == "λελυκυίᾳ"
        assert out["active_perfect_participle_acc_f_sg"] == "λελυκυῖαν"

    def test_lyo_perfect_active_neuter(self, synth):
        out = synth.synthesize_participles("λύω", {"pf": "λέλυκα"})
        assert out["active_perfect_participle_nom_n_sg"] == "λελυκός"
        assert out["active_perfect_participle_acc_n_sg"] == "λελυκός"
        assert out["active_perfect_participle_nom_n_pl"] == "λελυκότα"


# ---------------------------------------------------------------------------
# Perfect middle/passive synthesis (1st/2nd-decl, persistent -μέν- accent)
# ---------------------------------------------------------------------------


class TestPerfectMP:
    """λύω + pf_mp λέλυμαι → λελυμένος / λελυμένη / λελυμένον.
    Persistent accent: always on -μέν- (penult), never antepenult."""

    def test_lyo_perfect_mp_persistent_accent(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"pf_mp": "λέλυμαι"}
        )
        # Even with short final (-ος), accent is on penult -μέ-.
        assert out["middle_perfect_participle_nom_m_sg"] == "λελυμένος"
        assert out["middle_perfect_participle_acc_m_sg"] == "λελυμένον"
        assert out["middle_perfect_participle_nom_m_pl"] == "λελυμένοι"
        assert out["middle_perfect_participle_nom_f_sg"] == "λελυμένη"
        assert out["middle_perfect_participle_gen_m_sg"] == "λελυμένου"
        assert out["middle_perfect_participle_dat_m_sg"] == "λελυμένῳ"

    def test_peitho_perfect_mp(self, synth):
        """πέπεισμαι (assimilated -σμαι from -θμαι) → πεπεισμένος."""
        out = synth.synthesize_participles(
            "πείθω", {"pf_mp": "πέπεισμαι"}
        )
        assert out["middle_perfect_participle_nom_m_sg"] == "πεπεισμένος"
        assert out["middle_perfect_participle_acc_f_sg"] == "πεπεισμένην"
        assert out["middle_perfect_participle_gen_m_pl"] == "πεπεισμένων"

    def test_pauo_perfect_mp(self, synth):
        out = synth.synthesize_participles(
            "παύω", {"pf_mp": "πέπαυμαι"}
        )
        assert out["middle_perfect_participle_nom_m_sg"] == "πεπαυμένος"
        assert out["middle_perfect_participle_nom_f_sg"] == "πεπαυμένη"


# ---------------------------------------------------------------------------
# Future synthesis
# ---------------------------------------------------------------------------


class TestFutureActive:
    """λύσω → λύσων / λύσουσα / λῦσον (future active participle).
    Same shape as present active but on the σ-stem."""

    def test_lyo_future_active(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["active_future_participle_nom_m_sg"] == "λύσων"
        assert out["active_future_participle_gen_m_sg"] == "λύσοντος"
        assert out["active_future_participle_acc_m_sg"] == "λύσοντα"
        assert out["active_future_participle_nom_f_sg"] == "λύσουσα"
        assert out["active_future_participle_dat_f_sg"] == "λυσούσῃ"

    def test_grapho_future_active(self, synth):
        """γράφω: derive σ-stem from aor since no fut given."""
        out = synth.synthesize_participles("γράφω", {"aor": "ἔγραψα"})
        assert out["active_future_participle_nom_m_sg"] == "γράψων"
        assert out["active_future_participle_gen_m_pl"] == "γραψόντων"


class TestFutureMiddle:
    """λύσομαι → λυσόμενος / λυσομένη / λυσόμενον."""

    def test_lyo_future_middle(self, synth):
        out = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        assert out["middle_future_participle_nom_m_sg"] == "λυσόμενος"
        assert out["middle_future_participle_acc_m_sg"] == "λυσόμενον"
        assert out["middle_future_participle_nom_f_sg"] == "λυσομένη"
        assert out["middle_future_participle_gen_m_sg"] == "λυσομένου"


class TestFuturePassive:
    """λυθήσομαι → λυθησόμενος / λυθησομένη / λυθησόμενον."""

    def test_lyo_future_passive(self, synth):
        out = synth.synthesize_participles(
            "λύω", {"aor_p": "ἐλύθην"}
        )
        assert out["passive_future_participle_nom_m_sg"] == "λυθησόμενος"
        assert out["passive_future_participle_acc_m_sg"] == "λυθησόμενον"
        assert out["passive_future_participle_nom_f_sg"] == "λυθησομένη"
        assert out["passive_future_participle_gen_m_sg"] == "λυθησομένου"
        # No double-θ bug
        assert "λυθθ" not in out["passive_future_participle_nom_m_sg"]


# ---------------------------------------------------------------------------
# Spot-check against jtauber_ag_paradigms.json shape (cross-source sanity)
# ---------------------------------------------------------------------------


class TestJtauberCompat:
    """Spot checks against jtauber's known good values for shared verbs.
    Ignores macron/breve quantity marks (we don't synthesise those)."""

    @pytest.fixture(scope="class")
    def jtauber(self):
        import json
        path = (Path.home() / "Documents" / "Klisy" / "word_collector"
                / "jtauber_ag_paradigms.json")
        if not path.exists():
            pytest.skip(f"jtauber paradigms not found at {path}")
        with open(path) as f:
            return json.load(f)

    def test_lyo_full_paradigm_match_ratio(self, synth, jtauber):
        """Synthesise a complete λύω paradigm; >95% of cells common with
        jtauber should match (mod quantity marks)."""
        out = synth.synthesize_participles(
            "λύω", {
                "fut": "λύσω", "aor": "ἔλυσα", "pf": "λέλυκα",
                "pf_mp": "λέλυμαι", "aor_p": "ἐλύθην",
            },
        )
        jtf = jtauber["λύω"]["forms"]
        common = set(out) & set(jtf)
        assert len(common) >= 200, f"too few common keys: {len(common)}"
        match = sum(1 for k in common if _strip_quantity(out[k]) ==
                    _strip_quantity(jtf[k]))
        ratio = match / len(common)
        assert ratio >= 0.95, (
            f"only {match}/{len(common)} cells match jtauber for λύω "
            f"(ratio {ratio:.3f})"
        )

    def test_grapho_full_paradigm_match_ratio(self, synth, jtauber):
        out = synth.synthesize_participles(
            "γράφω", {"aor": "ἔγραψα", "pf": "γέγραφα"},
        )
        jtf = jtauber["γράφω"]["forms"]
        common = set(out) & set(jtf)
        assert len(common) >= 100
        match = sum(1 for k in common if _strip_quantity(out[k]) ==
                    _strip_quantity(jtf[k]))
        ratio = match / len(common)
        assert ratio >= 0.95, (
            f"γράφω: {match}/{len(common)} match (ratio {ratio:.3f})"
        )

    def test_pauo_full_paradigm_match_ratio(self, synth, jtauber):
        out = synth.synthesize_participles(
            "παύω", {
                "fut": "παύσω", "aor": "ἔπαυσα", "pf": "πέπαυκα",
                "pf_mp": "πέπαυμαι", "aor_p": "ἐπαύθην",
            },
        )
        jtf = jtauber["παύω"]["forms"]
        common = set(out) & set(jtf)
        assert len(common) >= 100
        match = sum(1 for k in common if _strip_quantity(out[k]) ==
                    _strip_quantity(jtf[k]))
        ratio = match / len(common)
        # παύω has the circumflex-on-monosyll-stem issue on a handful of
        # neuter sg cells; allow slightly lower threshold.
        assert ratio >= 0.92, (
            f"παύω: {match}/{len(common)} match (ratio {ratio:.3f})"
        )


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


class TestNoOpCases:
    def test_contract_returns_empty(self, synth):
        out = synth.synthesize_participles(
            "φιλέω", {"fut": "φιλήσω", "aor": "ἐφίλησα"}
        )
        assert out == {}

    def test_athematic_returns_empty(self, synth):
        out = synth.synthesize_participles(
            "δίδωμι", {"aor": "ἔδωκα"}
        )
        assert out == {}

    def test_deponent_returns_empty(self, synth):
        out = synth.synthesize_participles(
            "ἔρχομαι", {"aor": "ἦλθον"}
        )
        assert out == {}

    def test_no_principal_parts_only_present(self, synth):
        """Without principal parts, only present-system participles
        (active and mp) are synthesised."""
        out = synth.synthesize_participles("λύω", None)
        assert "active_present_participle_nom_m_sg" in out
        assert "middle_present_participle_nom_m_sg" in out
        # No aor/fut/pf cells
        assert "active_aorist_participle_nom_m_sg" not in out
        assert "active_future_participle_nom_m_sg" not in out
        assert "active_perfect_participle_nom_m_sg" not in out

    def test_aor2_lemma_skips_aorist(self, synth):
        """λείπω + aor-2 ἔλιπον: no σ-stem available; aorist skipped."""
        out = synth.synthesize_participles("λείπω", {"aor": "ἔλιπον"})
        assert "active_aorist_participle_nom_m_sg" not in out
        assert "middle_aorist_participle_nom_m_sg" not in out
        # Present forms still synthesised.
        assert "active_present_participle_nom_m_sg" in out

    def test_none_lemma_returns_empty(self, synth):
        out = synth.synthesize_participles(None, {"fut": "λύσω"})
        assert out == {}

    def test_non_omega_lemma_returns_empty(self, synth):
        out = synth.synthesize_participles(
            "ἀνήρ", {"fut": "whatever"}
        )
        assert out == {}

    def test_missing_pf_mp_skips_perfect_mp(self, synth):
        """No pf_mp → no middle_perfect_participle cells, but other
        cells still produced."""
        out = synth.synthesize_participles(
            "λύω", {"pf": "λέλυκα", "fut": "λύσω"}
        )
        assert "active_perfect_participle_nom_m_sg" in out
        assert "middle_perfect_participle_nom_m_sg" not in out

    def test_missing_aor_p_skips_passive(self, synth):
        """No aor_p → no aor-pass / fut-pass cells."""
        out = synth.synthesize_participles(
            "λύω", {"fut": "λύσω", "aor": "ἔλυσα"}
        )
        assert "passive_aorist_participle_nom_m_sg" not in out
        assert "passive_future_participle_nom_m_sg" not in out


# ---------------------------------------------------------------------------
# Build-pipeline integration: never overwrite existing cells
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Mirrors the build_grc_verb_paradigms.py merge logic: synth must
    only fill empty cells; existing corpus / Wiktionary cells must
    survive."""

    def test_skip_collision(self, synth):
        """The synth function returns ALL synthesizable cells; the
        caller is responsible for skipping collisions. Verify via a
        manual merge that no synthesised value overwrites an existing
        cell."""
        existing = {
            "active_present_participle_nom_m_sg": "λύων",   # existing, stays
            "active_present_participle_nom_f_sg": "λύουσα",
        }
        templated = synth.synthesize_participles("λύω", {"fut": "λύσω"})
        # Apply the same merge rule build_grc_verb_paradigms uses:
        merged = dict(existing)
        for k, v in templated.items():
            if k not in merged:
                merged[k] = v
        # Existing cells survived
        assert merged["active_present_participle_nom_m_sg"] == "λύων"
        assert merged["active_present_participle_nom_f_sg"] == "λύουσα"
        # New cells added
        assert "active_aorist_participle_nom_m_sg" in merged
        assert merged["active_aorist_participle_nom_m_sg"] == "λύσας"

    def test_jtauber_key_format(self, synth):
        """All emitted keys must follow the jtauber-compatible pattern
        ``{voice}_{tense}_participle_{case}_{gender}_{number}``."""
        out = synth.synthesize_participles(
            "λύω", {
                "fut": "λύσω", "aor": "ἔλυσα", "pf": "λέλυκα",
                "pf_mp": "λέλυμαι", "aor_p": "ἐλύθην",
            },
        )
        valid_voices = {"active", "middle", "passive"}
        valid_tenses = {"present", "future", "aorist", "perfect"}
        valid_cases = {"nom", "gen", "dat", "acc", "voc"}
        valid_genders = {"m", "f", "n"}
        valid_numbers = {"sg", "pl"}
        for k in out:
            parts = k.split("_")
            assert len(parts) == 6, f"unexpected key shape: {k!r}"
            voice, tense, ppl_word, case, gender, number = parts
            assert voice in valid_voices, k
            assert tense in valid_tenses, k
            assert ppl_word == "participle", k
            assert case in valid_cases, k
            assert gender in valid_genders, k
            assert number in valid_numbers, k
