#!/usr/bin/env python3
"""Orthographic normalizer for Greek texts with non-standard spelling.

Generates candidate normalized spellings for tokens exhibiting scribal
or epigraphic variation (itacism, vowel mergers, missing iota subscripta,
etc.). Candidates are meant to be checked against a lookup table.

Usage:
    from normalize import Normalizer

    n = Normalizer()                        # all periods
    n = Normalizer(period="byzantine")      # period-specific rules

    candidates = n.normalize("χέροντες")    # itacism: ε for αι
    candidates = n.normalize("θεω")         # missing iota subscriptum
"""

import unicodedata
from itertools import combinations


# --- Sound change rules ---
# Each rule: (surface_seq, normalized_seq, rule_id)
# "surface" = what the scribe wrote; "normalized" = what it should be.
# Rules are bidirectional: given a surface form, we try replacing
# surface_seq with normalized_seq and vice versa.

VOWEL_RULES = [
    # Itacism: η, ει, οι, υ all -> [i]
    ("ι", "ει", "itacism_ei_i"),
    ("ι", "η", "itacism_eta_i"),
    ("ι", "οι", "itacism_oi_i"),
    ("ι", "υ", "itacism_u_i"),
    ("η", "ει", "itacism_ei_eta"),
    ("οι", "ει", "itacism_ei_oi"),
    ("υ", "οι", "itacism_oi_u"),

    # αι / ε merger
    ("ε", "αι", "ai_e_merger"),
    ("αι", "ε", "ai_e_merger_rev"),

    # ο / ω confusion (loss of vowel length)
    ("ο", "ω", "o_omega"),
    ("ω", "ο", "o_omega_rev"),
]

CONSONANT_RULES = [
    # Spirantization: β -> [v], so β ↔ υ after vowels
    ("β", "υ", "spirant_beta"),

    # Aspiration loss: φ↔π, θ↔τ, χ↔κ
    ("π", "φ", "aspiration_p_ph"),
    ("φ", "π", "aspiration_ph_p"),
    ("τ", "θ", "aspiration_t_th"),
    ("θ", "τ", "aspiration_th_t"),
    ("κ", "χ", "aspiration_k_kh"),
    ("χ", "κ", "aspiration_kh_k"),

    # Geminate simplification
    ("λ", "λλ", "geminate_l"),
    ("ν", "νν", "geminate_n"),
    ("σ", "σσ", "geminate_s"),
    ("ρ", "ρρ", "geminate_r"),
    ("τ", "ττ", "geminate_t"),
    ("κ", "κκ", "geminate_k"),
    ("π", "ππ", "geminate_p"),
]


# --- Iota subscriptum restoration ---
# α -> ᾳ, η -> ῃ, ω -> ῳ (and accented variants)
SUBSCRIPTUM_MAP = {
    "α": "ᾳ", "ά": "ᾴ", "ᾶ": "ᾷ", "ὰ": "ᾲ",
    "η": "ῃ", "ή": "ῄ", "ῆ": "ῇ", "ὴ": "ῂ",
    "ω": "ῳ", "ώ": "ῴ", "ῶ": "ῷ", "ὼ": "ῲ",
}
# Reverse map for completeness
SUBSCRIPTUM_REV = {v: k for k, v in SUBSCRIPTUM_MAP.items()}


# --- Period profiles ---
# Weights 0.0-1.0 for each rule category per historical period.
# Higher weight = more likely this confusion occurs in texts of this period.
#
# Chronology calibrated against Horrocks (2010), "Greek: A History of the
# Language and its Speakers", 2nd ed., primarily chapters 4.11.2 and 6.
#
# Key dates from Horrocks:
#   - ει/ι merger: monophthongization by 7th c. BC (p.161), ει/ι confusion
#     in Attic inscriptions from 5th c. BC (p.163), standard by Hellenistic
#   - η/ι merger: gradual; η > [e:] by 4th c. BC in Attic (p.163), η/ι
#     interchange attested from late Ptolemaic but "never quite becomes
#     general even in the Roman period" (p.168), full merger in early
#     Byzantine (p.167)
#   - οι/ι merger: via /ø/ then /y/, complete only in middle Byzantine
#     period (p.163); οι still = /ø/ in mid-2nd c. BC (p.167)
#   - υ/ι merger: loss of lip-rounding "as late as the 9th/10th century AD"
#     (p.169)
#   - αι/ε merger: αι > /ae/ > /e/ complete in Attic majority by 4th c. BC
#     (p.163, 165); general interchange of ε/αι by beginning of 4th c. BC
#     (p.168)
#   - ο/ω confusion: loss of vowel length widespread by mid-2nd c. BC
#     (p.169, via stress accent shift)
#   - Iota subscriptum: long diphthongs lost [i] between c.150-50 BC
#     (p.164-165); frequently omitted in papyri from mid-2nd c. BC (p.175)
#   - Spirantization of voiced stops: began with /g/ by 2nd c. BC, labial
#     β = /β/ by 1st c. AD, complete for majority by 4th c. AD (p.170)
#   - Aspiration loss (φ,θ,χ > fricatives): meagre evidence for Hellenistic;
#     some evidence for φ > /f/ in 2nd c. BC Asia Minor (p.171); full
#     fricativization probably early Byzantine (p.171)
#   - Geminate simplification: from 3rd c. BC onwards (p.171)
#   - Loss of /h/ (breathing): progressive through Koine period, complete
#     by late Roman/Byzantine (p.171)

PROFILES = {
    "archaic_epigraphic": {
        # Pre-Hellenistic: almost no sound changes yet in standard spelling.
        # Only ει monophthongization is underway (Horrocks p.161, 163).
        "itacism_ei_i": 0.1,   # ει already monophthong, some confusion (p.163)
        "itacism_eta_i": 0.0, "itacism_oi_i": 0.0,
        "itacism_u_i": 0.0, "itacism_ei_eta": 0.0, "itacism_ei_oi": 0.0,
        "itacism_oi_u": 0.0,
        "ai_e_merger": 0.0, "ai_e_merger_rev": 0.0,
        "o_omega": 0.0, "o_omega_rev": 0.0,
        "iota_subscriptum": 0.0,
        "spirant_beta": 0.0,
        "aspiration_p_ph": 0.0, "aspiration_ph_p": 0.0,
        "aspiration_t_th": 0.0, "aspiration_th_t": 0.0,
        "aspiration_k_kh": 0.0, "aspiration_kh_k": 0.0,
        "geminate_l": 0.0, "geminate_n": 0.0, "geminate_s": 0.0,
        "geminate_r": 0.0, "geminate_t": 0.0, "geminate_k": 0.0,
        "geminate_p": 0.0,
        "breathing": 0.0,
    },
    "hellenistic": {
        # c. 300 BC - 1st c. BC. ει/ι well established; αι/ε merger advanced;
        # ο/ω confusion beginning; η still distinct from ι; οι still rounded.
        # Consonant changes just starting. (Horrocks pp.161-168)
        "itacism_ei_i": 0.7,   # ει/ι fully merged by now (p.163)
        "itacism_eta_i": 0.15, # η still [e:], only sporadic η/ι (p.168)
        "itacism_oi_i": 0.05,  # οι still rounded /ø/ (p.167)
        "itacism_u_i": 0.0,    # υ still /y/, no merger with ι yet (p.169)
        "itacism_ei_eta": 0.15, # ει/η confusion starting (both near [e:])
        "itacism_ei_oi": 0.05, # rare, ει and οι still distinct
        "itacism_oi_u": 0.1,   # οι/υ both rounded, some overlap (p.163)
        "ai_e_merger": 0.5,    # αι/ε well advanced by 4th c. BC (p.163, 168)
        "ai_e_merger_rev": 0.5,
        "o_omega": 0.3,        # vowel length starting to disappear (p.169)
        "o_omega_rev": 0.3,
        "iota_subscriptum": 0.3, # [i] lost from long diphthongs c.150-50 BC (p.164)
        "spirant_beta": 0.1,   # fricativization just beginning with /g/ (p.170)
        "aspiration_p_ph": 0.0, "aspiration_ph_p": 0.0,  # no evidence yet (p.170)
        "aspiration_t_th": 0.0, "aspiration_th_t": 0.0,
        "aspiration_k_kh": 0.0, "aspiration_kh_k": 0.0,
        "geminate_l": 0.15, "geminate_n": 0.15, "geminate_s": 0.15,  # from 3rd c. BC (p.171)
        "geminate_r": 0.1, "geminate_t": 0.1, "geminate_k": 0.1,
        "geminate_p": 0.1,
        "breathing": 0.05, # psilosis already in some dialects, but /h/ mostly retained
    },
    "late_antique": {
        # c. 1st-6th c. AD. All vowel mergers advancing; ο/ω well merged;
        # η approaching ι but not fully there; consonant changes in progress.
        # (Horrocks pp.167-171)
        "itacism_ei_i": 0.9,   # fully merged long ago (p.163)
        "itacism_eta_i": 0.5,  # η/ι interchange increasing but not yet general (p.168)
        "itacism_oi_i": 0.2,   # οι > /y/ complete by 1st c. AD, but ι merger
                                # only in middle Byzantine (p.163, 167)
        "itacism_u_i": 0.1,    # υ still /y/ for most speakers (p.169)
        "itacism_ei_eta": 0.4, # ει/η overlap as both approach [i] (p.167)
        "itacism_ei_oi": 0.15, # ει and οι starting to converge via [i]/[y]
        "itacism_oi_u": 0.3,   # both still rounded, /y/ = /y/ (p.167)
        "ai_e_merger": 0.7,    # fully merged (p.168)
        "ai_e_merger_rev": 0.7,
        "o_omega": 0.6,        # vowel length fully lost (p.169)
        "o_omega_rev": 0.6,
        "iota_subscriptum": 0.7, # universally lost in speech; omitted in papyri (p.175)
        "spirant_beta": 0.5,   # β = [v] by 1st c. AD, δ/γ advancing (p.170)
        "aspiration_p_ph": 0.2, "aspiration_ph_p": 0.2,  # some evidence for φ > [f]
        "aspiration_t_th": 0.15, "aspiration_th_t": 0.15, # beginning (p.170-171)
        "aspiration_k_kh": 0.15, "aspiration_kh_k": 0.15,
        "geminate_l": 0.3, "geminate_n": 0.3, "geminate_s": 0.25,  # well attested (p.171)
        "geminate_r": 0.15, "geminate_t": 0.15, "geminate_k": 0.15,
        "geminate_p": 0.15,
        "breathing": 0.3, # /h/ weakening, advanced in popular registers (p.171)
    },
    "byzantine": {
        # c. 7th-15th c. AD. All vowel mergers essentially complete; modern
        # Greek vowel system (i, e, a, o, u) in place. Full consonant
        # fricativization. (Horrocks pp.167-168, 171)
        "itacism_ei_i": 0.95,  # identical for centuries
        "itacism_eta_i": 0.9,  # η/ι merger complete in early Byzantine (p.167)
        "itacism_oi_i": 0.7,   # complete by middle Byzantine (p.163)
        "itacism_u_i": 0.4,    # υ/ι merger by 9th/10th c. (p.169); still
                                # in progress for early Byzantine
        "itacism_ei_eta": 0.7, # all three (ει, η, ι) now = [i]
        "itacism_ei_oi": 0.5,  # οι also = [i] by middle Byzantine
        "itacism_oi_u": 0.4,   # both still [i], but distinction only matters
                                # for earlier Byzantine when υ retained /y/
        "ai_e_merger": 0.8,    # fully merged (p.168)
        "ai_e_merger_rev": 0.8,
        "o_omega": 0.7,        # fully merged (p.169)
        "o_omega_rev": 0.7,
        "iota_subscriptum": 0.9, # purely orthographic convention by now (p.175)
        "spirant_beta": 0.7,   # β = [v] for centuries (p.170)
        "aspiration_p_ph": 0.5, "aspiration_ph_p": 0.5,  # full fricativization
        "aspiration_t_th": 0.4, "aspiration_th_t": 0.4,  # complete (p.171)
        "aspiration_k_kh": 0.4, "aspiration_kh_k": 0.4,
        "geminate_l": 0.5, "geminate_n": 0.5, "geminate_s": 0.4,
        "geminate_r": 0.3, "geminate_t": 0.3, "geminate_k": 0.3,
        "geminate_p": 0.3,
        "breathing": 0.6, # /h/ fully lost by now (p.171)
    },
}

# Default: all rules at moderate weights (vowel-only, conservative)
PROFILES["all"] = {
    k: min(0.7, max(v for p in PROFILES.values() for kk, v in p.items() if kk == k))
    for k in PROFILES["byzantine"]
}


class Normalizer:
    """Generate candidate normalized spellings for Greek tokens."""

    def __init__(self, period=None, max_candidates=50, max_substitutions=1):
        """
        Args:
            period: Historical period for rule weighting.
                    One of: archaic_epigraphic, hellenistic, late_antique,
                    byzantine, all (default).
            max_candidates: Hard cap on candidates per token.
            max_substitutions: Max simultaneous rule applications per candidate.
                    Default is 1 (single substitutions only) to avoid false
                    positives from over-eager double substitutions.
        """
        self.period = period or "all"
        self.profile = PROFILES.get(self.period, PROFILES["all"])
        self.max_candidates = max_candidates
        self.max_subs = max_substitutions

        # Consonant rules (spirantization, aspiration) are only active for
        # late_antique and byzantine periods, where the sound changes are
        # actually attested. For earlier periods and the default "all"
        # profile, consonant substitutions produce too many false positives
        # because these confusions are anachronistic.
        # (Horrocks pp.170-171: fricativization of voiced stops begins 2nd c.
        # BC but only complete by 4th c. AD; aspiration loss even later.)
        use_consonants = self.period in ("late_antique", "byzantine")

        # Build active rules sorted by weight (descending)
        self.vowel_rules = self._filter_rules(VOWEL_RULES)
        self.consonant_rules = (self._filter_rules(CONSONANT_RULES)
                                if use_consonants else [])
        self.all_rules = self.vowel_rules + self.consonant_rules

    def _filter_rules(self, rules):
        """Filter and sort rules by profile weight."""
        active = []
        for surface, normalized, rule_id in rules:
            weight = self.profile.get(rule_id, 0.0)
            if weight > 0:
                active.append((surface, normalized, rule_id, weight))
        return sorted(active, key=lambda x: -x[3])

    def normalize(self, token):
        """Generate candidate normalized forms for a token.

        Returns a list of candidate strings, ordered by likelihood:
        single substitutions first (sorted by rule weight), then doubles.
        The original token is NOT included (caller should check it first).
        """
        # Track candidates with their "cost" (lower = more likely)
        scored = {}  # candidate -> score

        # Minimum weight threshold: skip rules that are barely plausible
        # for this period. This prevents low-confidence substitutions from
        # generating false positives.
        MIN_WEIGHT = 0.1

        # Phase 1: Iota subscriptum restoration (most common single fix)
        sub_weight = self.profile.get("iota_subscriptum", 0.0)
        if sub_weight >= MIN_WEIGHT:
            for c in self._subscriptum_candidates(token):
                scored[c] = 1.0 - sub_weight  # lower score = better

        # Phase 2: Single substitution candidates
        for surface, normalized, rule_id, weight in self.all_rules:
            if weight < MIN_WEIGHT:
                continue
            for c in self._rule_candidates(token, surface, normalized):
                if c not in scored:
                    scored[c] = 1.0 - weight
            if len(scored) >= self.max_candidates:
                break

        # Phase 3: Double substitutions (if budget allows)
        # Only apply double subs using high-confidence rules (weight >= 0.4)
        # to avoid combinatorial explosion of unlikely candidates.
        if len(scored) < self.max_candidates and self.max_subs >= 2:
            high_conf_rules = [(s, n, r, w) for s, n, r, w in self.all_rules
                               if w >= 0.4]
            singles = sorted(scored.keys(), key=lambda c: scored[c])[:10]
            for single in singles:
                for surface, normalized, rule_id, weight in high_conf_rules[:6]:
                    for c in self._rule_candidates(single, surface, normalized):
                        if c not in scored:
                            scored[c] = 2.0 - weight * 0.5  # doubles rank lower
                    if len(scored) >= self.max_candidates:
                        break

        # Remove original
        scored.pop(token, None)

        # Sort by score (ascending = most likely first)
        return sorted(scored, key=lambda c: scored[c])[:self.max_candidates]

    def _subscriptum_candidates(self, token):
        """Try restoring iota subscriptum at each eligible position."""
        results = []

        # Find positions where subscriptum could be added
        positions = []
        for i, ch in enumerate(token):
            if ch in SUBSCRIPTUM_MAP:
                positions.append(i)

        # Single subscriptum additions
        for pos in positions:
            new = token[:pos] + SUBSCRIPTUM_MAP[token[pos]] + token[pos + 1:]
            results.append(new)

        # Also try REMOVING subscriptum (in case the token has one that's wrong)
        for i, ch in enumerate(token):
            if ch in SUBSCRIPTUM_REV:
                new = token[:i] + SUBSCRIPTUM_REV[ch] + token[i + 1:]
                results.append(new)

        return results

    def _rule_candidates(self, token, surface, normalized):
        """Apply a substitution rule at each occurrence, both directions."""
        results = []

        # Forward: replace surface with normalized
        start = 0
        while True:
            idx = token.find(surface, start)
            if idx == -1:
                break
            results.append(token[:idx] + normalized + token[idx + len(surface):])
            start = idx + 1

        # Reverse: replace normalized with surface
        start = 0
        while True:
            idx = token.find(normalized, start)
            if idx == -1:
                break
            results.append(token[:idx] + surface + token[idx + len(normalized):])
            start = idx + 1

        return results


def demo():
    """Demo the normalizer on sample Byzantine misspellings."""
    n = Normalizer(period="byzantine")

    test_cases = [
        ("θεω", "missing iota subscriptum (θεῷ)"),
        ("χεροντες", "itacism: ε for αι (χαίροντες)"),
        ("ξενι", "itacism: ι for η (ξένη or ξένοι)"),
        ("πιστι", "itacism: ι for ει (πίστει)"),
        ("κινος", "itacism: ι for υ (κυνός)"),
        ("ανθροπος", "o/ω confusion (ἄνθρωπος)"),
        ("εφτα", "aspiration: φ for π (ἑπτά)"),
        ("αλος", "geminate: λ for λλ (ἄλλος)"),
        ("τον", "iota sub: τόν or τῷν"),
    ]

    for token, description in test_cases:
        candidates = n.normalize(token)
        print(f"{token:20s} -> {candidates[:8]}")
        print(f"  ({description})")
        print()


if __name__ == "__main__":
    demo()
