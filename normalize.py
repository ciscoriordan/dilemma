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

PROFILES = {
    "archaic_epigraphic": {
        "itacism_ei_i": 0.0, "itacism_eta_i": 0.0, "itacism_oi_i": 0.0,
        "itacism_u_i": 0.0, "itacism_ei_eta": 0.0, "itacism_ei_oi": 0.0,
        "itacism_oi_u": 0.0,
        "ai_e_merger": 0.0, "ai_e_merger_rev": 0.0,
        "o_omega": 0.05, "o_omega_rev": 0.05,
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
        "itacism_ei_i": 0.6, "itacism_eta_i": 0.3, "itacism_oi_i": 0.1,
        "itacism_u_i": 0.05, "itacism_ei_eta": 0.3, "itacism_ei_oi": 0.1,
        "itacism_oi_u": 0.1,
        "ai_e_merger": 0.3, "ai_e_merger_rev": 0.3,
        "o_omega": 0.3, "o_omega_rev": 0.3,
        "iota_subscriptum": 0.2,
        "spirant_beta": 0.2,
        "aspiration_p_ph": 0.1, "aspiration_ph_p": 0.1,
        "aspiration_t_th": 0.1, "aspiration_th_t": 0.1,
        "aspiration_k_kh": 0.1, "aspiration_kh_k": 0.1,
        "geminate_l": 0.1, "geminate_n": 0.1, "geminate_s": 0.1,
        "geminate_r": 0.05, "geminate_t": 0.05, "geminate_k": 0.05,
        "geminate_p": 0.05,
        "breathing": 0.1,
    },
    "late_antique": {
        "itacism_ei_i": 0.8, "itacism_eta_i": 0.7, "itacism_oi_i": 0.5,
        "itacism_u_i": 0.2, "itacism_ei_eta": 0.5, "itacism_ei_oi": 0.3,
        "itacism_oi_u": 0.3,
        "ai_e_merger": 0.6, "ai_e_merger_rev": 0.6,
        "o_omega": 0.5, "o_omega_rev": 0.5,
        "iota_subscriptum": 0.5,
        "spirant_beta": 0.5,
        "aspiration_p_ph": 0.3, "aspiration_ph_p": 0.3,
        "aspiration_t_th": 0.2, "aspiration_th_t": 0.2,
        "aspiration_k_kh": 0.2, "aspiration_kh_k": 0.2,
        "geminate_l": 0.3, "geminate_n": 0.3, "geminate_s": 0.2,
        "geminate_r": 0.1, "geminate_t": 0.1, "geminate_k": 0.1,
        "geminate_p": 0.1,
        "breathing": 0.3,
    },
    "byzantine": {
        "itacism_ei_i": 0.95, "itacism_eta_i": 0.9, "itacism_oi_i": 0.8,
        "itacism_u_i": 0.5, "itacism_ei_eta": 0.7, "itacism_ei_oi": 0.5,
        "itacism_oi_u": 0.5,
        "ai_e_merger": 0.8, "ai_e_merger_rev": 0.8,
        "o_omega": 0.7, "o_omega_rev": 0.7,
        "iota_subscriptum": 0.9,
        "spirant_beta": 0.7,
        "aspiration_p_ph": 0.5, "aspiration_ph_p": 0.5,
        "aspiration_t_th": 0.4, "aspiration_th_t": 0.4,
        "aspiration_k_kh": 0.4, "aspiration_kh_k": 0.4,
        "geminate_l": 0.5, "geminate_n": 0.5, "geminate_s": 0.4,
        "geminate_r": 0.3, "geminate_t": 0.3, "geminate_k": 0.3,
        "geminate_p": 0.3,
        "breathing": 0.6,
    },
}

# Default: all rules at moderate weights
PROFILES["all"] = {
    k: min(0.7, max(v for p in PROFILES.values() for kk, v in p.items() if kk == k))
    for k in PROFILES["byzantine"]
}


class Normalizer:
    """Generate candidate normalized spellings for Greek tokens."""

    def __init__(self, period=None, max_candidates=50, max_substitutions=2):
        """
        Args:
            period: Historical period for rule weighting.
                    One of: archaic_epigraphic, hellenistic, late_antique,
                    byzantine, all (default).
            max_candidates: Hard cap on candidates per token.
            max_substitutions: Max simultaneous rule applications per candidate.
        """
        self.profile = PROFILES.get(period or "all", PROFILES["all"])
        self.max_candidates = max_candidates
        self.max_subs = max_substitutions

        # Build active rules sorted by weight (descending)
        self.vowel_rules = self._filter_rules(VOWEL_RULES)
        self.consonant_rules = self._filter_rules(CONSONANT_RULES)
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

        # Phase 1: Iota subscriptum restoration (most common single fix)
        sub_weight = self.profile.get("iota_subscriptum", 0.0)
        if sub_weight > 0:
            for c in self._subscriptum_candidates(token):
                scored[c] = 1.0 - sub_weight  # lower score = better

        # Phase 2: Single substitution candidates
        for surface, normalized, rule_id, weight in self.all_rules:
            for c in self._rule_candidates(token, surface, normalized):
                if c not in scored:
                    scored[c] = 1.0 - weight
            if len(scored) >= self.max_candidates:
                break

        # Phase 3: Double substitutions (if budget allows)
        if len(scored) < self.max_candidates and self.max_subs >= 2:
            singles = sorted(scored.keys(), key=lambda c: scored[c])[:15]
            for single in singles:
                for surface, normalized, rule_id, weight in self.all_rules[:8]:
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
