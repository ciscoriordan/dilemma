#!/usr/bin/env python3
"""Orthographic normalizer for Greek texts with non-standard spelling.

Generates candidate normalized spellings for tokens exhibiting scribal
or epigraphic variation (itacism, vowel mergers, missing iota subscripta,
etc.) as well as dialect-specific forms (Ionic, Doric, Aeolic).
Candidates are meant to be checked against a lookup table.

Usage:
    from normalize import Normalizer

    n = Normalizer()                        # all periods
    n = Normalizer(period="byzantine")      # period-specific rules
    n = Normalizer(dialect="ionic")         # Ionic dialect rules
    n = Normalizer(dialect="auto")          # try all dialect rules
    n = Normalizer(period="hellenistic", dialect="ionic")  # combined

    candidates = n.normalize("χέροντες")    # itacism: ε for αι
    candidates = n.normalize("θεω")         # missing iota subscriptum
    candidates = n.normalize("ἱστορίης")    # Ionic: -ης -> -ας after ρ
    candidates = n.normalize("θάλασσα")     # Ionic σσ -> Attic ττ
"""

import re
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


# ---------------------------------------------------------------------------
# Dialect-specific rules
# ---------------------------------------------------------------------------
# These handle Ancient Greek dialect variation (Ionic, Doric, Aeolic) where
# the surface form differs systematically from the Attic standard that
# dominates the lookup table. Unlike the orthographic rules above (which
# handle scribal error in any period), these are genuine linguistic variants.

# Ionic dialect: whole-word and pattern-based substitutions
# Each entry: (ionic_form, attic_form)
IONIC_WORD_MAP = {
    # κ/π interchange in interrogatives/indefinites/relatives
    "κῶς": "πῶς",
    "κως": "πως",
    "ὅκου": "ὅπου",
    "οκου": "οπου",
    "κοῖος": "ποῖος",
    "κοιος": "ποιος",
    "κότε": "πότε",
    "κοτε": "ποτε",
    "κότερος": "πότερος",
    "κοτερος": "ποτερος",
    "κόσος": "πόσος",
    "κοσος": "ποσος",
    "κόθεν": "πόθεν",
    "κοθεν": "ποθεν",
    "κοῦ": "ποῦ",
    "κου": "που",
    "κόσε": "πόσε",
    "κοσε": "ποσε",
    "ὁκόσος": "ὁπόσος",
    "οκοσος": "οποσος",
    "ὅκως": "ὅπως",
    "οκως": "οπως",
    # ου/ο alternation (Ionic ου where Attic has ο)
    "μοῦνος": "μόνος",
    "μουνος": "μονος",
    "μούνη": "μόνη",
    "μουνη": "μονη",
    "μοῦνον": "μόνον",
    "μουνον": "μονον",
    # ξεῖνος / ξένος
    "ξεῖνος": "ξένος",
    "ξεινος": "ξενος",
    "ξείνη": "ξένη",
    "ξεινη": "ξενη",
    "ξεῖνον": "ξένον",
    "ξεινον": "ξενον",
    # κεῖνος / ἐκεῖνος
    "κεῖνος": "ἐκεῖνος",
    "κεινος": "εκεινος",
    "κείνη": "ἐκείνη",
    "κεινη": "εκεινη",
    "κεῖνο": "ἐκεῖνο",
    "κεινο": "εκεινο",
    "κείνου": "ἐκείνου",
    "κεινου": "εκεινου",
    "κείνῳ": "ἐκείνῳ",
    "κεινῳ": "εκεινῳ",
    "κείνων": "ἐκείνων",
    "κεινων": "εκεινων",
    # εἵνεκα / ἕνεκα
    "εἵνεκα": "ἕνεκα",
    "εινεκα": "ενεκα",
    "εἵνεκεν": "ἕνεκεν",
    "εινεκεν": "ενεκεν",
}

# Doric dialect: whole-word substitutions
DORIC_WORD_MAP = {
    # ποτί / πρός
    "ποτί": "πρός",
    "ποτι": "προς",
    # τύ / σύ
    "τύ": "σύ",
    "τυ": "συ",
    "τύν": "σέ",
    # Ἀθάνα / Ἀθήνη (and variants)
    "Ἀθάνα": "Ἀθήνη",
    "αθανα": "αθηνη",
    "Ἀθαναία": "Ἀθηναία",
    "αθαναια": "αθηναια",
}


# Helper: strip all diacritics from a Greek string for base-letter comparison
def _strip_all_diacritics(s):
    """Strip all combining marks (accents, breathings, iotas) from Greek."""
    nfd = unicodedata.normalize("NFD", s)
    base = "".join(c for c in nfd if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", base)


# Characters that count as vowels for context matching in dialect rules
_VOWELS = set("αεηιουωάέήίόύώὰὲὴὶὸὺὼᾶῆῖῦῶἀἐἠἰὀὐἠὠ"
               "ἁἑἡἱὁὑἡὡᾳῃῳᾴῄῴᾲῂῲᾷῇῷ")

# Characters that are vowel base letters (without diacritics)
_VOWEL_BASES = set("αεηιουω")


def _is_vowel(ch):
    """Check if a character is a Greek vowel (including accented forms)."""
    if ch in _VOWELS:
        return True
    base = _strip_all_diacritics(ch.lower())
    return base in _VOWEL_BASES


class Normalizer:
    """Generate candidate normalized spellings for Greek tokens."""

    # Valid dialect names
    VALID_DIALECTS = frozenset({
        "ionic", "doric", "aeolic", "koine", "auto", None
    })

    def __init__(self, period=None, dialect=None,
                 max_candidates=50, max_substitutions=1):
        """
        Args:
            period: Historical period for orthographic rule weighting.
                    One of: archaic_epigraphic, hellenistic, late_antique,
                    byzantine, all (default).
            dialect: Dialect normalization to apply. One of:
                    "ionic" - Ionic-to-Attic mappings (Herodotus, etc.)
                    "doric" - Doric-to-Attic mappings (Pindar, etc.)
                    "aeolic" - Aeolic-to-Attic mappings (Sappho, etc.)
                    "koine" - Koine normalization (overlaps with period rules)
                    "auto" - try all dialect rules
                    None (default) - no dialect normalization
            max_candidates: Hard cap on candidates per token.
            max_substitutions: Max simultaneous rule applications per candidate.
                    Default is 1 (single substitutions only) to avoid false
                    positives from over-eager double substitutions.
        """
        if dialect not in self.VALID_DIALECTS:
            raise ValueError(
                f"Unknown dialect {dialect!r}. "
                f"Valid values: {sorted(d for d in self.VALID_DIALECTS if d)}, "
                f"or None."
            )

        self.period = period or "all"
        self.dialect = dialect
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

        # Build active dialect set
        self._dialects = set()
        if dialect == "auto":
            self._dialects = {"ionic", "doric", "aeolic", "koine"}
        elif dialect is not None:
            self._dialects.add(dialect)

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
        dialect-specific forms first (high confidence), then single
        substitutions (sorted by rule weight), then doubles.
        The original token is NOT included (caller should check it first).
        """
        # Track candidates with their "cost" (lower = more likely)
        scored = {}  # candidate -> score

        # Minimum weight threshold: skip rules that are barely plausible
        # for this period. This prevents low-confidence substitutions from
        # generating false positives.
        MIN_WEIGHT = 0.1

        # Phase 0: Dialect-specific normalization (highest priority)
        if self._dialects:
            for c in self._dialect_candidates(token):
                scored[c] = 0.05  # dialect matches rank highest

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

    # ------------------------------------------------------------------
    # Dialect candidate generation
    # ------------------------------------------------------------------

    def _dialect_candidates(self, token):
        """Generate candidates from dialect-specific transformations."""
        results = []
        token_lower = token.lower()
        token_stripped = _strip_all_diacritics(token_lower)

        if "ionic" in self._dialects:
            results.extend(self._ionic_candidates(token, token_lower,
                                                   token_stripped))

        if "doric" in self._dialects:
            results.extend(self._doric_candidates(token, token_lower,
                                                   token_stripped))

        if "aeolic" in self._dialects:
            results.extend(self._aeolic_candidates(token, token_lower,
                                                    token_stripped))

        if "koine" in self._dialects:
            results.extend(self._koine_candidates(token, token_lower,
                                                   token_stripped))

        return results

    def _ionic_candidates(self, token, token_lower, token_stripped):
        """Ionic dialect -> Attic normalization.

        Covers:
        - Whole-word mappings (interrogatives, common words)
        - η -> ᾱ after ε, ι, ρ (first-declension nouns)
        - Uncontracted vowels (ε-contract verbs)
        - σσ -> ττ alternation
        - ρσ -> ρρ alternation
        """
        results = []

        # 1. Whole-word lookup (exact, lowercase, stripped)
        for form in (token, token_lower, token_stripped):
            if form in IONIC_WORD_MAP:
                results.append(IONIC_WORD_MAP[form])

        # 2. Ionic η -> Attic ᾱ after ε, ι, ρ in endings
        # This is the first-declension pattern: -ης/-ης/-ῃ/-ην etc.
        # become -ας/-ας/-ᾳ/-αν in Attic after ε, ι, ρ.
        # We try replacing final-syllable η with α (preserving accents).
        results.extend(self._ionic_eta_to_alpha(token))

        # 3. Uncontracted vowels -> contracted
        # ε-contract verbs: -εε- -> -ει-, -εο- -> -ου-, -εω -> -ω
        results.extend(self._ionic_contraction(token))

        # 4. σσ -> ττ (Ionic/Koine σσ = Attic ττ)
        results.extend(self._replace_all_occurrences(token, "σσ", "ττ"))

        # 5. ρσ -> ρρ (θάρσος -> θάρρος, ἄρσην -> ἄρρην)
        results.extend(self._replace_all_occurrences(token, "ρσ", "ρρ"))

        return results

    def _ionic_eta_to_alpha(self, token):
        """Replace η with α in endings after ε, ι, ρ (Ionic -> Attic).

        Handles accented variants: ή->ά, ῆ->ᾶ, ῇ->ᾷ, ὴ->ὰ, ῃ->ᾳ.
        Only applies when preceded by ε, ι, or ρ (possibly with
        intervening diacritics).
        """
        results = []
        # Map η-family to α-family (preserving accent type)
        eta_to_alpha = {
            "η": "α", "ή": "ά", "ῆ": "ᾶ", "ὴ": "ὰ",
            "ῃ": "ᾳ", "ῄ": "ᾴ", "ῇ": "ᾷ", "ῂ": "ᾲ",
        }
        # Also handle breathed forms
        eta_to_alpha.update({
            "ἡ": "ἁ", "ἥ": "ἅ", "ἧ": "ἇ", "ἣ": "ἃ",
            "ἠ": "ἀ", "ἤ": "ἄ", "ἦ": "ἆ", "ἢ": "ἂ",
        })

        # Preceding context letters (base forms)
        context_chars = set("ειρέίρὲὶεἐἑἔἕἒἓἰἱἴἵἲἳἶἷ")

        for i, ch in enumerate(token):
            if ch in eta_to_alpha and i > 0:
                # Check if preceded by ε, ι, or ρ
                prev = token[i - 1]
                prev_base = _strip_all_diacritics(prev.lower())
                if prev_base in ("ε", "ι", "ρ"):
                    new = token[:i] + eta_to_alpha[ch] + token[i + 1:]
                    results.append(new)

        return results

    def _ionic_contraction(self, token):
        """Handle Ionic uncontracted vowels -> Attic contracted forms.

        Patterns:
        - εε -> ει (ποιέεσθαι -> ποιεῖσθαι)
        - εο -> ου (when not word-initial)
        - εω -> ω (τιμέω -> τιμῶ) - only at word end or before consonant
        - εει -> ει
        - εου -> ου
        """
        results = []

        # Simple substring replacements for uncontracted -> contracted
        contraction_pairs = [
            ("εει", "ει"),
            ("εου", "ου"),
            ("εε", "ει"),
            ("εο", "ου"),
        ]

        for ionic, attic in contraction_pairs:
            results.extend(self._replace_all_occurrences(token, ionic, attic))

        # εω -> ω at word end (verb infinitive/1sg endings)
        if token.endswith("εω"):
            results.append(token[:-2] + "ω")
        if token.endswith("έω"):
            results.append(token[:-2] + "ῶ")
        # Also handle εων -> ων (genitive plural)
        if "εων" in token or "έων" in token or "εών" in token:
            results.extend(self._replace_all_occurrences(token, "εων", "ων"))
            results.extend(self._replace_all_occurrences(token, "έων", "ῶν"))
            results.extend(self._replace_all_occurrences(token, "εών", "ών"))

        return results

    def _doric_candidates(self, token, token_lower, token_stripped):
        """Doric dialect -> Attic normalization.

        Covers:
        - Whole-word mappings
        - Doric ᾱ -> Attic η
        - Doric futures in -σέω -> Attic -σω
        """
        results = []

        # 1. Whole-word lookup
        for form in (token, token_lower, token_stripped):
            if form in DORIC_WORD_MAP:
                results.append(DORIC_WORD_MAP[form])

        # 2. Doric ᾱ -> Attic η (reverse of Ionic η -> ᾱ)
        # This is tricky because Greek text doesn't mark vowel length.
        # We conservatively try α -> η substitution only in common
        # grammatical endings where the alternation is predictable:
        # -ας (gen.sg) -> -ης, -αν (acc.sg) -> -ην,
        # -α (nom.sg) -> -η, -ᾳ (dat.sg) -> -ῃ
        results.extend(self._doric_alpha_to_eta(token))

        # 3. Doric futures: -σεω -> -σω, -σέω -> -σῶ
        if token_lower.endswith("σεω"):
            results.append(token[:-3] + "σω")
        if token_lower.endswith("σέω"):
            results.append(token[:-3] + "σῶ")
        if token_lower.endswith("ξεω"):
            results.append(token[:-3] + "ξω")
        if token_lower.endswith("ξέω"):
            results.append(token[:-3] + "ξῶ")

        return results

    def _doric_alpha_to_eta(self, token):
        """Replace Doric long alpha with Attic eta in common endings.

        Only targets final-syllable α where Doric/Attic systematically
        differ. Conservative to avoid false positives.
        """
        results = []

        # Map α-family to η-family
        alpha_to_eta = {
            "α": "η", "ά": "ή", "ᾶ": "ῆ", "ὰ": "ὴ",
            "ᾳ": "ῃ", "ᾴ": "ῄ", "ᾷ": "ῇ", "ᾲ": "ῂ",
            "ἁ": "ἡ", "ἅ": "ἥ", "ἇ": "ἧ", "ἃ": "ἣ",
            "ἀ": "ἠ", "ἄ": "ἤ", "ἆ": "ἦ", "ἂ": "ἢ",
        }

        # Only try replacing the LAST alpha-like vowel in the word
        # (most Doric/Attic alternation is in endings)
        for i in range(len(token) - 1, -1, -1):
            if token[i] in alpha_to_eta:
                new = token[:i] + alpha_to_eta[token[i]] + token[i + 1:]
                results.append(new)
                break  # only the last one

        return results

    def _aeolic_candidates(self, token, token_lower, token_stripped):
        """Aeolic dialect -> Attic normalization.

        Aeolic features are harder to normalize systematically because
        many involve morphological rather than phonological differences.
        We handle the most common patterns:
        - Labial treatment of labiovelars (π where Attic has τ/κ in some words)
        - Initial πτ- for Attic πτ- (already same, skip)
        """
        results = []

        # Aeolic has limited systematic orthographic differences that can
        # be handled by simple substitution. Most Aeolic variation is
        # morphological (different verb endings, etc.) which is better
        # handled by the lookup table directly.

        # Psilosis: loss of rough breathing. In polytonic text, we can
        # try adding rough breathing to smooth-breathed initial vowels.
        # This is worth trying since the lookup table expects standard
        # Attic breathing.
        results.extend(self._try_add_rough_breathing(token))

        return results

    def _try_add_rough_breathing(self, token):
        """Try replacing smooth breathing with rough breathing on initial vowel.

        Handles Aeolic/Ionic psilosis where the text may lack aspiration
        that the Attic-based lookup expects.
        """
        results = []
        if not token:
            return results

        # Map smooth-breathed initial vowels to rough-breathed equivalents
        smooth_to_rough = {
            "ἀ": "ἁ", "ἄ": "ἅ", "ἆ": "ἇ", "ἂ": "ἃ",
            "ἐ": "ἑ", "ἔ": "ἕ", "ἒ": "ἓ",
            "ἠ": "ἡ", "ἤ": "ἥ", "ἦ": "ἧ", "ἢ": "ἣ",
            "ἰ": "ἱ", "ἴ": "ἵ", "ἶ": "ἷ", "ἲ": "ἳ",
            "ὀ": "ὁ", "ὄ": "ὅ", "ὂ": "ὃ",
            "ὐ": "ὑ", "ὔ": "ὕ", "ὖ": "ὗ", "ὒ": "ὓ",
            "ὠ": "ὡ", "ὤ": "ὥ", "ὦ": "ὧ", "ὢ": "ὣ",
        }

        first = token[0]
        if first in smooth_to_rough:
            results.append(smooth_to_rough[first] + token[1:])

        return results

    def _koine_candidates(self, token, token_lower, token_stripped):
        """Koine normalization.

        Koine largely overlaps with the period-based rules (itacism,
        αι/ε merger, etc.) already handled by the main rule engine.
        Here we handle a few Koine-specific patterns not covered elsewhere:
        - σσ/ττ alternation (Koine retains σσ like Ionic)
        - αι/ε interchange (already in main rules, but ensure it fires)
        """
        results = []

        # σσ -> ττ (Koine inherits Ionic σσ, lookup may expect Attic ττ)
        results.extend(self._replace_all_occurrences(token, "σσ", "ττ"))

        # ττ -> σσ (some forms may be Atticized in text but Koine in lookup)
        results.extend(self._replace_all_occurrences(token, "ττ", "σσ"))

        return results

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _replace_all_occurrences(self, token, old, new):
        """Replace each occurrence of `old` with `new`, one at a time."""
        results = []
        start = 0
        while True:
            idx = token.find(old, start)
            if idx == -1:
                break
            results.append(token[:idx] + new + token[idx + len(old):])
            start = idx + 1
        return results

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
    """Demo the normalizer on sample Byzantine and dialect forms."""
    print("=== Byzantine orthographic normalization ===\n")
    n = Normalizer(period="byzantine")

    test_cases = [
        ("θεω", "missing iota subscriptum (-> θεῷ)"),
        ("χεροντες", "itacism: ε for αι (-> χαίροντες)"),
        ("ξενι", "itacism: ι for η (-> ξένη or ξένοι)"),
        ("πιστι", "itacism: ι for ει (-> πίστει)"),
        ("κινος", "itacism: ι for υ (-> κυνός)"),
        ("ανθροπος", "o/ω confusion (-> ἄνθρωπος)"),
        ("εφτα", "aspiration: φ for π (-> ἑπτά)"),
        ("αλος", "geminate: λ for λλ (-> ἄλλος)"),
        ("τον", "iota sub: τόν or τῷν"),
    ]

    for token, description in test_cases:
        candidates = n.normalize(token)
        print(f"{token:20s} -> {candidates[:8]}")
        print(f"  ({description})")
        print()

    print("\n=== Ionic dialect normalization ===\n")
    n_ionic = Normalizer(dialect="ionic")

    ionic_cases = [
        ("ἱστορίης", "Ionic -ης -> Attic -ας after ρ"),
        ("χώρης", "Ionic -ης -> Attic -ας after ρ"),
        ("ποιέειν", "uncontracted: -εει- -> -ει-"),
        ("τιμέω", "uncontracted: -εω -> -ω"),
        ("θάλασσα", "Ionic σσ -> Attic ττ"),
        ("θάρσος", "Ionic ρσ -> Attic ρρ"),
        ("κῶς", "Ionic κ- -> Attic π- interrogative"),
        ("μοῦνος", "Ionic μοῦνος -> Attic μόνος"),
        ("ξεῖνος", "Ionic ξεῖνος -> Attic ξένος"),
    ]

    for token, description in ionic_cases:
        candidates = n_ionic.normalize(token)
        print(f"{token:20s} -> {candidates[:8]}")
        print(f"  ({description})")
        print()

    print("\n=== Doric dialect normalization ===\n")
    n_doric = Normalizer(dialect="doric")

    doric_cases = [
        ("ποτί", "Doric ποτί -> Attic πρός"),
        ("τύ", "Doric τύ -> Attic σύ"),
        ("Ἀθάνα", "Doric Ἀθάνα -> Attic Ἀθήνη"),
    ]

    for token, description in doric_cases:
        candidates = n_doric.normalize(token)
        print(f"{token:20s} -> {candidates[:8]}")
        print(f"  ({description})")
        print()

    print("\n=== Auto mode (all dialects) ===\n")
    n_auto = Normalizer(dialect="auto")

    auto_cases = [
        ("ἱστορίης", "Ionic -> Attic"),
        ("ποτί", "Doric -> Attic"),
        ("θάλασσα", "σσ -> ττ"),
    ]

    for token, description in auto_cases:
        candidates = n_auto.normalize(token)
        print(f"{token:20s} -> {candidates[:8]}")
        print(f"  ({description})")
        print()


if __name__ == "__main__":
    demo()
