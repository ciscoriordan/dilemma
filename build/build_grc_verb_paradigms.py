#!/usr/bin/env python3
"""Build per-lemma Ancient Greek verb paradigms from dilemma's tagged pairs.

Aggregates tagged form->lemma pairs (kaikki + GLAUx + LSJ expansion) into
per-lemma paradigm dicts keyed by `<voice>_<tense>_<mood>_<person><number>`,
with infinitives keyed `<voice>_<tense>_infinitive` and participle nominative
forms keyed `<voice>_<tense>_participle_nom_<gender>_sg`. Output schema
matches `jtauber_ag_paradigms.json` so Klisy's `build_canonical_ag.rb` can
consume the file as a drop-in alternative or supplement.

Inputs (from data/):
  - ag_pairs.json     Wiktionary kaikki form-lemma pairs (now tense-tagged)
  - glaux_pairs.json  GLAUx corpus, fully morph-tagged
  - verb_extra_pairs.json (optional) pairs from LSJ expansion

Output:
  - data/ag_verb_paradigms.json    {lemma: {forms: {key: form},
                                            form_count: N,
                                            source: "dilemma"}}

Dialect handling: the default Attic forms (no dialect tag, or explicit
"Attic") populate the top-level paradigm. Forms tagged with one of
{Epic, Ionic, Doric, Aeolic, Koine, ...} are emitted under
`forms.<dialect_lower>` as a parallel paradigm slice. The Klisy consumer
(`build_canonical_ag.rb`) currently only reads the Attic slice; we keep
the dialect data so it can pick it up later without rebuilding.

Usage:
  python build/build_grc_verb_paradigms.py
  python build/build_grc_verb_paradigms.py --sanity   # 5-lemma smoke test
  python build/build_grc_verb_paradigms.py --only γράφω,τίθημι
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"

# Make sibling build/ modules importable when this file is run as a
# script (`python build/build_grc_verb_paradigms.py`).
_BUILD_DIR = Path(__file__).resolve().parent
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

AG_PAIRS = DATA_DIR / "ag_pairs.json"
GLAUX_PAIRS = DATA_DIR / "glaux_pairs.json"
VERB_EXTRA_PAIRS = DATA_DIR / "verb_extra_pairs.json"
LSJ_VERB_PAIRS = DATA_DIR / "ag_lsj_verb_pairs.json"  # produced by expand_lsj
LSJ_HEADWORDS_PATH = DATA_DIR / "lsj_headwords.json"
LSJ9_GLOSSES = Path.home() / "Documents" / "lsj9" / "lsj9_glosses.jsonl"
OUT_PATH = DATA_DIR / "ag_verb_paradigms.json"


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if not unicodedata.combining(c)))

# Tag vocabulary - matches dilemma/Wiktionary tag strings
TENSE_TAGS = {
    "present", "imperfect", "future", "aorist",
    "perfect", "pluperfect", "future-perfect",
}
VOICE_TAGS = {"active", "middle", "passive", "mediopassive"}
MOOD_TAGS = {
    "indicative", "subjunctive", "optative", "imperative",
    "infinitive", "participle",
}
PERSON_TAGS = {"first-person", "second-person", "third-person"}
NUMBER_TAGS = {"singular", "plural", "dual"}
CASE_TAGS = {"nominative", "genitive", "dative", "accusative", "vocative"}
GENDER_TAGS = {"masculine", "feminine", "neuter"}
DIALECT_TAGS = {
    "Attic", "Epic", "Ionic", "Doric", "Koine", "Aeolic",
    "Homeric", "Laconian", "Boeotian", "Arcadocypriot",
}

NUMBER_SHORT = {"singular": "sg", "plural": "pl", "dual": "du"}
PERSON_SHORT = {"first-person": "1", "second-person": "2", "third-person": "3"}
CASE_SHORT = {
    "nominative": "nom", "genitive": "gen", "dative": "dat",
    "accusative": "acc", "vocative": "voc",
}
GENDER_SHORT = {"masculine": "m", "feminine": "f", "neuter": "n"}


def grave_to_acute(s: str) -> str:
    """Convert combining grave (U+0300) to combining acute (U+0301).

    For citation forms the acute is canonical; the grave only appears
    mid-sentence to avoid stacking acutes.
    """
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", nfd.replace("̀", "́"))


def has_polytonic(s: str) -> bool:
    nfd = unicodedata.normalize("NFD", s)
    return any(c in nfd for c in ("̓", "̔", "͂"))


def is_stripped(s: str) -> bool:
    nfd = unicodedata.normalize("NFD", s)
    return not any(unicodedata.combining(c) for c in nfd)


def is_elided(s: str) -> bool:
    return bool(s) and s[-1] in ("'", "’", "ʼ", "᾽", "ʹ")


_GREEK_RANGES = [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]
_GREEK_DIACRITICS_RANGE = (0x0300, 0x036F)


def is_pure_greek(s: str) -> bool:
    """True if every character is a Greek letter / diacritic.

    Forms / lemmas that contain non-Greek punctuation (hyphens between
    LSJ compound prefixes, modifier apostrophes from old OCR scans,
    digits, Latin letters, etc.) are filtered out. dilemma's downstream
    consumers expect bare Greek tokens.
    """
    if not s:
        return False
    for c in s:
        cp = ord(c)
        if not any(lo <= cp <= hi for lo, hi in _GREEK_RANGES):
            return False
    return True


def has_internal_capital(s: str) -> bool:
    """True when the lemma has a capital letter past position 0.

    Verbs are essentially never proper nouns, and capitals after the
    first character are a strong signal of a corpus annotation glitch
    (ΒΕἔστημι), prefixed table cells leaking into a lemma slot, or
    accidentally-spliced strings. We drop these entirely rather than
    try to canonicalize.
    """
    nfd = unicodedata.normalize("NFD", s)
    chars = [c for c in nfd if not unicodedata.combining(c)]
    if not chars:
        return False
    for c in chars[1:]:
        if c.isupper():
            return True
    return False


def lowercase_initial(s: str) -> str:
    """Lowercase only the first letter; preserve diacritics on the
    underlying letter via NFD/NFC roundtrip. Used to normalize verb
    lemmas like Τίκτω -> τίκτω where the corpus capitalized a sentence-
    initial token."""
    if not s:
        return s
    nfd = unicodedata.normalize("NFD", s)
    out = []
    seen_letter = False
    for c in nfd:
        if not seen_letter and not unicodedata.combining(c) and c.isalpha():
            out.append(c.lower())
            seen_letter = True
        else:
            out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


def pick_best_form(forms):
    """Pick the canonical surface form from a list of variants."""
    if not forms:
        return None
    if isinstance(forms, set):
        forms = list(forms)
    polyt = [f for f in forms if has_polytonic(f)]
    diacr = [f for f in forms if not is_stripped(f)]
    pool = polyt or diacr or list(forms)
    no_elide = [f for f in pool if not is_elided(f)]
    if no_elide:
        pool = no_elide
    counts = Counter(forms)
    return max(pool, key=lambda f: (
        counts[f],          # most attested wins
        has_polytonic(f),   # break ties by polytonic richness
        -len(f),            # shorter wins (ἐστί over ἐστίν)
        f,                  # alphabetical for determinism
    ))


def verb_key_from_tags(tags):
    """Convert a tag set to a paradigm key.

    Returns None if the tags are not a fully-specified finite verb /
    infinitive / participle cell.
    """
    tags = set(tags)
    voice = next(iter(tags & VOICE_TAGS), None)
    tense = next(iter(tags & TENSE_TAGS), None)
    mood = next(iter(tags & MOOD_TAGS), None)
    if not voice or not tense or not mood:
        return None
    if "mediopassive" in tags:
        voice = "middle"

    if mood == "infinitive":
        return f"{voice}_{tense}_infinitive"

    if mood == "participle":
        case = next(iter(tags & CASE_TAGS), None)
        gender = next(iter(tags & GENDER_TAGS), None)
        number = next(iter(tags & NUMBER_TAGS), None)
        # Wiktionary tables usually give only the nom-sg of each gender;
        # corpus data (GLAUx) carries full case+number on participle forms.
        # Default to nom-sg when only gender is present (Wiktionary case).
        if gender and not case:
            case = "nominative"
        if gender and not number:
            number = "singular"
        if not (case and gender and number):
            return None
        return (
            f"{voice}_{tense}_participle_"
            f"{CASE_SHORT[case]}_{GENDER_SHORT[gender]}_{NUMBER_SHORT[number]}"
        )

    person = next(iter(tags & PERSON_TAGS), None)
    number = next(iter(tags & NUMBER_TAGS), None)
    if not person or not number:
        return None
    # Imperative has no first person. kaikki sometimes mistags Koine
    # alternatives like λυσάτωσαν as 1sg (Wiktionary's 'type-a/type-b'
    # appendix rows leak into the 1sg slot). Drop these to avoid
    # corrupting the paradigm.
    if mood == "imperative" and person == "first-person":
        return None
    return (
        f"{voice}_{tense}_{mood}_"
        f"{PERSON_SHORT[person]}{NUMBER_SHORT[number]}"
    )


def extract_dialect(tags):
    """Return the (single) dialect this form belongs to, or '' for Attic/default."""
    tags = set(tags)
    for d in DIALECT_TAGS:
        if d in tags:
            if d in ("Attic", "Homeric"):
                # Treat Attic as the default slice. Homeric is alias for Epic
                # and we fold it in.
                return "" if d == "Attic" else "epic"
            return d.lower()
    return ""


def load_pairs(path: Path):
    if not path.exists():
        print(f"  skipping {path.name} (not present)", flush=True)
        return []
    print(f"  loading {path.name} ...", flush=True)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_lsj_head_texts() -> dict:
    """Load the leading paragraph of every LSJ entry (the gloss without
    `level`/`number` is the entry head, which carries the principal-
    parts header before the English definition starts).

    Returns a dict ``{headword: head_text}``. Empty dict if the LSJ9
    glosses file is unavailable.
    """
    heads: dict[str, str] = {}
    if not LSJ9_GLOSSES.exists():
        print(f"  lsj9 glosses not found at {LSJ9_GLOSSES}; "
              f"principal-parts synthesis disabled")
        return heads
    print(f"  loading lsj9 head texts from {LSJ9_GLOSSES.name} ...",
          flush=True)
    with open(LSJ9_GLOSSES, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            hw = e.get("headword")
            if not hw:
                continue
            if "level" in e or "number" in e:
                continue
            if hw not in heads:
                heads[hw] = e.get("text", "")
    print(f"  lsj9 head texts: {len(heads):,}")
    return heads


def synthesize_missing_moods(results: dict) -> tuple[int, int]:
    """Fill in missing finite-mood cells via principal-parts templating.

    For each verb in ``results``, parse its LSJ head text into
    principal parts and run them through
    ``synth_verb_moods.synthesize_active_moods`` and
    ``synth_verb_moods.synthesize_mp_moods`` to produce templated
    subjunctive / optative / imperative / aorist-infinitive forms in
    active, middle, and passive voices. Only writes into slots that are
    currently empty; real corpus / Wiktionary cells are never overwritten.

    Returns ``(verbs_touched, cells_added)``.
    """
    try:
        from synth_verb_moods import (
            synthesize_active_moods,
            synthesize_mp_moods,
            synthesize_aor2_moods,
            synthesize_contract_moods,
        )
        from lsj_principal_parts import parse_principal_parts
    except ImportError as e:
        print(f"  synthesis skipped (import failure: {e})")
        return 0, 0

    head_texts = load_lsj_head_texts()
    verbs_touched = 0
    cells_added = 0
    mp_cells_added = 0
    aor2_cells_added = 0
    contract_cells_added = 0
    cells_skipped_overlap = 0
    for lemma, paradigm in results.items():
        head_text = head_texts.get(lemma, "")
        try:
            parts = parse_principal_parts(head_text, lemma) if head_text else {}
        except Exception:
            parts = {}
        try:
            templated = synthesize_active_moods(lemma, parts)
        except Exception:
            templated = {}
        try:
            templated_mp = synthesize_mp_moods(lemma, parts)
        except Exception:
            templated_mp = {}
        try:
            templated_aor2 = synthesize_aor2_moods(lemma, parts)
        except Exception:
            templated_aor2 = {}
        try:
            templated_contract = synthesize_contract_moods(lemma, parts)
        except Exception:
            templated_contract = {}
        # Track per-voice to report stats; merge for the actual write.
        if not (templated or templated_mp or templated_aor2 or templated_contract):
            continue
        forms = paradigm.setdefault("forms", {})
        added = 0
        added_mp = 0
        added_aor2 = 0
        added_contract = 0
        for key, val in templated.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added += 1
        for key, val in templated_mp.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_mp += 1
        for key, val in templated_aor2.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_aor2 += 1
        for key, val in templated_contract.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_contract += 1
        if added or added_mp or added_aor2 or added_contract:
            verbs_touched += 1
            cells_added += added
            mp_cells_added += added_mp
            aor2_cells_added += added_aor2
            contract_cells_added += added_contract
            paradigm["form_count"] = len(forms)
    print(f"  synthesised active cells: {cells_added:,} across "
          f"{verbs_touched:,} verbs")
    print(f"  synthesised mp/passive cells: {mp_cells_added:,}")
    print(f"  synthesised aor-2 cells: {aor2_cells_added:,}")
    print(f"  synthesised contract cells: {contract_cells_added:,}")
    print(f"  cells skipped (already present): {cells_skipped_overlap:,}")
    return verbs_touched, (
        cells_added + mp_cells_added + aor2_cells_added + contract_cells_added
    )


def synthesize_missing_participles(results: dict) -> tuple[int, int]:
    """Fill in missing participle cells via principal-parts templating.

    For each verb in ``results``, parse its LSJ head text into principal
    parts and run them through
    ``synth_verb_participles.synthesize_participles`` to produce the
    full case×gender×number declension for present-active /
    present-mp / future-active / future-middle / future-passive /
    aorist-active / aorist-middle / aorist-passive / perfect-active /
    perfect-mp participles. Only writes into slots that are currently
    empty; real corpus / Wiktionary cells are never overwritten.

    Returns ``(verbs_touched, cells_added)``.
    """
    try:
        from synth_verb_participles import (
            synthesize_participles,
            synthesize_aor2_participles,
            synthesize_contract_participles,
        )
        from lsj_principal_parts import parse_principal_parts
    except ImportError as e:
        print(f"  participle synthesis skipped (import failure: {e})")
        return 0, 0

    head_texts = load_lsj_head_texts()
    verbs_touched = 0
    cells_added = 0
    aor2_cells_added = 0
    contract_cells_added = 0
    cells_skipped_overlap = 0
    for lemma, paradigm in results.items():
        head_text = head_texts.get(lemma, "")
        try:
            parts = parse_principal_parts(head_text, lemma) if head_text else {}
        except Exception:
            parts = {}
        try:
            templated = synthesize_participles(lemma, parts)
        except Exception:
            templated = {}
        try:
            templated_aor2 = synthesize_aor2_participles(lemma, parts)
        except Exception:
            templated_aor2 = {}
        try:
            templated_contract = synthesize_contract_participles(lemma, parts)
        except Exception:
            templated_contract = {}
        if not (templated or templated_aor2 or templated_contract):
            continue
        forms = paradigm.setdefault("forms", {})
        added = 0
        added_aor2 = 0
        added_contract = 0
        for key, val in templated.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added += 1
        for key, val in templated_aor2.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_aor2 += 1
        for key, val in templated_contract.items():
            if key in forms:
                cells_skipped_overlap += 1
                continue
            forms[key] = val
            added_contract += 1
        if added or added_aor2 or added_contract:
            verbs_touched += 1
            cells_added += added
            aor2_cells_added += added_aor2
            contract_cells_added += added_contract
            paradigm["form_count"] = len(forms)
    print(f"  synthesised participle cells: {cells_added:,} across "
          f"{verbs_touched:,} verbs")
    print(f"  synthesised aor-2 participle cells: {aor2_cells_added:,}")
    print(f"  synthesised contract participle cells: {contract_cells_added:,}")
    print(f"  participle cells skipped (already present): "
          f"{cells_skipped_overlap:,}")
    return verbs_touched, cells_added + aor2_cells_added + contract_cells_added


def build_paradigms(only_lemmas=None):
    """Aggregate verb pairs from all sources into per-lemma paradigms."""
    print("Building Ancient Greek verb paradigms ...")
    sources = []
    for path, name in [
        (AG_PAIRS, "ag_pairs.json"),
        (GLAUX_PAIRS, "glaux_pairs.json"),
    ]:
        pairs = load_pairs(path)
        if isinstance(pairs, list):
            verb_pairs = [p for p in pairs if isinstance(p, dict)
                          and p.get("pos") == "verb"]
        else:
            verb_pairs = []
        print(f"  {name}: {len(verb_pairs)} verb pairs")
        sources.append((name, verb_pairs))
    # ag_lsj_verb_pairs.json is already merged into ag_pairs.json by
    # build_data.py. verb_extra_pairs.json is a flat form->lemma dict
    # (no tags) so it can't contribute paradigm entries.

    # Load LSJ headwords for canonical-spelling lookup. Used to
    # normalize variants where kaikki / corpus data emit a slightly
    # off form (wrong breathing, missing iota subscript). The LSJ
    # headword is treated as authoritative when both the input lemma
    # and a single accent-stripped LSJ candidate exist.
    lsj_headwords: set[str] = set()
    lsj_by_stripped: dict[str, list[str]] = defaultdict(list)
    if LSJ_HEADWORDS_PATH.exists():
        with open(LSJ_HEADWORDS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        for hw in raw:
            if not isinstance(hw, str) or not hw:
                continue
            # Strip the LSJ "Α α" header rows and multi-word phrases.
            if " " in hw:
                continue
            if not is_pure_greek(hw):
                continue
            lsj_headwords.add(hw)
            stripped = strip_accents(hw.lower())
            if stripped:
                lsj_by_stripped[stripped].append(hw)
    print(f"  LSJ headwords loaded: {len(lsj_headwords):,}")

    def canonicalize_lemma(lemma: str) -> str | None:
        """Apply lemma normalization. Returns None if the lemma should
        be dropped entirely (e.g. mojibake-like ΒΕἔστημι)."""
        if not lemma or not is_pure_greek(lemma):
            return None
        if has_internal_capital(lemma):
            return None  # mojibake / corpus glitch, drop
        # LSJ canonicalization first: pick up the LSJ-canonical spelling
        # for breathing / iota-subscript variants (ἀλίσκω -> ἁλίσκω,
        # ἀθωόω -> ἀθῳόω) when kaikki used a different convention than
        # LSJ for the same lemma.
        if lsj_headwords and lemma not in lsj_headwords:
            stripped = strip_accents(lemma.lower())
            candidates = lsj_by_stripped.get(stripped, [])
            if len(candidates) == 1:
                lemma = candidates[0]
        # Then force lowercase: verbs aren't proper nouns, and LSJ /
        # kaikki occasionally preserve a capital from a denominal
        # (Αἰγυπτιάζω, Βακχεύω, Δημοσθενίζω - "to act like X" verbs
        # derived from proper-noun X). Citation form should be
        # lowercase regardless.
        if lemma and lemma[0].isupper():
            lemma = lowercase_initial(lemma)
        return lemma

    # Group by lemma, then by dialect, then by paradigm key
    by_lemma_dialect_key = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dropped_internal_capital = 0
    canonicalized = 0
    lowercased = 0
    for src_name, verb_pairs in sources:
        for p in verb_pairs:
            lemma_raw = p.get("lemma")
            form = (p.get("form") or "").strip()
            tags = p.get("tags", [])
            if not lemma_raw or not form:
                continue
            if not is_pure_greek(lemma_raw) or not is_pure_greek(form):
                continue
            if has_internal_capital(lemma_raw):
                dropped_internal_capital += 1
                continue
            lemma = canonicalize_lemma(lemma_raw)
            if lemma is None:
                continue
            if lemma != lemma_raw:
                if lemma_raw[0].isupper() and lemma == lowercase_initial(lemma_raw):
                    lowercased += 1
                else:
                    canonicalized += 1
            if only_lemmas is not None and lemma not in only_lemmas:
                continue
            if "form-of" in tags or "alt-of" in tags or "alternative" in tags:
                continue
            key = verb_key_from_tags(tags)
            if not key:
                continue
            dialect = extract_dialect(tags)
            by_lemma_dialect_key[lemma][dialect][key].append(form)
    if dropped_internal_capital:
        print(f"  dropped (internal capitals / mojibake): "
              f"{dropped_internal_capital:,}")
    if lowercased:
        print(f"  lowercased sentence-initial verb lemmas: {lowercased:,}")
    if canonicalized:
        print(f"  LSJ-canonicalized lemmas: {canonicalized:,}")

    print(f"  candidate lemmas: {len(by_lemma_dialect_key)}")

    results = {}
    for lemma in sorted(by_lemma_dialect_key.keys()):
        by_dialect = by_lemma_dialect_key[lemma]
        attic_forms_raw = by_dialect.get("", {})
        # Pick the best form for each key in the Attic slice
        attic_forms = {}
        for key, variants in attic_forms_raw.items():
            best = pick_best_form(variants)
            if best:
                attic_forms[key] = grave_to_acute(best)

        # Default convention: the lemma is the active present indicative 1sg
        # if it ends in -ω, or middle present indicative 1sg if it ends in
        # -μαι (deponent). For -μι verbs we leave the slot blank since the
        # paradigm itself should attest the 1sg form.
        if attic_forms or any(by_dialect.values()):
            if lemma.endswith("ω"):  # ω
                attic_forms.setdefault(
                    "active_present_indicative_1sg", lemma)
            elif lemma.endswith("μαι"):  # μαι
                attic_forms.setdefault(
                    "middle_present_indicative_1sg", lemma)

        if not attic_forms:
            continue

        paradigm = {
            "forms": attic_forms,
            "form_count": len(attic_forms),
            "source": "dilemma",
        }

        # Add per-dialect paradigm slices for non-Attic dialects
        for dialect, kv in by_dialect.items():
            if not dialect:
                continue
            picked = {}
            for key, variants in kv.items():
                best = pick_best_form(variants)
                if best:
                    picked[key] = grave_to_acute(best)
            if picked:
                paradigm.setdefault("dialects", {})[dialect] = picked

        results[lemma] = paradigm

    print(f"  built paradigms for {len(results)} lemmas")
    if results:
        counts = sorted(v["form_count"] for v in results.values())
        n = len(counts)
        print(f"  forms per lemma: min={counts[0]} median={counts[n//2]} "
              f"max={counts[-1]} avg={sum(counts)/n:.1f}")

    # Procedural synthesis pass: fill missing finite-mood cells
    # (subjunctive / optative / imperative / aorist infinitive) for
    # thematic -ω verbs from LSJ-extracted principal parts. Only writes
    # into empty slots, never overwrites corpus-derived forms.
    print("  synthesising missing moods from principal parts ...")
    synthesize_missing_moods(results)
    if results:
        counts = sorted(v["form_count"] for v in results.values())
        n = len(counts)
        print(f"  forms per lemma (post-mood-synth): "
              f"min={counts[0]} median={counts[n//2]} "
              f"max={counts[-1]} avg={sum(counts)/n:.1f}")

    # Second synthesis pass: full case×gender×number participle
    # declension. Like the mood pass, only fills empty cells.
    print("  synthesising missing participles from principal parts ...")
    synthesize_missing_participles(results)
    if results:
        counts = sorted(v["form_count"] for v in results.values())
        n = len(counts)
        print(f"  forms per lemma (post-participle-synth): "
              f"min={counts[0]} median={counts[n//2]} "
              f"max={counts[-1]} avg={sum(counts)/n:.1f}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sanity", action="store_true",
                    help="6-lemma smoke test (writes to *.sanity.json)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated lemma list (debug)")
    ap.add_argument("--out", type=str, default=None,
                    help="output path override")
    args = ap.parse_args()

    only = None
    if args.sanity:
        only = ["γράφω",       # γράφω
                "τίθημι", # τίθημι
                "αἰρέω",       # αἰρέω - normal contract
                "δίδωμι", # δίδωμι
                "λύω",                   # λύω
                "εἰμί",             # εἰμί
                "ἀκούω",       # ἀκούω
                "φιλέω",       # φιλέω - epsilon contract
                "τιμάω",       # τιμάω - alpha contract
                "δηλόω",       # δηλόω - omicron contract
                ]
    elif args.only:
        only = [s.strip() for s in args.only.split(",") if s.strip()]
    only_set = set(only) if only else None

    paradigms = build_paradigms(only_lemmas=only_set)

    if args.sanity:
        out = OUT_PATH.with_suffix(".sanity.json")
    elif args.out:
        out = Path(args.out)
    else:
        out = OUT_PATH

    out.write_text(json.dumps(paradigms, ensure_ascii=False))
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  -> {out} ({size_mb:.1f} MB)")

    if only:
        # Print sanity output for inspection
        for lemma in only:
            if lemma in paradigms:
                p = paradigms[lemma]
                print(f"\n{lemma}: {p['form_count']} forms")
                for k in sorted(p["forms"].keys())[:8]:
                    print(f"  {k} = {p['forms'][k]}")
                if len(p["forms"]) > 8:
                    print(f"  ... ({len(p['forms']) - 8} more)")
            else:
                print(f"\n{lemma}: NOT FOUND")


if __name__ == "__main__":
    main()
