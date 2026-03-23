#!/usr/bin/env python3
"""Expand LSJ headwords into inflected forms using Wiktionary Lua modules.

Uses wikitextprocessor to run Wiktionary's grc-decl and grc-conj templates
on LSJ headwords that don't have Wiktionary articles. The 14K+ overlap
between LSJ and Wiktionary serves as validation.

Phase 1: nouns (40K+ LSJ entries have gender from article in entry text)
Phase 2: adjectives
Phase 3: verbs

Usage:
    python expand_lsj.py --setup          # build Wiktionary module database (first run)
    python expand_lsj.py --test           # test on overlap entries
    python expand_lsj.py --expand         # expand LSJ-only entries
    python expand_lsj.py --test-one λύπη  # test a single word
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "data"
LSJ9_DIR = Path.home() / "Documents" / "lsjpre" / "output" / "lsj9"
LSJ9_FORMS = LSJ9_DIR / "lsj9_forms.tsv"
LSJ9_HEADWORDS = LSJ9_DIR / "lsj9_headwords.json"
KAIKKI_DIR = Path(os.environ.get(
    "KAIKKI_DIR", Path.home() / "Documents" / "Klisy" / "word_collector"))
# Try nested layout first (en-el/), then flat layout
_KAIKKI_AG_NESTED = KAIKKI_DIR / "en-el" / "kaikki.org-dictionary-AncientGreek.jsonl"
_KAIKKI_AG_FLAT = KAIKKI_DIR / "kaikki.org-en-dictionary-AncientGreek.jsonl"
KAIKKI_AG = _KAIKKI_AG_NESTED if _KAIKKI_AG_NESTED.exists() else _KAIKKI_AG_FLAT
WTP_DB = DATA_DIR / "wtp.db"

# Article -> gender mapping
GENDER_MAP = {
    "ὁ": "m", "ἡ": "f", "τό": "n",
    "τά": "n", "οἱ": "m", "αἱ": "f",
    "ἁ": "f",  # Doric feminine article
}

# Nominative ending -> likely genitive ending (for regular nouns)
# Used when LSJ doesn't provide an explicit genitive
REGULAR_GENITIVE = {
    # 2nd declension
    ("ος", "m"): "ου",
    ("ος", "f"): "ου",
    ("ον", "n"): "ου",
    # 1st declension
    ("η", "f"): "ης",
    ("ᾱ", "f"): "ᾱς",
    ("α", "f"): "ας",  # short alpha (ambiguous - could be -ης)
    ("ά", "f"): "άς",
    ("ή", "f"): "ῆς",
    # 1st declension masculine
    ("ης", "m"): "ου",
    ("ᾱς", "m"): "ου",
    ("ας", "m"): "ου",
    # 3rd declension regular patterns
    ("μα", "n"): "ματος",
    ("ξ", "m"): "κος",
    ("ξ", "f"): "κος",
    ("ψ", "m"): "πος",
    ("ψ", "f"): "πος",
}

# 3rd declension patterns that need stem analysis (ending, gender) -> genitive suffix
# These replace the entire ending, not just append
THIRD_DECL_GENITIVE = {
    # -ις / -εως (πόλις type) vs -ις / -ιδος (ἐλπίς type)
    # Ambiguous: needs itype or Wiktionary cross-ref to disambiguate
    # -υς patterns
    ("ευς", "m"): "εως",       # βασιλεύς -> βασιλέως
    # -ων patterns
    ("ων", "m"): "ονος",       # λέων -> λέοντος is irregular, but δαίμων -> δαίμονος
    ("ων", "f"): "ονος",
    # -ηρ patterns
    ("ηρ", "m"): "ηρος",       # πατήρ -> πατρός is irregular, σωτήρ -> σωτῆρος
    ("ωρ", "m"): "ορος",       # ῥήτωρ -> ῥήτορος
    ("ωρ", "f"): "ορος",
    # -ης 3rd decl (proper nouns, Σωκράτης -> Σωκράτους)
    # Covered by itype usually
}


def strip_length_marks(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        ''.join(c for c in nfd if ord(c) not in (0x0306, 0x0304)))


def strip_diacritics(s: str) -> str:
    """Strip all combining diacritics for accent-free comparison."""
    nfd = unicodedata.normalize("NFD", s)
    return ''.join(c for c in nfd if not unicodedata.combining(c))


def build_genitive_from_itype(headword, itype):
    """Build genitive form from headword + LSJ itype (genitive suffix).

    The itype replaces a portion of the headword's ending. The number of
    characters to strip depends on the nominative ending pattern.
    """
    if not itype:
        return ""

    hw_plain = strip_diacritics(headword)
    it_plain = strip_diacritics(itype)

    # (nominative ending, itype ending) → chars to strip from headword
    # Ordered so more specific patterns match first
    STRIP_RULES = [
        # -ευς / -εως: strip 3 (the ε is shared between stem and itype)
        ("ευς", "εως", 3),
        # 1st/2nd declension
        ("ος", "ου", 2),
        ("ον", "ου", 2),
        ("ης", "ου", 2),
        ("ας", "ου", 2),
        ("η", "ης", 1),
        # 3rd declension -ις/-εως (biggest: 4451 entries, πόλις-type)
        ("ις", "εως", 2),
        # 3rd declension -υς/-εως (πῆχυς-type, NOT -ευς which is above)
        ("υς", "εως", 2),
        # 3rd declension -ης/-ους (Attic, e.g. Σωκράτης)
        ("ης", "ους", 2),
        # 3rd declension -ης/-ητος
        ("ης", "ητος", 2),
        # 3rd declension -ως/-ω (Attic, e.g. ἥρως)
        ("ως", "ωος", 2),
    ]

    for nom_end, gen_itype, strip_n in STRIP_RULES:
        if hw_plain.endswith(nom_end) and it_plain == strip_diacritics(gen_itype):
            return headword[:-strip_n] + itype

    # Default: strip 1 char (the final case marker) and append itype.
    # Works for most 3rd declension patterns where itype starts from
    # the oblique stem consonant:
    #   -μα + ατος → -ματος (strip α, append ατος)
    #   -ίς + ίδος → -ίδος (strip ς, append... wait, ίδος)
    #   -ήρ + ῆρος → -ῆρος (strip ρ... hmm)
    #
    # Actually for consonant stems, strip 1 often leaves extra chars.
    # Try strip 2 if itype starts with a char that matches the second-to-last
    # char of the headword (accent-free).
    if len(hw_plain) >= 2 and len(it_plain) >= 1:
        if hw_plain[-2] == it_plain[0]:
            # The itype "restarts" from a character that's already in the headword
            # e.g. ἐλπίς + ίδος: ί matches → strip 2, append ίδος
            return headword[:-2] + itype
    return headword[:-1] + itype


def parse_lsj_entries():
    """Parse LSJ entries from lsj9 exports.

    Priority: lsj9_forms.tsv (63K entries with explicit grammar) is loaded
    first, then lsj9_headwords.json fills in remaining entries (those without
    grammar in forms.tsv but present in the full headword list).
    """
    # Start with lsj9 forms data (entries with explicit grammar)
    entries = parse_lsj9_entries()

    # Fill in from headwords.json for entries not covered by forms.tsv
    hw_entries = _parse_lsj9_headwords()
    hw_only = 0
    gen_fills = 0
    for hw, entry in hw_entries.items():
        if hw not in entries:
            entries[hw] = entry
            hw_only += 1
        elif not entries[hw]["genitive"] and entry.get("genitive"):
            # forms.tsv entry exists but lacks genitive - fill from headwords
            entries[hw]["genitive"] = entry["genitive"]
            entries[hw]["itype"] = entry.get("itype", "")
            gen_fills += 1

    if hw_only:
        print(f"  headwords-only: {hw_only:,} additional entries")
    if gen_fills:
        print(f"  genitive fills from headwords: {gen_fills:,}")
    print(f"  Total: {len(entries):,} entries")

    return entries


def _parse_lsj9_headwords():
    """Load all LSJ entries from lsj9_headwords.json.

    Returns entries keyed by length-mark-stripped headword, with gender and
    genitive extracted from the structured JSON.
    """
    entries = {}
    if not LSJ9_HEADWORDS.exists():
        print(f"  lsj9_headwords.json not found at {LSJ9_HEADWORDS}")
        return entries

    with open(LSJ9_HEADWORDS, encoding="utf-8") as f:
        headwords_list = json.load(f)

    _grammar_to_info = {
        "ὁ": ("ὁ", "m"),
        "ἡ": ("ἡ", "f"),
        "τό": ("τό", "n"),
        "ον": ("ον", ""),
        "ές": ("ές", ""),
    }

    for e in headwords_list:
        hw_orig = e["headword"]
        hw = strip_length_marks(hw_orig)
        grammar = e.get("grammar", "")
        genitive = e.get("genitive", "")

        article, gender = _grammar_to_info.get(grammar, ("", ""))
        itype = genitive if grammar in ("ὁ", "ἡ", "τό") and genitive else ""

        if hw not in entries:
            entries[hw] = {
                "headword": hw,
                "orth_orig": hw_orig,
                "article": article,
                "gender": gender,
                "itype": itype,
                "genitive": genitive if gender else "",
            }

    print(f"  lsj9_headwords: {len(entries):,} entries")
    return entries


def parse_lsj9_entries(forms_path: Path = LSJ9_FORMS) -> dict:
    """Load entries from lsj9_forms.tsv (output of lsjpre export_lsj9.py).

    This provides explicit grammar (ὁ/ἡ/τό/ον/ές) and pre-extracted
    genitive endings for 63K entries.

    Returns same format as parse_lsj_entries(): {headword: {headword,
    orth_orig, article, gender, itype, genitive}}.
    """
    if not forms_path.exists():
        print(f"  lsj9_forms.tsv not found at {forms_path}")
        return {}

    # Grammar -> (article, gender) mapping
    _grammar_to_article = {
        "ὁ": ("ὁ", "m"),
        "ἡ": ("ἡ", "f"),
        "τό": ("τό", "n"),
        "ον": ("ον", ""),    # adjective, no article/gender
        "ές": ("ές", ""),    # adjective, no article/gender
    }

    entries = {}
    with open(forms_path, encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            hw_raw, grammar, genitive, etymology = parts
            hw = strip_length_marks(hw_raw)

            article_info = _grammar_to_article.get(grammar, ("", ""))
            article, gender = article_info

            # For adjectives (ον/ές), try to infer genitive from pattern
            itype = ""
            if grammar == "ον":
                # 2-termination adjective: -ος/-ον type
                # genitive would be -ου (same as masculine)
                itype = "ον"
            elif grammar == "ές":
                # -ής/-ές type adjective
                itype = "ές"
            elif genitive:
                # Use the extracted genitive as itype
                itype = genitive

            if hw not in entries:
                entries[hw] = {
                    "headword": hw,
                    "orth_orig": hw_raw,
                    "article": article,
                    "gender": gender,
                    "itype": itype,
                    "genitive": genitive if grammar in ("ὁ", "ἡ", "τό") else "",
                }

    print(f"  lsj9: {len(entries):,} entries from {forms_path.name}")
    return entries


def load_wiktionary_forms():
    """Load form sets from Wiktionary kaikki for overlap validation."""
    wikt = {}
    if not KAIKKI_AG.exists():
        return wikt
    with open(KAIKKI_AG, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            word = e.get("word", "")
            if not word:
                continue
            forms = e.get("forms", [])
            real_forms = set()
            for fe in forms:
                tags = fe.get("tags", [])
                if "table-tags" in tags or "inflection-template" in tags:
                    continue
                form = strip_length_marks(fe.get("form", ""))
                if form and any('\u0370' <= c <= '\u03FF' or
                                '\u1F00' <= c <= '\u1FFF' for c in form):
                    real_forms.add(form)
            if word not in wikt or len(real_forms) > len(wikt[word]["forms"]):
                wikt[word] = {
                    "pos": e.get("pos", ""),
                    "forms": real_forms,
                }
    return wikt


def load_wiktionary_genitives():
    """Load genitive forms from Wiktionary for cross-referencing with LSJ.

    Returns {headword: genitive_form} for nouns/adjectives that have
    genitive singular forms tagged in Wiktionary.
    """
    genitives = {}
    if not KAIKKI_AG.exists():
        return genitives
    with open(KAIKKI_AG, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            word = e.get("word", "")
            pos = e.get("pos", "")
            if pos not in ("noun", "adj"):
                continue
            forms = e.get("forms", [])
            for fe in forms:
                tags = fe.get("tags", [])
                form = strip_length_marks(fe.get("form", ""))
                if "genitive" in tags and "singular" in tags and form:
                    # Strip article prefix if present (e.g. "τῆς κυνός" -> "κυνός")
                    if " " in form:
                        form = form.split()[-1]
                    genitives[strip_length_marks(word)] = form
                    break
    return genitives


def infer_genitive(headword, gender, wikt_genitives=None):
    """Infer genitive singular from nominative ending + gender.

    Priority:
    1. Wiktionary cross-reference (gold standard)
    2. Regular 1st/2nd declension patterns
    3. 3rd declension heuristics (lower confidence)
    """
    # Check Wiktionary cross-reference first
    if wikt_genitives:
        hw_clean = strip_length_marks(headword)
        if hw_clean in wikt_genitives:
            return wikt_genitives[hw_clean]

    hw_plain = strip_diacritics(headword)

    # Regular patterns (1st/2nd declension - high confidence)
    for (ending, g), gen_ending in REGULAR_GENITIVE.items():
        if gender == g and hw_plain.endswith(ending):
            stem = headword[:-len(ending)]
            return stem + gen_ending

    # 3rd declension patterns (moderate confidence)
    for (ending, g), gen_ending in THIRD_DECL_GENITIVE.items():
        if gender == g and hw_plain.endswith(ending):
            stem = headword[:-len(ending)]
            return stem + gen_ending

    return ""


def setup_wtp():
    """Set up wikitextprocessor database from kaikki.org module/template tarballs.

    Downloads tarballs if needed, loads modules and templates into a SQLite db.
    Only needs to run once.
    """
    import tarfile
    import urllib.request
    from wikitextprocessor import Wtp

    print("Setting up wikitextprocessor database...")

    modules_url = "https://kaikki.org/dictionary/wiktionary-modules.tar.gz"
    templates_url = "https://kaikki.org/dictionary/wiktionary-templates.tar.gz"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for url, name in [(modules_url, "wiktionary-modules"),
                      (templates_url, "wiktionary-templates")]:
        dest = DATA_DIR / f"{name}.tar.gz"
        if not dest.exists():
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  {size_mb:.1f} MB")

    if WTP_DB.exists():
        WTP_DB.unlink()

    wtp = Wtp(db_path=str(WTP_DB))

    NS_MODULE = 828
    NS_TEMPLATE = 10

    def tar_to_title(member_name, namespace):
        """Convert tar path like 'Module/grc-decl.txt' to 'Module:grc-decl'."""
        name = member_name
        if name.startswith(f"{namespace}/"):
            name = name[len(f"{namespace}/"):]
        if name.endswith(".txt"):
            name = name[:-4]
        return f"{namespace}:{name}"

    for ns_name, ns_id, model in [("Module", NS_MODULE, "Scribunto"),
                                   ("Template", NS_TEMPLATE, "wikitext")]:
        tarball = DATA_DIR / f"wiktionary-{ns_name.lower()}s.tar.gz"
        print(f"  Loading {ns_name.lower()}s...")
        count = 0
        with tarfile.open(tarball, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                body = f.read().decode("utf-8", errors="replace")
                title = tar_to_title(member.name, ns_name)
                wtp.add_page(title, ns_id, body, model=model)
                count += 1
        print(f"  Loaded {count:,} {ns_name.lower()}s")

    wtp.db_conn.commit()
    print("  Database ready.")
    return wtp


def get_wtp():
    """Get wikitextprocessor instance, setting up if needed."""
    from wikitextprocessor import Wtp

    if WTP_DB.exists():
        wtp = Wtp(db_path=str(WTP_DB))
        return wtp
    return setup_wtp()


MACRON = "\u0304"  # combining macron
BREVE = "\u0306"   # combining breve

def mark_alpha_length(word):
    """Add macron to final alpha if it follows ι, ρ, or ε (long alpha rule).
    Add breve for short alpha patterns."""
    base = strip_diacritics(word)
    if not base.endswith("α") and not base.endswith("ας"):
        return word
    # Find the character before the alpha
    alpha_pos = len(base) - 1 if base.endswith("α") else len(base) - 2
    if alpha_pos < 1:
        return word
    preceding = base[alpha_pos - 1]
    if preceding in ("ι", "ρ", "ε"):
        # Long alpha after ι, ρ, ε - insert macron after the α
        # Find the actual α in the original word (may have accents)
        nfd = unicodedata.normalize("NFD", word)
        # Find the alpha at the right position and add macron after it
        result = []
        base_idx = 0
        for ch in nfd:
            if not unicodedata.combining(ch):
                base_idx += 1
            if base_idx == alpha_pos + 1 and ch == "α":
                result.append(ch)
                result.append(MACRON)
                base_idx_done = True
            else:
                result.append(ch)
        return unicodedata.normalize("NFC", "".join(result))
    return word


# Cache: (gender, nom_ending, gen_ending) -> list of full forms from reference word
# Nouns with the same declension pattern produce forms that differ only in the stem.
# We store (ref_stem, ref_forms) and apply by swapping stems.
_NOUN_CACHE = {}


def _classify_noun(headword, gender, genitive):
    """Classify a noun into declension type for caching.

    Returns (cache_key, stem_len) where cache_key = (gender, nom_ending, gen_ending).
    stem_len is how many chars of the headword form the invariant stem.
    """
    hw_plain = strip_diacritics(headword.lower())
    gen_plain = strip_diacritics(genitive.lower()) if genitive else ""

    # Find the longest common prefix = invariant stem
    # Then trim back to exclude the thematic/connecting vowel, which
    # is shared between nom and gen but changes in other cases
    # (e.g. ανθρωπ-ος/ου share 'ο' but vocative is ανθρωπ-ε)
    stem_len = 0
    for i in range(min(len(hw_plain), len(gen_plain)) if gen_plain else 0):
        if hw_plain[i] == gen_plain[i]:
            stem_len = i + 1
        else:
            break

    # Trim back thematic vowel: if the last shared char is a vowel
    # and it's the first char of both endings, exclude it from stem
    vowels = set("αεηιουω")
    if stem_len > 1 and hw_plain[stem_len - 1] in vowels:
        stem_len -= 1

    # Ensure at least 1 char as ending
    stem_len = min(stem_len, len(hw_plain) - 1)

    nom_ending = hw_plain[stem_len:]
    gen_ending = gen_plain[stem_len:] if gen_plain else ""

    return (gender, nom_ending, gen_ending), stem_len


def _apply_noun_cache(headword, stem_len, ref_stem_plain, ref_forms):
    """Apply cached declension forms to a new headword by swapping the stem.

    ref_forms are the full accented forms from the reference word.
    We strip the reference stem prefix and replace it with the new stem.
    """
    new_stem = strip_diacritics(headword[:stem_len].lower())
    forms = set()
    for form in ref_forms:
        form_plain = strip_diacritics(form.lower())
        if form_plain.startswith(ref_stem_plain):
            ending = form_plain[len(ref_stem_plain):]
            new_form = new_stem + ending
            if len(new_form) > 1:
                forms.add(new_form)
    return forms


def expand_noun(wtp, headword, gender, genitive="", wikt_genitives=None):
    """Expand a noun using grc-decl template. Returns set of forms.

    Uses a cache keyed by declension pattern: nouns with the same
    (gender, nom_ending, gen_ending) produce the same inflection endings.
    """
    if not genitive:
        genitive = infer_genitive(headword, gender, wikt_genitives)

    # Check cache
    if genitive:
        cache_key, stem_len = _classify_noun(headword, gender, genitive)
        if cache_key in _NOUN_CACHE:
            ref_stem, ref_forms = _NOUN_CACHE[cache_key]
            forms = _apply_noun_cache(headword, stem_len, ref_stem, ref_forms)
            if forms:
                return forms, ""

    # grc-decl gender codes
    gender_code = {"m": "M", "f": "F", "n": "N"}.get(gender, "")

    # Mark alpha length for disambiguation
    hw_marked = mark_alpha_length(headword)
    gen_marked = mark_alpha_length(genitive) if genitive else ""

    parts = [hw_marked]
    if gen_marked:
        parts.append(gen_marked)
    else:
        parts.append("")

    form_param = f"|form={gender_code}" if gender_code else ""
    template = "{{grc-decl|" + "|".join(parts) + form_param + "}}"

    try:
        wtp.start_page(headword)
        html = wtp.expand(template)
    except Exception as e:
        return set(), str(e)

    forms = parse_html_forms(html, headword)

    # Cache: store (ref_stem_plain, ref_forms) for this declension pattern
    if forms and genitive:
        cache_key, stem_len = _classify_noun(headword, gender, genitive)
        if cache_key not in _NOUN_CACHE:
            ref_stem = strip_diacritics(headword[:stem_len].lower())
            _NOUN_CACHE[cache_key] = (ref_stem, forms)

    return forms, ""


# Cache: (conj_type, stem_len) -> list of suffix offsets extracted from a reference expansion
# Each cached entry is [(suffix, strip_n), ...] where form = stem[:-strip_n] + suffix
_VERB_CACHE = {}


def _classify_verb(headword):
    """Classify a verb into conjugation type and stem. Returns (conj_type, stem) or (None, None)."""
    hw_plain = strip_diacritics(headword)

    if hw_plain.endswith("εω"):
        return "pres-con-e", strip_diacritics(headword[:-2])
    elif hw_plain.endswith("αω"):
        return "pres-con-a", strip_diacritics(headword[:-2])
    elif hw_plain.endswith("οω"):
        return "pres-con-o", strip_diacritics(headword[:-2])
    elif hw_plain.endswith("ννυμι"):
        return "pres-numi", strip_diacritics(headword[:-5])
    elif hw_plain.endswith("νυμι"):
        return "pres-numi", strip_diacritics(headword[:-4])
    elif hw_plain.endswith("ημι"):
        return "pres-emi", strip_diacritics(headword[:-3])
    elif hw_plain.endswith("ωμι"):
        return "pres-omi", strip_diacritics(headword[:-3])
    elif hw_plain.endswith("αμι"):
        return "pres-ami", strip_diacritics(headword[:-3])
    elif hw_plain.endswith("μι"):
        return "pres-mi", strip_diacritics(headword[:-2])
    elif hw_plain.endswith("ω"):
        return "pres", strip_diacritics(headword[:-1])
    elif hw_plain.endswith("μαι"):
        return "pres", strip_diacritics(headword[:-3])
    return None, None


def _build_suffix_cache(forms, stem):
    """Extract suffix patterns from expanded forms relative to the stem.

    Returns list of suffixes where each form = stem_prefix + suffix.
    The stem_prefix is the longest common prefix between stem and form.
    """
    stem_plain = strip_diacritics(stem.lower())
    suffixes = []
    for form in forms:
        form_plain = strip_diacritics(form.lower())
        # Find how much of the stem matches the beginning of the form
        match_len = 0
        for i in range(min(len(stem_plain), len(form_plain))):
            if stem_plain[i] == form_plain[i]:
                match_len = i + 1
            else:
                break
        if match_len > 0:
            suffixes.append(form_plain[match_len:])
    return suffixes


def _apply_suffix_cache(stem, suffixes, headword):
    """Apply cached suffix patterns to a new stem. Returns set of forms."""
    stem_plain = strip_diacritics(stem.lower())
    forms = set()
    for suffix in suffixes:
        form = stem_plain + suffix
        if len(form) > 1:
            forms.add(form)
    return forms


def expand_verb(wtp, headword):
    """Expand a verb using grc-conj template. Returns set of forms.

    Uses a cache keyed by conjugation type: if we've already expanded a verb
    with the same conj_type, apply the suffix pattern instead of calling Lua.
    """
    conj_type, stem = _classify_verb(headword)
    if conj_type is None:
        return set(), f"unknown-ending:{headword[-3:]}"

    # Check cache
    if conj_type in _VERB_CACHE:
        forms = _apply_suffix_cache(stem, _VERB_CACHE[conj_type], headword)
        if forms:
            return forms, ""

    # Cache miss - call Lua
    template = "{{grc-conj|" + conj_type + "|" + stem + "}}"

    try:
        wtp.start_page(headword)
        html = wtp.expand(template)
    except Exception as e:
        # For -μι verbs, try falling back to simpler conjugation types
        if conj_type.startswith("pres-") and conj_type != "pres":
            for fallback in ["pres-mi", "pres"]:
                if fallback == conj_type:
                    continue
                try:
                    template = "{{grc-conj|" + fallback + "|" + stem + "}}"
                    wtp.start_page(headword)
                    html = wtp.expand(template)
                    forms = parse_html_forms(html, headword)
                    if forms:
                        # Cache the working fallback under the original conj_type
                        _VERB_CACHE[conj_type] = _build_suffix_cache(forms, stem)
                        return forms, ""
                except Exception:
                    continue
        return set(), str(e)

    forms = parse_html_forms(html, headword)
    # Cache the suffix pattern for this conjugation type
    if forms and conj_type not in _VERB_CACHE:
        _VERB_CACHE[conj_type] = _build_suffix_cache(forms, stem)

    return forms, ""


ARTICLES = {"ὁ", "ἡ", "τό", "τοῦ", "τῆς", "τῷ", "τῇ", "τόν", "τήν",
            "τών", "τῶν", "τοῖς", "ταῖς", "τούς", "τάς", "τά",
            "τοῖν", "ταῖν", "τώ", "τὼ", "αἱ", "οἱ",
            "τὰς", "τὴν", "τὸ", "τὸν", "τοὺς", "τὰ"}

def parse_html_forms(html, headword):
    """Extract Greek word forms from expanded HTML table."""
    forms = set()
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    for token in re.split(r'[\s,/;]+', text):
        token = token.strip('.,;:()[]—– ')
        # Handle wikilink pipe artifacts like "Greek|λύπη"
        if '|' in token:
            token = token.split('|')[-1]
        # Strip anchor fragments like "λύπη#Ancient&#95"
        if '#' in token:
            token = token.split('#')[0]
        token = strip_length_marks(token)
        if (token and len(token) > 1 and
            any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF'
                for c in token)
            and token not in ARTICLES
            and not token.isupper()
            and token[0] != '-'):
            forms.add(token)
    return forms


def test_one(word):
    """Test expansion on a single word."""
    lsj_entries = parse_lsj_entries()
    wikt = load_wiktionary_forms()

    entry = lsj_entries.get(word)
    if not entry:
        print(f"{word} not in LSJ")
        return

    print(f"LSJ: {entry}")
    if word in wikt:
        print(f"Wiktionary: pos={wikt[word]['pos']}, {len(wikt[word]['forms'])} forms")
    else:
        print("Not in Wiktionary")

    wtp = get_wtp()

    if entry["gender"]:
        forms, err = expand_noun(wtp, word, entry["gender"], entry["genitive"])
        print(f"\ngrc-decl result: {len(forms)} forms" + (f" (error: {err})" if err else ""))
    else:
        forms, err = expand_verb(wtp, word)
        print(f"\ngrc-conj result: {len(forms)} forms" + (f" (error: {err})" if err else ""))

    if forms:
        print(f"Sample forms: {sorted(forms)[:20]}")

    if word in wikt:
        wikt_forms = wikt[word]["forms"]
        overlap = forms & wikt_forms
        lua_only = forms - wikt_forms
        wikt_only = wikt_forms - forms
        print(f"\nOverlap with Wiktionary: {len(overlap)}")
        print(f"Lua-only: {len(lua_only)}")
        print(f"Wiktionary-only: {len(wikt_only)}")
        if wikt_only:
            print(f"  Missing: {sorted(wikt_only)[:10]}")


def test_overlap():
    """Test expansion accuracy on overlap entries."""
    print("Loading data...")
    lsj_entries = parse_lsj_entries()
    wikt = load_wiktionary_forms()
    wikt_genitives = load_wiktionary_genitives()
    print(f"  {len(wikt_genitives):,} Wiktionary genitives loaded")
    overlap = {w for w in lsj_entries if w in wikt and lsj_entries[w]["gender"]}

    print(f"Overlap nouns with gender: {len(overlap)}")

    wtp = get_wtp()

    results = {"success": 0, "partial": 0, "fail": 0, "error": 0}
    total_recall = []
    total_precision = []

    sample_size = min(200, len(overlap))  # test a sample first
    import random
    random.seed(42)
    sample = random.sample(sorted(overlap), sample_size)

    for i, word in enumerate(sample):
        entry = lsj_entries[word]
        forms, err = expand_noun(wtp, word, entry["gender"], entry["genitive"],
                                 wikt_genitives=wikt_genitives)

        if err:
            results["error"] += 1
            continue

        wikt_forms = wikt[word]["forms"]
        if not wikt_forms:
            continue

        recall = len(forms & wikt_forms) / len(wikt_forms) if wikt_forms else 0
        precision = len(forms & wikt_forms) / len(forms) if forms else 0
        total_recall.append(recall)
        total_precision.append(precision)

        if recall > 0.8:
            results["success"] += 1
        elif recall > 0.3:
            results["partial"] += 1
        else:
            results["fail"] += 1

        if (i + 1) % 50 == 0:
            avg_r = sum(total_recall) / len(total_recall)
            avg_p = sum(total_precision) / len(total_precision)
            print(f"  {i+1}/{sample_size}: recall={avg_r:.2f} precision={avg_p:.2f}")

    avg_recall = sum(total_recall) / len(total_recall) if total_recall else 0
    avg_precision = sum(total_precision) / len(total_precision) if total_precision else 0

    print(f"\nResults ({sample_size} nouns):")
    print(f"  Avg recall: {avg_recall:.1%}")
    print(f"  Avg precision: {avg_precision:.1%}")
    print(f"  Success (>80% recall): {results['success']}")
    print(f"  Partial (30-80%): {results['partial']}")
    print(f"  Fail (<30%): {results['fail']}")
    print(f"  Error: {results['error']}")


AG_LOOKUP = DATA_DIR / "ag_lookup.json"


def expand_all():
    """Expand LSJ-only nouns and merge into ag_lookup.json."""
    import time

    print("Loading data...")
    lsj_entries = parse_lsj_entries()
    wikt = load_wiktionary_forms()
    print("Loading Wiktionary genitives for cross-reference...")
    wikt_genitives = load_wiktionary_genitives()
    print(f"  {len(wikt_genitives):,} Wiktionary genitives loaded")

    # Load existing lookup
    print(f"Loading {AG_LOOKUP}...")
    with open(AG_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)
    original_size = len(lookup)
    print(f"  {original_size:,} existing entries")

    # Find LSJ-only nouns with gender
    candidates = []
    for hw, entry in lsj_entries.items():
        if hw in wikt:
            continue  # already covered by Wiktionary
        if not entry["gender"]:
            continue  # no gender = can't decline
        if not entry["genitive"] and not infer_genitive(hw, entry["gender"], wikt_genitives):
            continue  # no genitive info at all
        candidates.append(hw)

    print(f"LSJ-only nouns to expand: {len(candidates):,}")

    wtp = get_wtp()

    stats = {"expanded": 0, "failed": 0, "new_forms": 0, "collisions": 0}
    t0 = time.time()

    for i, hw in enumerate(candidates):
        entry = lsj_entries[hw]
        forms, err = expand_noun(wtp, hw, entry["gender"], entry["genitive"],
                                 wikt_genitives=wikt_genitives)

        if err or not forms:
            stats["failed"] += 1
            continue

        stats["expanded"] += 1

        for form in forms:
            # Accented version
            if form not in lookup:
                lookup[form] = hw
                stats["new_forms"] += 1
            elif lookup[form] != hw:
                stats["collisions"] += 1

            # Accent-stripped version
            plain = strip_diacritics(form)
            if plain != form:
                if plain not in lookup:
                    lookup[plain] = hw
                    stats["new_forms"] += 1
                elif lookup[plain] != hw:
                    stats["collisions"] += 1

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(candidates) - i - 1) / rate
            print(f"  {i+1:,}/{len(candidates):,} "
                  f"({stats['expanded']:,} ok, {stats['failed']:,} fail, "
                  f"{stats['new_forms']:,} new forms) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Expanded: {stats['expanded']:,} / {len(candidates):,} nouns")
    print(f"  Failed: {stats['failed']:,}")
    print(f"  New forms added: {stats['new_forms']:,}")
    print(f"  Collisions (kept existing): {stats['collisions']:,}")
    print(f"  Lookup size: {original_size:,} -> {len(lookup):,}")

    # Save
    out_path = AG_LOOKUP
    print(f"\nSaving to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")


def expand_verbs():
    """Expand LSJ verbs and merge into ag_lookup.json."""
    import time

    print("Loading data...")
    lsj_entries = parse_lsj_entries()
    wikt = load_wiktionary_forms()

    print(f"Loading {AG_LOOKUP}...")
    with open(AG_LOOKUP, encoding="utf-8") as f:
        lookup = json.load(f)
    original_size = len(lookup)
    print(f"  {original_size:,} existing entries")

    # Find LSJ-only verbs (entries without gender = likely verbs)
    candidates = []
    for hw, entry in lsj_entries.items():
        if hw in wikt:
            continue
        if entry["gender"]:
            continue  # has gender = noun/adj, not verb
        dp = strip_diacritics(hw)
        if (dp.endswith("ω") or dp.endswith("μι")
                or dp.endswith("μαι")):
            candidates.append(hw)

    print(f"LSJ-only verbs to expand: {len(candidates):,}")

    wtp = get_wtp()

    stats = {"expanded": 0, "failed": 0, "new_forms": 0, "collisions": 0}
    t0 = time.time()

    for i, hw in enumerate(candidates):
        forms, err = expand_verb(wtp, hw)

        if err or not forms:
            stats["failed"] += 1
            continue

        stats["expanded"] += 1

        for form in forms:
            if form not in lookup:
                lookup[form] = hw
                stats["new_forms"] += 1
            elif lookup[form] != hw:
                stats["collisions"] += 1

            plain = strip_diacritics(form)
            if plain != form:
                if plain not in lookup:
                    lookup[plain] = hw
                    stats["new_forms"] += 1
                elif lookup[plain] != hw:
                    stats["collisions"] += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(candidates) - i - 1) / rate
            print(f"  {i+1:,}/{len(candidates):,} "
                  f"({stats['expanded']:,} ok, {stats['failed']:,} fail, "
                  f"{stats['new_forms']:,} new forms) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Expanded: {stats['expanded']:,} / {len(candidates):,} verbs")
    print(f"  Failed: {stats['failed']:,}")
    print(f"  New forms added: {stats['new_forms']:,}")
    print(f"  Collisions (kept existing): {stats['collisions']:,}")
    print(f"  Lookup size: {original_size:,} -> {len(lookup):,}")

    print(f"\nSaving to {AG_LOOKUP}...")
    with open(AG_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)
    size_mb = AG_LOOKUP.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB written")


def main():
    parser = argparse.ArgumentParser(description="Expand LSJ headwords via Wiktionary Lua")
    parser.add_argument("--setup", action="store_true",
                        help="Set up wikitextprocessor database (first run)")
    parser.add_argument("--test", action="store_true",
                        help="Test on overlap entries")
    parser.add_argument("--test-one", type=str, default=None,
                        help="Test a single word")
    parser.add_argument("--expand", action="store_true",
                        help="Expand LSJ-only noun entries and add to lookup")
    parser.add_argument("--expand-verbs", action="store_true",
                        help="Expand LSJ-only verb entries and add to lookup")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --expand, show stats but don't save")
    args = parser.parse_args()

    if args.setup:
        setup_wtp()
    elif args.test_one:
        test_one(args.test_one)
    elif args.test:
        test_overlap()
    elif args.expand:
        expand_all()
    elif args.expand_verbs:
        expand_verbs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
