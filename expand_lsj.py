#!/usr/bin/env python3
"""Expand LSJ headwords into inflected forms using Wiktionary Lua modules.

Uses wikitextprocessor to run Wiktionary's grc-decl and grc-conj templates
on LSJ headwords that don't have Wiktionary articles. The 14K+ overlap
between LSJ and Wiktionary serves as validation.

Phase 1: nouns (43K LSJ entries have gender from <gen> element)
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
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
LSJ_DIR = Path.home() / "Documents" / "LSJLogeion"
KAIKKI_AG = SCRIPT_DIR / "kaikki" / "kaikki.org-en-dictionary-AncientGreek.jsonl"
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
}


def strip_length_marks(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        ''.join(c for c in nfd if ord(c) not in (0x0306, 0x0304)))


def parse_lsj_entries():
    """Parse LSJ XML files, extract headword + gender + genitive."""
    entries = {}
    for xml_file in sorted(LSJ_DIR.glob("greatscott*.xml")):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue
        for entry in tree.findall('.//div2'):
            head = entry.find('.//head')
            if head is None:
                continue
            headword = ''.join(head.itertext()).strip()
            hw = headword.split(',')[0].strip().rstrip('.,;:()[]- ')
            hw = strip_length_marks(hw)
            if not hw or not any('\u0370' <= c <= '\u03FF' or
                                  '\u1F00' <= c <= '\u1FFF' for c in hw):
                continue

            # Gender from <gen> element
            gen_elem = entry.find('.//gen')
            article = ''.join(gen_elem.itertext()).strip() if gen_elem is not None else ""
            article = article.rstrip('.,; ')
            gender = GENDER_MAP.get(article, "")

            # Try to extract explicit genitive
            xml_str = ET.tostring(entry, encoding='unicode')
            genitive = ""
            m = re.search(r'gen\.\s*<foreign[^>]*>([α-ωά-ώἀ-ᾧΑ-ΩῬ]+)', xml_str)
            if not m:
                m = re.search(r'gen\.\s*([α-ωά-ώἀ-ᾧ]+)', xml_str)
            if m:
                genitive = strip_length_marks(m.group(1).strip().rstrip(','))

            entries[hw] = {
                "headword": hw,
                "article": article,
                "gender": gender,
                "genitive": genitive,
            }
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


def infer_genitive(headword, gender):
    """Infer genitive singular from nominative ending + gender."""
    for (ending, g), gen_ending in REGULAR_GENITIVE.items():
        if gender == g and headword.endswith(ending):
            stem = headword[:-len(ending)]
            return stem + gen_ending
    return ""


def setup_wtp():
    """Set up wikitextprocessor database from Wiktionary dump.

    This processes the dump to extract all Lua modules and templates.
    Only needs to run once.
    """
    from wikitextprocessor import Wtp

    print("Setting up wikitextprocessor database...")
    print("This processes the Wiktionary dump to extract Lua modules.")
    print("It takes a while on first run but only needs to happen once.")

    # We need an enwiktionary dump. Check if we have one.
    dump_dir = Path.home() / "Documents" / "Klisy" / "word_collector"
    dump_candidates = list(dump_dir.glob("enwiktionary-*-pages-articles.xml*"))
    if not dump_candidates:
        # Try downloading just the modules
        print("\nNo Wiktionary dump found. Downloading module data from kaikki.org...")
        print("(This is much faster than processing a full dump)")

        import urllib.request
        import tempfile

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

        # Load into wtp
        import tarfile

        wtp = Wtp(cache_file=str(WTP_DB))

        def tar_to_title(member_name, namespace):
            """Convert tar path like 'Module/grc-decl.txt' to 'Module:grc-decl'."""
            # Strip namespace prefix directory (Module/ or Template/)
            name = member_name
            for prefix in (f"{namespace}/", ):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            # Strip .txt extension
            if name.endswith(".txt"):
                name = name[:-4]
            # Convert / to : for submodules (e.g. grc-decl/decl -> grc-decl/decl)
            # MediaWiki uses / for subpages, which is already correct
            return f"{namespace}:{name}"

        # Modules (Scribunto model)
        modules_path = DATA_DIR / "wiktionary-modules.tar.gz"
        print(f"  Loading modules...")
        count = 0
        with tarfile.open(modules_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                body = f.read().decode("utf-8", errors="replace")
                title = tar_to_title(member.name, "Module")
                wtp.add_page("Scribunto", title, body)
                count += 1
        print(f"  Loaded {count:,} modules")

        # Templates (wikitext model)
        templates_path = DATA_DIR / "wiktionary-templates.tar.gz"
        print(f"  Loading templates...")
        count = 0
        with tarfile.open(templates_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                body = f.read().decode("utf-8", errors="replace")
                title = tar_to_title(member.name, "Template")
                wtp.add_page("wikitext", title, body)
                count += 1
        print(f"  Loaded {count:,} templates")

        wtp.analyze_templates()
        print("  Database ready.")
        return wtp

    # Process dump
    dump = dump_candidates[0]
    print(f"  Processing dump: {dump}")
    wtp = Wtp(cache_file=str(WTP_DB))
    wtp.process_dump(str(dump), phase1_only=True)
    print("  Database ready.")
    return wtp


def get_wtp():
    """Get wikitextprocessor instance, setting up if needed."""
    from wikitextprocessor import Wtp

    if WTP_DB.exists():
        wtp = Wtp(cache_file=str(WTP_DB))
        return wtp
    return setup_wtp()


def expand_noun(wtp, headword, gender, genitive=""):
    """Expand a noun using grc-decl template. Returns set of forms."""
    if not genitive:
        genitive = infer_genitive(headword, gender)

    article = {"m": "ὁ", "f": "ἡ", "n": "τό"}.get(gender, "")

    # Try with genitive if we have it
    if genitive:
        template = f"{{{{grc-decl|{headword}|{genitive}|{article}}}}}"
    else:
        template = f"{{{{grc-decl|{headword}||{article}}}}}"

    try:
        wtp.start_page(headword)
        html = wtp.expand(template)
    except Exception as e:
        return set(), str(e)

    return parse_html_forms(html, headword), ""


def expand_verb(wtp, headword):
    """Expand a verb using grc-conj template. Returns set of forms."""
    # Determine conjugation type from ending
    if headword.endswith("ω"):
        # Regular thematic
        stem = headword[:-1]
        template = f"{{{{grc-conj|pres|act|{stem}}}}}"
    elif headword.endswith("έω") or headword.endswith("εω"):
        stem = headword[:-2]
        template = f"{{{{grc-conj|pres-ew|act|{stem}}}}}"
    elif headword.endswith("άω") or headword.endswith("αω"):
        stem = headword[:-2]
        template = f"{{{{grc-conj|pres-aw|act|{stem}}}}}"
    elif headword.endswith("όω") or headword.endswith("οω"):
        stem = headword[:-2]
        template = f"{{{{grc-conj|pres-ow|act|{stem}}}}}"
    elif headword.endswith("μι"):
        # -μι verbs need more info, skip for now
        return set(), "mi-verb"
    else:
        return set(), f"unknown-ending:{headword[-3:]}"

    try:
        wtp.start_page(headword)
        html = wtp.expand(template)
    except Exception as e:
        return set(), str(e)

    return parse_html_forms(html, headword), ""


def parse_html_forms(html, headword):
    """Extract Greek word forms from expanded HTML table."""
    forms = set()
    # Find all text content that looks like Greek words
    # Strip HTML tags and extract Greek tokens
    text = re.sub(r'<[^>]+>', ' ', html)
    for token in re.split(r'[\s,/;]+', text):
        token = token.strip('.,;:()[]—– ')
        token = strip_length_marks(token)
        if (token and len(token) > 1 and
            any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF'
                for c in token)):
            forms.add(token)
    # Remove the headword label texts that aren't inflected forms
    noise = {"Active", "Middle", "Passive", "Indicative", "Subjunctive",
             "Optative", "Imperative", "Infinitive", "Participle",
             "Present", "Imperfect", "Future", "Aorist", "Perfect",
             "Pluperfect", "Singular", "Dual", "Plural",
             "Nominative", "Genitive", "Dative", "Accusative", "Vocative"}
    forms -= {n.lower() for n in noise}
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
        forms, err = expand_noun(wtp, word, entry["gender"], entry["genitive"])

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


def main():
    parser = argparse.ArgumentParser(description="Expand LSJ headwords via Wiktionary Lua")
    parser.add_argument("--setup", action="store_true",
                        help="Set up wikitextprocessor database (first run)")
    parser.add_argument("--test", action="store_true",
                        help="Test on overlap entries")
    parser.add_argument("--test-one", type=str, default=None,
                        help="Test a single word")
    parser.add_argument("--expand", action="store_true",
                        help="Expand LSJ-only entries and add to lookup")
    args = parser.parse_args()

    if args.setup:
        setup_wtp()
    elif args.test_one:
        test_one(args.test_one)
    elif args.test:
        test_overlap()
    elif args.expand:
        print("Not yet implemented - run --test first to validate")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
