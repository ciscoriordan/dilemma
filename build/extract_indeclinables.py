#!/usr/bin/env python3
"""Extract indeclinable words (adverbs, prepositions, conjunctions, particles)
from LSJ XML and save as form->lemma pairs for the lookup table.

Parses all greatscott*.xml files from ~/Documents/LSJLogeion/ and identifies
entries that are indeclinable based on POS markers in the XML.

Detection strategy:
- STRONG markers: entry text before first <sense> element says "Particle",
  "Prep.", "Conj.", "Interj.", or "exclamation" (after stripping <title>,
  <bibl>, and <etym> tags and parenthetical content to avoid false matches
  from bibliographic references and etymologies)
- WEAK markers: <pos>Adv.</pos> element before first <sense>, or "Adv." as
  plain text in pre-sense region, for entries without gender/adjective/verb
  markers
- Ending-based: entries with adverb-like suffixes (-θεν, -δε, -δόν, -δην,
  -τί, -κις, etc.) that also have "Adv." somewhere in the entry text

Filters:
- Skip entries with gender markers AND adjective itypes (ά/όν, ή/όν, etc.)
- Skip verb headwords (ending in -ω, -μι, -μαι, -σκω, etc.)
- Skip entries shorter than 2 characters
- Skip multi-word entries

Output: data/indeclinable_pairs.json with form->lemma mappings (both accented
and accent-stripped variants), excluding forms already in ag_lookup.json.

Usage:
    python extract_indeclinables.py
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expand_lsj import strip_diacritics, strip_length_marks, LSJ_DIR

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ADJ_ITYPE_SET = {"ον", "ή", "η", "ές", "εῖα", "ύ", "εια", "ά", "όν", "ός"}


def get_all_itypes(entry):
    """Get all itype text values from an entry."""
    itypes = []
    for it in entry.findall('.//itype'):
        t = ''.join(it.itertext()).strip().rstrip('.,; ')
        if t:
            itypes.append(t)
    return itypes


def has_adjective_itypes(itypes):
    """Check if itypes indicate an adjective (e.g. ά, όν or ον or ή)."""
    for it in itypes:
        if it in ADJ_ITYPE_SET:
            return True
    if len(itypes) >= 2:
        stripped = {strip_diacritics(t) for t in itypes}
        if 'ον' in stripped or 'η' in stripped:
            return True
    return False


def clean_pre_sense(xml_str):
    """Strip bibliographic, title, and etymological content to avoid
    false POS matches from references like A.D. Conj. or (neg. Particle)."""
    s = re.sub(r'<title[^>]*>.*?</title>', '', xml_str, flags=re.DOTALL)
    s = re.sub(r'<bibl[^>]*>.*?</bibl>', '', s, flags=re.DOTALL)
    s = re.sub(r'<etym[^>]*>.*?</etym>', '', s, flags=re.DOTALL)
    s = re.sub(r'\([^)]*\)', '', s)
    return s


def is_verb_headword(hw_plain):
    """Check if headword looks like a verb based on its ending."""
    VERB_ENDINGS = ['ομαι', 'αομαι', 'εομαι', 'οομαι', 'νυμι', 'σκω',
                    'αζω', 'ιζω', 'υζω', 'αινω', 'ειρω', 'ττω', 'σσω', 'πτω',
                    'αζομαι', 'ιζομαι', 'εω', 'αω', 'οω', 'ζω',
                    'λλω', 'ρω', 'νω', 'πω', 'βω', 'φω', 'γω', 'χω',
                    'μαι', 'μι', 'ννυμι', 'λλυμι']
    for ve in sorted(VERB_ENDINGS, key=len, reverse=True):
        if hw_plain.endswith(ve) and len(hw_plain) > len(ve) + 1:
            return True
    return False


def get_pre_sense_text(xml_str):
    """Get the XML text before the first <sense> element."""
    m = re.search(r'<sense\b', xml_str)
    if m:
        return xml_str[:m.start()]
    return xml_str[:500]


def extract_indeclinables():
    """Scan LSJ XML and extract indeclinable entries."""
    print("Deep-scanning LSJ XML for indeclinable entries...")

    indeclinables = {}
    stats = Counter()

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
            if ' ' in hw:
                continue

            hw_plain = strip_diacritics(hw)
            if len(hw_plain) < 2:
                stats['skipped_short'] += 1
                continue

            gen_elem = entry.find('.//gen')
            has_gender = (gen_elem is not None and
                          ''.join(gen_elem.itertext()).strip() != "")
            all_itypes = get_all_itypes(entry)
            is_adj = has_adjective_itypes(all_itypes)
            is_verb = is_verb_headword(hw_plain)

            xml_str = ET.tostring(entry, encoding='unicode')
            pre_sense = get_pre_sense_text(xml_str)
            pre_clean = clean_pre_sense(pre_sense)

            category = None
            is_strong = False

            # ---- STRONG detection: explicit POS marker before first <sense> ----
            if 'Particle' in pre_clean and not is_verb:
                category = 'particle'
                is_strong = True
            elif re.search(r'\bPrep\.', pre_clean) and not is_verb:
                category = 'preposition'
                is_strong = True
            elif re.search(r'\bConj\.', pre_clean) and not is_verb:
                category = 'conjunction'
                is_strong = True
            elif re.search(r'[Ii]nterj\.', pre_clean) and not is_verb:
                category = 'interjection'
                is_strong = True
            elif ('exclamation' in pre_clean[:300] and not is_verb and
                  len(hw_plain) <= 6):
                category = 'interjection'
                is_strong = True

            # Skip if strongly marked but clearly an adjective with gender
            if is_strong and has_gender and is_adj:
                category = None
                is_strong = False

            # ---- WEAK: <pos>Adv.</pos> before first <sense> ----
            if category is None and not is_verb:
                pos_elem = entry.find('.//pos')
                if pos_elem is not None:
                    pos_text = ''.join(pos_elem.itertext()).strip()
                    if pos_text.lower().startswith('adv'):
                        pos_xml = ET.tostring(pos_elem, encoding='unicode')
                        if pos_xml in pre_sense:
                            if not has_gender and not is_adj:
                                adj_endings = ['ος', 'ον', 'ης', 'ικος',
                                               'ικη', 'ικον']
                                is_adj_like = any(hw_plain.endswith(e)
                                                  for e in adj_endings)
                                if not is_adj_like:
                                    category = 'adverb'

            # ---- WEAK: "Adv." as text in pre-sense ----
            if (category is None and not has_gender and not is_adj and
                    not is_verb):
                if re.search(r'\bAdv\.', pre_clean):
                    adj_endings = ['ος', 'ον', 'ης', 'ικος', 'ικη', 'ικον']
                    is_adj_like = any(hw_plain.endswith(e)
                                      for e in adj_endings)
                    if not is_adj_like:
                        category = 'adverb'

            # ---- WEAK: adverb-like ending + Adv. in entry ----
            if (category is None and not has_gender and not all_itypes and
                    not is_verb):
                adv_suffixes = ['θεν', 'δε', 'σε', 'ζε', 'χου', 'χοι',
                                'ποι', 'τι', 'δι', 'ει', 'χι', 'δον', 'δην',
                                'κις', 'ακις', 'οθι', 'οσε', 'ηδον', 'αδον',
                                'ωδον', 'αδα', 'ηδα', 'ινδα', 'ωθεν', 'ηθεν']
                has_adv_ending = any(
                    hw_plain.endswith(suffix) and
                    len(hw_plain) > len(suffix) + 1
                    for suffix in adv_suffixes
                )
                if has_adv_ending:
                    entry_text_clean = clean_pre_sense(xml_str[:1500])
                    if 'Adv.' in entry_text_clean:
                        category = 'adverb'

            if category:
                indeclinables[hw] = category
                stats[category] += 1

    return indeclinables, stats


def main():
    indeclinables, stats = extract_indeclinables()

    print(f"\nFound {len(indeclinables):,} indeclinable entries:")
    for cat, count in stats.most_common():
        if cat.startswith('skipped'):
            continue
        print(f"  {cat}: {count}")
    print(f"  (skipped short: {stats['skipped_short']})")

    for cat in ['adverb', 'preposition', 'conjunction', 'particle',
                'interjection']:
        examples = [hw for hw, c in indeclinables.items() if c == cat][:10]
        if examples:
            print(f"\n  {cat} examples: {', '.join(examples)}")

    # Load existing lookup
    ag_lookup_path = DATA_DIR / "ag_lookup.json"
    print(f"\nLoading {ag_lookup_path}...")
    with open(ag_lookup_path, encoding="utf-8") as f:
        existing_lookup = json.load(f)
    print(f"  {len(existing_lookup):,} existing entries")

    # Build new pairs, filtering already-present entries
    new_pairs = {}
    already_present = 0
    for hw, category in indeclinables.items():
        if hw not in existing_lookup:
            new_pairs[hw] = hw
        else:
            already_present += 1

        plain = strip_diacritics(hw)
        if plain != hw and plain not in existing_lookup:
            new_pairs[plain] = hw

    print(f"\n  Indeclinable headwords already in lookup: {already_present}")
    print(f"  New form->lemma pairs to add: {len(new_pairs):,}")

    new_by_cat = Counter()
    for hw, cat in indeclinables.items():
        if (hw not in existing_lookup or
                (strip_diacritics(hw) != hw and
                 strip_diacritics(hw) not in existing_lookup)):
            new_by_cat[cat] += 1
    print(f"  New entries by category:")
    for cat, count in new_by_cat.most_common():
        print(f"    {cat}: {count}")

    # Save
    out_path = DATA_DIR / "indeclinable_pairs.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_pairs, f, ensure_ascii=False, indent=None)
    size_kb = out_path.stat().st_size / 1024
    print(f"\nSaved {len(new_pairs):,} pairs to {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
