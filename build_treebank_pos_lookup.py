"""Build POS-indexed disambiguation lookup from gold treebank data.

Reads CoNLL-U files from UD_Ancient_Greek-Perseus, UD_Ancient_Greek-PROIEL,
and DiGreC treebanks. Extracts genuinely ambiguous forms: same surface form
maps to different lemmas depending on UPOS tag.

Output: data/treebank_pos_lookup.json
Format: {form: {UPOS: lemma, ...}, ...}

Only forms that are genuinely ambiguous (2+ distinct UPOS->lemma mappings)
are included. Monotonic and lowercase variants are added for lookup cascade.
"""

import json
import unicodedata
from collections import defaultdict
from pathlib import Path

OPLA_DATA = Path.home() / "Documents" / "opla" / "data"
OUTPUT_PATH = Path(__file__).parent / "data" / "treebank_pos_lookup.json"

TREEBANK_DIRS = [
    OPLA_DATA / "UD_Ancient_Greek-Perseus",
    OPLA_DATA / "UD_Ancient_Greek-PROIEL",
    OPLA_DATA / "DiGreC",
]

# Reuse Dilemma's monotonic conversion
_POLYTONIC_STRIP = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
_POLYTONIC_TO_ACUTE = {0x0300, 0x0342}


def to_monotonic(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _POLYTONIC_STRIP:
            continue
        if cp in _POLYTONIC_TO_ACUTE:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(c for c in nfd if unicodedata.category(c) != "Mn"))


def parse_conllu(path: Path):
    """Yield (form, lemma, upos) tuples from a CoNLL-U file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tok_id = cols[0]
            # Skip multiword tokens (1-2) and empty nodes (1.1)
            if "-" in tok_id or "." in tok_id:
                continue
            form = cols[1]
            lemma = cols[2]
            upos = cols[3]
            # Skip punctuation
            if upos == "PUNCT":
                continue
            yield form, lemma, upos


def build_lookup():
    # Collect all (form, upos) -> {lemma: count} from treebanks
    # form_upos_lemmas[form][upos][lemma] = count
    form_upos_lemmas = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_tokens = 0

    for treebank_dir in TREEBANK_DIRS:
        if not treebank_dir.exists():
            print(f"  Skipping {treebank_dir} (not found)")
            continue
        conllu_files = sorted(treebank_dir.glob("*.conllu"))
        for f in conllu_files:
            count = 0
            for form, lemma, upos in parse_conllu(f):
                form_upos_lemmas[form][upos][lemma] += 1
                count += 1
            total_tokens += count
            print(f"  {f.name}: {count} tokens")

    print(f"\nTotal tokens: {total_tokens}")
    print(f"Unique forms: {len(form_upos_lemmas)}")

    # Filter to genuinely ambiguous forms:
    # A form is ambiguous if it has multiple DISTINCT (upos -> lemma) mappings,
    # meaning different UPOS tags lead to different lemmas.
    # For each UPOS, pick the most frequent lemma.
    lookup = {}
    for form, upos_dict in form_upos_lemmas.items():
        # For each UPOS, pick the most frequent lemma
        resolved = {}
        for upos, lemma_counts in upos_dict.items():
            best_lemma = max(lemma_counts, key=lemma_counts.get)
            resolved[upos] = best_lemma

        # Only keep forms where different UPOS tags map to different lemmas
        unique_lemmas = set(resolved.values())
        if len(unique_lemmas) < 2:
            continue

        lookup[form] = resolved

    print(f"Ambiguous forms (different UPOS -> different lemma): {len(lookup)}")

    # Add lowercase and monotonic variants
    extra = {}
    for form, upos_lemmas in list(lookup.items()):
        lower = form.lower()
        if lower != form and lower not in lookup:
            extra[lower] = upos_lemmas
        mono = to_monotonic(form.lower())
        if mono != form and mono != lower and mono not in lookup:
            extra[mono] = upos_lemmas

    lookup.update(extra)
    print(f"After adding lowercase/monotonic variants: {len(lookup)}")

    # Sort for stable output
    lookup = dict(sorted(lookup.items()))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=1)

    print(f"\nSaved to {OUTPUT_PATH}")
    return lookup


if __name__ == "__main__":
    build_lookup()
