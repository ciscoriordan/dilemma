#!/usr/bin/env python3
"""Benchmark multiple Greek lemmatizers on the DBBE dataset.

Compares Dilemma, stanza (el + grc), spaCy (el_core_news_sm), and CLTK
(BackoffGreekLemmatizer) on 10K tokens of Byzantine Greek epigrams.

All tools are evaluated with the same normalization and equivalence logic
to ensure fair comparison.
"""

import json
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DBBE_PATH = SCRIPT_DIR / "data" / "dbbe" / "lingAnn_GS_medievalGreek.tsv"
EQUIV_PATH = SCRIPT_DIR / "data" / "lemma_equivalences.json"

# DBBE POS category (first char) -> UPOS mapping
DBBE_TO_UPOS = {
    "n": "NOUN", "v": "VERB", "a": "ADJ", "l": "DET",
    "g": "PART", "d": "ADV", "p": "PRON", "r": "ADP",
    "c": "CCONJ", "m": "NUM", "i": "INTJ", "e": "INTJ",
}


def to_monotonic(s):
    _strip = {0x0313, 0x0314, 0x0345, 0x0306, 0x0304}
    _to_acute = {0x0300, 0x0342}
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        cp = ord(ch)
        if cp in _strip:
            continue
        if cp in _to_acute:
            out.append("\u0301")
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def strip_accents(s):
    nfd = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC",
        "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn"))


def parse_dbbe(tsv_path):
    tokens = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            form, pos, lemma = parts
            if pos.startswith("u"):
                continue
            tokens.append({
                "form": form,
                "gold_lemma": lemma,
                "pos": pos[0] if pos else "?",
                "pos_full": pos,
            })
    return tokens


def load_equiv_map():
    equiv_map = {}
    if EQUIV_PATH.exists():
        with open(EQUIV_PATH, encoding="utf-8") as f:
            equiv_data = json.load(f)
        for group in equiv_data["groups"]:
            group_set = set(group)
            for lemma in group:
                equiv_map[lemma] = group_set
                equiv_map[strip_accents(lemma.lower())] = {
                    strip_accents(l.lower()) for l in group
                }
    return equiv_map


def score(predictions, tokens, equiv_map):
    """Score predictions against gold. Returns dict of metrics."""
    assert len(predictions) == len(tokens)
    strict = mono = stripped = equiv = 0
    for pred, t in zip(predictions, tokens):
        gold = t["gold_lemma"]
        # Strict: exact case-sensitive polytonic match
        if pred == gold:
            strict += 1
            mono += 1
            stripped += 1
            equiv += 1
            continue
        # Monotonic: lowercased + monotonic normalized
        pred_m = to_monotonic(pred).lower()
        gold_m = to_monotonic(gold).lower()
        if pred_m == gold_m:
            mono += 1
            stripped += 1
            equiv += 1
            continue
        # Stripped: lowercased + all accents removed
        pred_s = strip_accents(pred.lower())
        gold_s = strip_accents(gold.lower())
        if pred_s == gold_s:
            stripped += 1
            equiv += 1
            continue
        # Equivalence: check lemma equivalence groups
        pred_l = pred.lower()
        gold_l = gold.lower()
        if pred_l in equiv_map and gold_l in equiv_map.get(pred_l, set()):
            equiv += 1
            continue
        if pred_s in equiv_map and gold_s in equiv_map.get(pred_s, set()):
            equiv += 1
            continue
        if pred in equiv_map and gold in equiv_map.get(pred, set()):
            equiv += 1
    n = len(tokens)
    return {
        "n": n,
        "strict": strict, "strict_pct": strict / n * 100,
        "mono": mono, "mono_pct": mono / n * 100,
        "stripped": stripped, "stripped_pct": stripped / n * 100,
        "equiv": equiv, "equiv_pct": equiv / n * 100,
    }


def errors_list(predictions, tokens, equiv_map, limit=20):
    """Return list of error dicts for analysis."""
    errs = []
    for pred, t in zip(predictions, tokens):
        gold = t["gold_lemma"]
        pred_s = strip_accents(pred.lower())
        gold_s = strip_accents(gold.lower())
        # Check if it's an equiv match
        is_equiv = False
        if pred.lower() in equiv_map and gold.lower() in equiv_map.get(pred.lower(), set()):
            is_equiv = True
        if pred_s in equiv_map and gold_s in equiv_map.get(pred_s, set()):
            is_equiv = True

        if pred_s != gold_s and not is_equiv:
            errs.append({
                "form": t["form"],
                "gold": gold,
                "pred": pred,
                "pos": t["pos"],
            })
            if len(errs) >= limit:
                break
    return errs


# ── Tool runners ──

def run_dilemma(tokens, use_pos=False):
    """Run Dilemma lemmatizer."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from dilemma import Dilemma
    d = Dilemma(resolve_articles=True)

    # Monkey-patch for greedy decoding
    orig_predict = d._predict
    def _greedy_predict(words, num_beams=1):
        return orig_predict(words, num_beams=1)
    d._predict = _greedy_predict

    forms = [t["form"] for t in tokens]
    batch_size = 500

    if use_pos:
        upos_tags = [DBBE_TO_UPOS.get(t["pos"], "X") for t in tokens]
        preds = []
        for i in range(0, len(forms), batch_size):
            preds.extend(d.lemmatize_batch_pos(
                forms[i:i+batch_size], upos_tags[i:i+batch_size]))
        return preds
    else:
        preds = []
        for i in range(0, len(forms), batch_size):
            preds.extend(d.lemmatize_batch(forms[i:i+batch_size]))
        return preds


def run_stanza(tokens, lang="el"):
    """Run stanza lemmatizer (el or grc model)."""
    import stanza
    nlp = stanza.Pipeline(
        lang, processors='tokenize,pos,lemma',
        tokenize_pretokenized=True, use_gpu=False, verbose=False
    )
    forms = [t["form"] for t in tokens]
    # stanza expects sentences as lists of tokens
    # Process in chunks to avoid memory issues
    chunk_size = 200
    preds = []
    for i in range(0, len(forms), chunk_size):
        chunk = forms[i:i+chunk_size]
        # Each token as its own "sentence" to avoid context effects
        # Actually, feed them as one sentence to allow POS context
        doc = nlp([chunk])
        for sent in doc.sentences:
            for word in sent.words:
                preds.append(word.lemma if word.lemma else word.text)
    return preds


def run_spacy(tokens):
    """Run spaCy el_core_news_sm lemmatizer."""
    import spacy
    nlp = spacy.load("el_core_news_sm")
    forms = [t["form"] for t in tokens]
    preds = []
    # Process in chunks
    chunk_size = 200
    for i in range(0, len(forms), chunk_size):
        chunk = forms[i:i+chunk_size]
        # Join with spaces to create a doc, but use pre-tokenized approach
        # spaCy doesn't have a simple pre-tokenized API, so use Doc
        from spacy.tokens import Doc
        doc = Doc(nlp.vocab, words=chunk)
        # Run the pipeline (tagger, etc.)
        for name, proc in nlp.pipeline:
            doc = proc(doc)
        for token in doc:
            preds.append(token.lemma_ if token.lemma_ else token.text)
    return preds


def run_cltk(tokens):
    """Run CLTK BackoffGreekLemmatizer."""
    from cltk.lemmatize.grc import GreekBackoffLemmatizer
    lem = GreekBackoffLemmatizer()
    forms = [t["form"] for t in tokens]
    results = lem.lemmatize(forms)
    return [lemma for _, lemma in results]


def main():
    print("=" * 80)
    print("DBBE Benchmark: Greek Lemmatizer Comparison")
    print("=" * 80)

    tokens = parse_dbbe(DBBE_PATH)
    print(f"\nDataset: {len(tokens)} tokens (punctuation excluded)")
    print(f"Source: DBBE (Database of Byzantine Book Epigrams)")
    print(f"Genre: Unedited Byzantine Greek epigrams")

    equiv_map = load_equiv_map()
    print(f"Lemma equivalence groups: {len([k for k in equiv_map if k == strip_accents(k.lower())])}")

    pos_counts = Counter(t["pos"] for t in tokens)
    print(f"\nPOS breakdown: ", end="")
    pos_names = {"v": "verb", "n": "noun", "a": "adj", "l": "det",
                 "r": "prep", "d": "adv", "c": "conj", "p": "pron",
                 "m": "num", "i": "intj", "g": "particle", "e": "exclam"}
    parts = []
    for pos, count in pos_counts.most_common():
        parts.append(f"{pos_names.get(pos, pos)}={count}")
    print(", ".join(parts))

    # Define tools to benchmark
    tools = [
        {
            "name": "Dilemma (no POS)",
            "func": lambda: run_dilemma(tokens, use_pos=False),
            "uses_pos": False,
            "training_data": "3.4M pairs + 9.7M lookup",
            "notes": "Character-level encoder-decoder + large lookup table",
        },
        {
            "name": "Dilemma (gold POS)",
            "func": lambda: run_dilemma(tokens, use_pos=True),
            "uses_pos": True,
            "training_data": "3.4M pairs + 9.7M lookup",
            "notes": "Same model, POS-aware disambiguation",
        },
        {
            "name": "stanza (grc)",
            "func": lambda: run_stanza(tokens, lang="grc"),
            "uses_pos": True,
            "training_data": "AG treebanks (AGLDT + Perseus, ~310K tokens)",
            "notes": "Ancient Greek model, uses own POS tagger",
        },
        {
            "name": "stanza (el)",
            "func": lambda: run_stanza(tokens, lang="el"),
            "uses_pos": True,
            "training_data": "GDT treebank (~63K tokens)",
            "notes": "Modern Greek model, uses own POS tagger",
        },
        {
            "name": "spaCy (el)",
            "func": lambda: run_spacy(tokens),
            "uses_pos": True,
            "training_data": "~30K tokens (MG news)",
            "notes": "Modern Greek only, rule-based lemmatizer",
        },
        {
            "name": "CLTK (grc)",
            "func": lambda: run_cltk(tokens),
            "uses_pos": False,
            "training_data": "Perseus treebank (~310K tokens)",
            "notes": "Backoff lemmatizer: dictionary + regex + identity",
        },
    ]

    results_all = {}
    for tool in tools:
        print(f"\n{'─' * 60}")
        print(f"Running: {tool['name']}...")
        t0 = time.time()
        try:
            preds = tool["func"]()
            elapsed = time.time() - t0
            if len(preds) != len(tokens):
                print(f"  WARNING: got {len(preds)} predictions for {len(tokens)} tokens")
                # Pad or truncate
                if len(preds) < len(tokens):
                    preds.extend([""] * (len(tokens) - len(preds)))
                else:
                    preds = preds[:len(tokens)]
            s = score(preds, tokens, equiv_map)
            results_all[tool["name"]] = {
                "scores": s,
                "preds": preds,
                "elapsed": elapsed,
                "uses_pos": tool["uses_pos"],
                "training_data": tool["training_data"],
                "notes": tool["notes"],
            }
            print(f"  Time: {elapsed:.1f}s")
            print(f"  strict={s['strict_pct']:.1f}%  "
                  f"mono={s['mono_pct']:.1f}%  "
                  f"stripped={s['stripped_pct']:.1f}%  "
                  f"equiv={s['equiv_pct']:.1f}%")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"\n{'Tool':<25s} {'POS':>4s} {'Strict':>7s} {'Mono':>7s} "
          f"{'Stripped':>8s} {'Equiv':>7s} {'Time':>6s}")
    print(f"{'─' * 25} {'─' * 4} {'─' * 7} {'─' * 7} {'─' * 8} {'─' * 7} {'─' * 6}")
    for name, r in results_all.items():
        s = r["scores"]
        pos = "yes" if r["uses_pos"] else "no"
        print(f"{name:<25s} {pos:>4s} {s['strict_pct']:>6.1f}% {s['mono_pct']:>6.1f}% "
              f"{s['stripped_pct']:>7.1f}% {s['equiv_pct']:>6.1f}% {r['elapsed']:>5.1f}s")

    # Error analysis for each tool
    print(f"\n\n{'=' * 80}")
    print("ERROR EXAMPLES (first 10 per tool)")
    print(f"{'=' * 80}")
    for name, r in results_all.items():
        errs = errors_list(r["preds"], tokens, equiv_map, limit=10)
        print(f"\n{name} ({len(errs)} shown):")
        for e in errs:
            print(f"  {e['form']:20s} gold={e['gold']:20s} pred={e['pred']:20s} [{e['pos']}]")

    # Save results as JSON for later use
    output = {}
    for name, r in results_all.items():
        output[name] = {
            "scores": r["scores"],
            "elapsed": r["elapsed"],
            "uses_pos": r["uses_pos"],
            "training_data": r["training_data"],
            "notes": r["notes"],
        }
    out_path = SCRIPT_DIR / "data" / "dbbe" / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
