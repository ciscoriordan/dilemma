# Dilemma <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/ancient-greece.svg" width="28" alt="Ancient Greece"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/languages/el.svg" width="28" alt="Greek"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/countries/cy.svg" width="28" alt="Cyprus"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/byzantine.svg" width="28" alt="Byzantine">

<p align="center">
  <img width="500" alt="dilemma" src="dilemma.png">
</p>

Greek lemmatizer with a 5.2 million form lookup table and a ~4M
parameter character-level transformer trained on 3.2 million Wiktionary
inflection pairs spanning Modern Greek, Ancient Greek, and Medieval Greek.

Most Greek words resolve instantly via the lookup table. For unseen forms,
Dilemma uses a small encoder-decoder transformer that learns morphological
patterns at the character level, the standard architecture from
[SIGMORPHON](https://sigmorphon.github.io/) shared tasks. At 4M parameters
it trains from scratch in minutes and runs inference in under a millisecond,
compared to fine-tuning approaches like *ByT5-small* (300M params) which take
hours to train and ~10ms per word. Greek lemmatization is highly
pattern-based - a small specialized model matches a large general-purpose
one, and the 5.2M lookup table handles the rest.

**ONNX support:** Dilemma can run without PyTorch. When ONNX model files
are present, inference uses ONNX Runtime (~50 MB) instead of PyTorch (~2 GB).
The lookup table (which handles 95%+ of words) needs neither.

Handles Standard Modern Greek, Katharevousa, Cypriot, Cretan, and other
regional varieties alongside Ancient and Medieval Greek. Existing
lemmatizers (*stanza*, *spaCy*) are trained on ~30K tokens of modern news
and fail on anything outside standard SMG. Dilemma trains on **100x more
data** from all three periods of the language.

### Modern Greek varieties

| Variety | Tagged entries |
|---------|---------------|
| **Standard Modern Greek (SMG/Demotic)** | 877K entries (core) |
| **Katharevousa** | 283+ tagged, hundreds more formal/place terms |
| **Cretan** | 273 |
| **Cypriot** | 199 |
| **Heptanesian (Ionian)** | 18 |
| **Maniot** | 3 |
| **Medieval/Byzantine** | 3K (separate dump) |

### Ancient Greek varieties

| Variety | Tagged entries |
|---------|---------------|
| **Epic/Homeric** | 3,755 |
| **Ionic** | 1,638 |
| **Attic** | 1,279 |
| **Koine** | 1,209 |
| **Byzantine** | 496 |
| **Doric** | 456 |
| **Aeolic** | 163 |
| **Laconian** | 52 |
| **Boeotian** | 15 |
| **Arcadocypriot** | 11 |

The tagged entry counts above are Wiktionary headwords explicitly labeled
with a variety. Each headword generates a full inflection paradigm (10-40
forms for verbs, 4-8 for nouns), so the actual form coverage is much
larger: **2.8M MG forms, 2.4M AG forms, 5.2M combined**.

Beyond the lookup, the transformer model generalizes to forms not in
Wiktionary. Katharevousa forms are the primary non-SMG target - they mix
AG morphology (augments, 3rd declension genitives) with MG vocabulary.
The strong Epic/Homeric coverage (3,755 entries) is directly relevant for
literary texts based on Homer.

> `εσκότωσε` → `σκοτώνω` · `πολεμούσαν` → `πολεμώ` · `δώση` → `δίνω`

<p align="center">
  <img src="https://raw.githubusercontent.com/ciscoriordan/dilemma/main/diagram.svg" width="700" alt="Dilemma architecture">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ciscoriordan/dilemma/main/examples.svg" width="700" alt="Lemmatization examples">
</p>

---

## Quick Start

```python
from dilemma import Dilemma

d = Dilemma()                                  # all periods (default)
d.lemmatize("εσκότωσε")                       # "σκοτώνω"
d.lemmatize("πάθης")                          # "παθαίνω"
d.lemmatize_batch(["δώση", "σκότωσε"])        # ["δίνω", "σκοτώνω"]

# Elision expansion (AG elided forms resolved via Wiktionary lookup)
d.lemmatize("ἀλλ̓")                            # "ἀλλά"
d.lemmatize("ἔφατ̓")                           # "φημί"
d.lemmatize("δ̓")                              # "δέ"
d.lemmatize("ἐπ̓")                             # "ἐπί"

# Single period
d_mg = Dilemma(lang="el")                     # MG only
d_grc = Dilemma(lang="grc")                   # AG only

# Specific model scale
d = Dilemma(scale=1)                          # use scale 1 model

# Treebank evaluation mode: resolve articles to ὁ, pronouns to ἐγώ/σύ
d_eval = Dilemma(resolve_articles=True)
d_eval.lemmatize("τῆς")                       # "ὁ" (not "τῆς")
d_eval.lemmatize("μοι")                       # "ἐγώ" (not "μοι")
```

By default, articles and pronoun clitics self-map (e.g. `τῆς` returns
`τῆς`). This is better for alignment pipelines where you want
surface-form matching. Set `resolve_articles=True` to resolve them
to canonical lemmas (`ὁ`, `ἐγώ`, `σύ`), matching treebank conventions
(AGDT, DiGreC, PROIEL).

### Verbose mode

For ambiguous forms, `lemmatize_verbose` returns all candidates with
metadata so downstream tools can disambiguate using context:

```python
from dilemma import Dilemma

d = Dilemma()

# Proper noun vs common noun: Ἔρις (goddess) vs ἔρις (strife)
candidates = d.lemmatize_verbose("ἔριδι")
for c in candidates:
    print(f"{c.lemma:10s} lang={c.lang} proper={c.proper} via={c.via}")
# Ἔρις       lang=grc proper=True  via=exact

# Multiple language matches
candidates = d.lemmatize_verbose("πόλεμο")
# -> [LemmaCandidate(lemma="πόλεμος", lang="el", ...),
#     LemmaCandidate(lemma="πόλεμος", lang="grc", ...)]

# Elision with multiple valid expansions
candidates = d.lemmatize_verbose("δ̓")
# -> [LemmaCandidate(lemma="δέ", source="elision", via="elision:ε"),
#     LemmaCandidate(lemma="δή", source="elision", via="elision:η"), ...]
```

Each `LemmaCandidate` has:
- `lemma` — the lemma string
- `lang` — `"el"` (SMG), `"grc"` (AG), `"med"` (Medieval)
- `proper` — `True` if lemma is a proper noun (capitalized headword)
- `source` — `"lookup"`, `"elision"`, `"crasis"`, `"model"`, `"identity"`
- `via` — how it matched: `"exact"`, `"lower"`, `"elision:ε"`, `"+case_alt"`, etc.
- `score` — `1.0` for lookup, `0.5` for model, `0.0` for identity fallback

### Elision expansion

Ancient Greek texts frequently elide final vowels before a following
vowel, marking the elision with an apostrophe (U+0313 in polytonic
encoding). Dilemma resolves these by stripping the elision mark and
trying each Greek vowel against the lookup table:

| Elided | Expanded | Lemma |
|--------|----------|-------|
| `ἀλλ̓` | `ἀλλά` | `ἀλλά` |
| `δ̓` | `δέ` | `δέ` |
| `τ̓` | `τε` | `τε` |
| `ἐπ̓` | `ἐπί` | `ἐπί` |
| `ἔφατ̓` | `ἔφατο` | `φημί` |
| `κατ̓` | `κατά` | `κατά` |
| `βάλλ̓` | `βάλλε` | `βάλλω` |

Polytonic input automatically restricts expansion to the AG lookup
table, avoiding false matches from MG monotonic forms. Candidates are
ranked by vowel frequency in elision contexts (ε, α, ο most common).

## How It Works

| Layer | Speed | Coverage | Source |
|-------|-------|----------|--------|
| **Lookup table** | instant | 5.2M+ known forms | Wiktionary inflection paradigms + treebanks |
| **Elision expansion** | instant | AG elided forms | vowel expansion against lookup |
| **Crasis table** | instant | ~50 common crasis forms | hand-curated |
| **Transformer** | <1ms/word | generalizes to unseen forms | trained on Wiktionary pairs |
The lookup table is built from all 5 Wiktionary [kaikki dumps](https://kaikki.org/)
(EN and EL editions for MG and AG, plus EL Medieval Greek), augmented
with form-lemma pairs from gold-standard treebanks (Gorman, AGDT).
Each form is indexed under its original, monotonic, and accent-stripped
variants, so `θεοὶ` (polytonic with grave), `θεοί` (monotonic with
acute), and `θεοι` (stripped) all resolve to `θεός`. Input can be
polytonic, monotonic, or unaccented. MG forms take priority, then
Medieval, then AG.

When the transformer handles an unseen form, beam search generates
multiple candidates and picks the first that matches a known headword
from Wiktionary, [LSJ](https://github.com/helmadik/LSJLogeion) (116K
headwords), or Cunliffe's Homeric Lexicon (11K headwords). If nothing
matches, the input is returned unchanged.

**Wiktionary as upstream:** Because Dilemma's lookup tables are built
directly from Wiktionary, any missing or incorrect lemmatization can
often be fixed by editing the Wiktionary entry itself. When the kaikki
dumps are next regenerated and `build_data.py` re-run, the fix flows
into Dilemma automatically. This means the coverage and accuracy of
Dilemma improve over time as Wiktionary's Greek coverage improves,
without any changes to Dilemma's code.

The transformer is a small (~4M param) character-level encoder-decoder,
the standard architecture from
[SIGMORPHON](https://sigmorphon.github.io/) morphological inflection
shared tasks. It learns character-level patterns and generalizes to forms
not in Wiktionary. Training on MG + AG + Medieval data means the model
sees AG augment patterns (`ἔλυσε` → `λύω`) alongside MG stem
transformations (`σκότωσε` → `σκοτώνω`). For katharevousa forms like
`εσκότωσε`, it has both signals to draw from.

## Installation

### Inference only (no GPU needed)

```bash
git clone https://github.com/ciscoriordan/dilemma.git && cd dilemma
pip install onnxruntime                # ~50 MB, no PyTorch needed
python build_data.py --download        # downloads Wiktionary dumps, builds lookup tables
```

The lookup table handles 95%+ of words with no model at all. For the
remaining ~5% (unseen forms), the ONNX model files (`encoder.onnx`,
`decoder_step.onnx`) in `model/combined-s3/` provide transformer
inference without PyTorch. If these files aren't present, install
PyTorch and run `python export_onnx.py` to generate them from the
`.pt` checkpoint.

### With PyTorch (for training or if ONNX files aren't available)

```bash
pip install torch                      # ~2 GB
python build_data.py --download
```

Dilemma auto-detects: if ONNX files exist, uses ONNX Runtime. Otherwise
falls back to PyTorch. Both produce identical output.

---

## Training

### 1. Build data

Downloads all 5 kaikki dumps and extracts every form-lemma pair from
inflection tables. Non-Greek characters are filtered out.

```bash
pip install -r requirements.txt
python build_data.py --download             # downloads + extracts (~1.5GB total)
```

<details>
<summary>Already have the dumps locally?</summary>

```bash
python build_data.py --kaikki /path/to/kaikki/dumps
```

By default, `build_data.py` looks in `~/Documents/Klisy/word_collector/`
(configurable via `KLISY_DIR` env var or `--kaikki` flag).
</details>

### 2. Train

Trains the character-level transformer on the extracted pairs. Use
`--scale` to match your GPU and time budget.

```bash
python train.py --scale 1                   # default (15 sec)
python train.py --scale 2                   # recommended (13 min on 2080 Ti)
python train.py --scale 3                   # full data (~45 min)
```

### Training scales

Every scale includes **100% of non-standard varieties** (Medieval,
Katharevousa, Cypriot, Cretan, Maniot, Heptanesian, archaic, dialectal).
The remaining budget is split 50/50 between Ancient Greek and standard MG.

| Scale | Training pairs | Varieties | AG | SMG | Time (2080 Ti) | Eval | Tests |
|:-----:|---------------:|----------:|-------:|-------:|:--------------:|:----:|:-----:|
| 1 | 20K | 9K (100%) | 5.5K | 5.5K | 16 sec | 2.6% | 53/55 |
| 2 | 1M | 9K (100%) | 496K | 496K | 13 min | 62% | 54/55 |
| 3 | 3.2M (all) | 9K (100%) | 1.5M (100%) | 1.7M (100%) | 36 min | 75% | 55/55 |

Eval accuracy is the model's score on held-out pairs *without* the
lookup table. In practice, the lookup resolves most forms instantly
and the model only handles truly novel words. When the model is used,
beam search generates 4 candidates and the first one that matches a
known headword in the lookup wins. If none match, the input is returned
unchanged (safe fallback).

Tests are a 55-case suite covering SMG, Epic, Attic, Koine, Byzantine,
Katharevousa, crasis, and model fallback across all resolution paths.

Models are saved to `model/combined-s0/`, `model/combined-s1/`, etc.
`Dilemma()` auto-detects the best available scale, or you can specify one:

```python
d = Dilemma(scale=0)                  # use scale 0 model explicitly
d = Dilemma()                         # auto-detect highest available
```

Medieval/Byzantine Greek is treated as part of Modern Greek, not a
separate language. The `"el"` mode includes Medieval forms alongside
SMG, Katharevousa, and regional varieties. The Medieval corpus (~3K
entries) covers Byzantine-era morphology that feeds directly into
Katharevousa and formal MG.

### Language codes

| Code | Period | ISO standard |
|------|--------|-------------|
| `el` | Modern Greek (including Medieval, Katharevousa, regional) | ISO 639-1 |
| `grc` | Ancient Greek (Homer through late antiquity) | ISO 639-2 |
| `med` | Medieval/Byzantine Greek (~300-1453 CE) | No ISO code exists (proposed `gkm` was [rejected](https://iso639-3.sil.org/request/2006-084)); `med` is an internal shorthand |

`med` appears in `LemmaCandidate.lang` when a form is found only in
the Medieval lookup table. In practice, Medieval forms are grouped
with `el` for lookup priority since Byzantine morphology is the direct
ancestor of Modern Greek.

Note: [Opla](https://github.com/ciscoriordan/opla) (POS tagging +
dependency parsing) handles `med` differently, grouping it with `grc`
instead of `el`. This is intentional - lemmatization and syntactic
analysis have different grouping needs. Medieval *morphology* (inflection
patterns, form lookup) is closer to Modern Greek, but Medieval *syntax*
(polytonic script, full case system, optative mood) is closer to Ancient
Greek. Each tool groups `med` with whichever period best serves its task.

### 3. Export to ONNX (optional)

Generates ONNX model files so inference works without PyTorch.

```bash
python export_onnx.py                  # exports encoder.onnx + decoder_step.onnx
```

### GPU quick start

```bash
git clone https://github.com/ciscoriordan/dilemma.git && cd dilemma
pip install -r requirements.txt
python build_data.py --download
python train.py --scale 1
python export_onnx.py                  # optional: enable PyTorch-free inference
```

## Architecture

Small character-level encoder-decoder transformer (~4M parameters),
trained from scratch on Greek lemmatization pairs. This is the standard
architecture from [SIGMORPHON](https://sigmorphon.github.io/)
morphological inflection shared tasks.

| Component | Config |
|-----------|--------|
| Encoder | 3 transformer layers, 256 hidden, 4 heads |
| Decoder | 3 transformer layers, 256 hidden, 4 heads |
| FFN | 512 dim |
| Vocabulary | ~160 Greek characters + special tokens |
| Parameters | ~4M |
| Inference | <1ms/word (GPU), ~2ms/word (CPU) |

No pretrained weights,the model is small enough to train from scratch
on 500K+ pairs in minutes. The character vocabulary covers all Greek
Unicode ranges (monotonic, polytonic, extended).

### Why not *ByT5*?

An earlier version of Dilemma fine-tuned Google's
[*ByT5-small*](https://huggingface.co/google/byt5-small) (300M params).
*ByT5* processes raw UTF-8 bytes, so a 10-character Greek word becomes
~20 encoder steps. The custom transformer uses a Greek character
vocabulary (~160 tokens), so the same word is ~10 steps. Combined with
75x fewer parameters:

|  | ByT5-small | Dilemma |
|--|:----------:|:-------:|
| Parameters | 300M | 4M |
| Training (500K pairs, 3 epochs) | ~4 hours | ~10 min |
| Training (3.4M pairs, 3 epochs) | ~20 hours | ~1 hour |
| Inference | ~10ms/word | <1ms/word |
| Dependencies | torch + transformers | torch only |

The custom model trains **10-20x faster** and runs **10x faster** at
inference, with no loss in accuracy for this task. Greek lemmatization
is highly pattern-based,a small specialized model matches a large
general-purpose one.

## Data

| Source | Dump size | Training Pairs | Lookup Forms |
|--------|-----------|----------------|--------------|
| EN + EL Wiktionary (MG) | 1.2 GB | 1.7M | 2.8M |
| EN + EL Wiktionary (AG) | 339 MB | 1.5M | 2.4M |
| EL Wiktionary (Medieval) | 3.7 MB | 4.3K | 6.9K |
| **Combined** | **1.5 GB** | **3.2M** | **5.2M** |

All data is extracted automatically from
[kaikki.org](https://kaikki.org/) Wiktionary JSONL dumps (EN and EL
editions). Each form is indexed under its original, monotonic, and
accent-stripped variants for fuzzy matching.

### Extraction sources

Form-lemma pairs come from three sources per Wiktionary entry:

1. **Inflection tables** (primary). Every cell in a verb conjugation or
   noun declension table becomes a form-lemma pair. Covers all tenses,
   moods, cases, numbers. Multi-form cells (e.g. `Πηλείδᾱο / Πηλείδεω`)
   are split into separate pairs.
2. **`form_of` references**. When a page says "form of X", that gives
   us an additional pair. Adds ~44K MG and ~6K AG pairs not found in
   inflection tables.
3. **`alt_of` references**. Alternative/variant spellings. Adds ~1K
   pairs.

### Confidence tiers

Not all lookup entries are equally trustworthy. Forms from inflection
tables are template-generated and may be wrong for irregular words.
Each entry is scored on a 5-point scale:

| Tier | Condition | MG count | AG count |
|:----:|-----------|:--------:|:--------:|
| 5 | Both EN + EL Wiktionary have a page for this form | 63K | 14K |
| 4 | EN Wiktionary has a page (no EL page) | 22K | 50K |
| 3 | EL Wiktionary has a page (no EN page) | 1.05M | 131K |
| 2 | Both EN + EL tables agree on the lemma | 199K | 49K |
| 1 | Single source, table-only | 1.49M | 2.12M |

Higher confidence wins when two sources map the same form to different
lemmas.

### Dialect tagging

Ancient Greek forms from EN Wiktionary carry dialect tags extracted from
inflection table headers (e.g. "Epic declension-1", "Attic contracted
present"). These are propagated to every form in that table section:

| Dialect | Tagged forms |
|---------|:------------:|
| Attic | 245K |
| Epic | 92K |
| Ionic | 14K |
| Doric | 9K |
| Koine | 9K |
| Aeolic | 3K |
| Laconian | 672 |
| Boeotian | 555 |
| Arcadocypriot | 407 |

### Quality controls

- **Greek-only filter**. All forms must contain only Greek Unicode
  characters (U+0370-03FF, U+1F00-1FFF, U+0300-036F). Removes Latin
  letters, digits, template artifacts.
- **Chain-breaking**. If form A maps to lemma B, and B maps to C, the
  chain is followed to the real headword. Fixes ~300K entries caused by
  accent-stripped key collisions.
- **Pronoun cross-contamination**. Greek Wiktionary dumps the entire
  pronoun paradigm table into each pronoun entry (e.g. `εσύ` lists
  `εγώ` as a "form"). Articles and determiners are restricted to
  headword-only. Pronoun forms that are headwords of other closed-class
  entries are skipped.
- **Proper noun plural filter**. EL Wiktionary generates plural forms
  for proper nouns via templates (413K junk entries like `Αχιλλείς`).
  These are skipped unless EN Wiktionary also lists them (which
  indicates a human editor intentionally added them, e.g. `Έλληνες`).
- **Training pair validation**. Every training pair's lemma must be a
  headword (maps to itself in the lookup). Pairs with non-headword
  lemmas are resolved to the real headword or dropped.

## Comparison

| Tool | Coverage | Training data | Katharevousa | Updates |
|------|----------|--------------|--------------|---------|
| *spaCy* `el_core_news_sm` | MG only | ~30K tokens (news) | no | static |
| *stanza* `el` | MG only | ~30K tokens (GDT treebank) | fails on augmented forms | static |
| Perseus *Morpheus* | AG only | hand-crafted rules | no | not actively developed |
| **Dilemma** | **MG + AG + Medieval + dialects** | **3.2M pairs (Wiktionary)** | **yes (AG+MG combined)** | **monthly from Wiktionary** |

Dilemma trains on **100x more data** than *stanza* or *spaCy*. *Morpheus*
is more accurate on classical AG (decades of hand-tuned rules), but only
covers one period. Dilemma covers all periods in one model, the only tool
that handles Katharevousa, which mixes AG morphology with MG vocabulary.

### gr-nlp-toolkit

[gr-nlp-toolkit](https://github.com/nlpaueb/gr-nlp-toolkit) provides
POS tagging, NER, morphological tagging, and dependency parsing for
Modern Greek, but does not do lemmatization. It uses *ByT5* for its
Greeklish transliteration module. *ByT5* processes raw UTF-8 bytes
rather than linguistic units, so a 10-character Greek word becomes
~20 encoder steps. This makes it unnecessarily slow for tasks like
lemmatization where the input/output vocabulary is small and fixed.
Dilemma's character-level transformer uses a Greek-specific vocabulary
(~160 tokens), giving it the same sequence length as the actual word
while being 75x smaller and 10-20x faster to train.

### Related work

[Vatri & McGillivray (2020)](https://brill.com/view/journals/jgl/20/2/article-p179_4.xml)
assessed the state of the art in Ancient Greek lemmatization via a
blinded evaluation by expert readers. They found that methods using
large lexica combined with POS tagging (CLTK backoff lemmatizer,
Diorisis corpus) consistently outperformed pure ML approaches with
smaller lexica. Dilemma follows the same principle: a large lookup
table (5.2M forms from Wiktionary) handles the vast majority of words,
with a small model as fallback.

[Celano (2025)](https://aclanthology.org/2025.lm4dh-1.5/) presented
state-of-the-art morphosyntactic parsing and lemmatization for Ancient
Greek using GreTa and PhilTa models trained on the AGDT and OGA
corpora. Best lemmatization F1 was 95.6% on classical text. These
models require POS context; Dilemma operates on isolated words but
benefits from a much larger form inventory.

[Swaelens et al. (2024)](https://aclanthology.org/2024.lrec-main.899/)
tested lemmatization on unedited Byzantine Greek epigrams and found
that classical accuracy (~95%) dropped 30+ points on Byzantine text
due to itacism, crasis, and non-standard orthography. Their best hybrid
method (transformer embeddings + dictionary lookup) reached 72%.
Dilemma's architecture follows a similar hybrid strategy, with
Byzantine/Medieval coverage from the EL Wiktionary Medieval Greek dump
and AG inflection tables that include Koine and Byzantine-era paradigms.

### Evaluation on DiGreC

Evaluated on the [DiGreC treebank](https://github.com/mdm33/digrec)
(119K manually reviewed tokens spanning Homer through 15th century
Byzantine Greek) with `resolve_articles=True`:

| Mode | Accuracy (stripped) |
|------|:-------------------:|
| Form-only | 70.6% |
| + resolve_articles | 81.4% |
| **+ context heuristics** | **83.8%** |

Context heuristics (`lemmatize_verbose` + disambiguation): prefer
non-proper lemmas for lowercase mid-sentence words, prefer `lang=grc`
candidates for AG text.

For comparison, Swaelens et al.'s best approach achieved 66% on
Byzantine epigrams (and 53-56% for pure transformer methods). Dilemma
reaches 83.8% on DiGreC without any task-specific training, using only
Wiktionary inflection tables.

Note: Dilemma operates on isolated words without POS context.
Context-aware models like Celano's GreTa (95.6% F1) achieve higher
accuracy on classical text by using full-sentence information.

### Known Issues

These are inherent limitations or Wiktionary coverage gaps, not code
bugs. Most can be fixed by editing the relevant Wiktionary entry, which
will propogate into Dilemma via Kaikki dumps.

| Issue | Tokens | Notes |
|-------|--------|-------|
| **Grave accent variants** | ~700 | ὣς does not resolve to ὡς. Grave is a positional variant of acute in Greek but Wiktionary only lists acute forms. Needs a normalization step. |
| **αὐτοῦ ambiguity** | ~200 | Genuine lexical ambiguity: both an adverb ("here/there") and genitive of αὐτός. Needs sentence context. |
| **ταῦτα self-map** | ~100 | Also an adverb headword in Wiktionary, so self-maps instead of mapping to οὗτος. |
| **μιν → οὗ** | ~340 | Wiktionary-correct (μιν is accusative of the 3rd person pronoun). Perseus treebank uses μιν as its own lemma — a convention difference. |
| **Lemma convention differences** | ~400 | αὐτάρ vs ἀτάρ, κε vs ἄν — Wiktionary and the Perseus treebank use different citation forms for some Homeric particles. |

## Credits

- Training data from [English Wiktionary](https://en.wiktionary.org/) and [Greek Wiktionary](https://el.wiktionary.org/) via [kaikki.org](https://kaikki.org/) JSONL dumps
- Flag icons by [svg-flags](https://github.com/ciscoriordan/svg-flags)

## How to Cite

```
Francisco Riordan, "Dilemma: Greek Lemmatizer" (2026).
https://github.com/ciscoriordan/dilemma
```

## License

MIT
