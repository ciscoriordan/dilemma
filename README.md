# Dilemma <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/languages/el.svg" width="28" alt="Greek"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/countries/cy.svg" width="28" alt="Cyprus"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/byzantine.svg" width="28" alt="Byzantine">

**Greek lemmatizer** with a 5.2 million form lookup table and a ~4M
parameter character-level transformer trained on 3.2 million Wiktionary
inflection pairs spanning Modern Greek, Ancient Greek, and Medieval Greek.

Most Greek words resolve instantly via the lookup table. For unseen forms,
Dilemma uses a small encoder-decoder transformer that learns morphological
patterns at the character level,the standard architecture from
[SIGMORPHON](https://sigmorphon.github.io/) shared tasks. At 4M parameters
it trains from scratch in minutes and runs inference in under a millisecond,
compared to fine-tuning approaches like *ByT5-small* (300M params) which take
hours to train and ~10ms per word. Greek lemmatization is highly
pattern-based,a small specialized model matches a large general-purpose
one, and the 5.2M lookup table handles the rest.

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

# Single period
d_mg = Dilemma(lang="el")                     # MG only
d_grc = Dilemma(lang="grc")                   # AG only

# Specific model scale
d = Dilemma(scale=1)                          # use scale 1 model
```

## How It Works

| Layer | Speed | Coverage | Source |
|-------|-------|----------|--------|
| **Lookup table** | instant | 5.2M known forms | Wiktionary inflection paradigms |
| **Transformer** | <1ms/word | generalizes to unseen forms | trained on lookup pairs |

The lookup table is built from all 5 Wiktionary [kaikki dumps](https://kaikki.org/)
(EN and EL editions for MG and AG, plus EL Medieval Greek). Each form is
indexed under its original, monotonic, and accent-stripped variants, so
`θεοὶ` (polytonic with grave), `θεοί` (monotonic with acute), and `θεοι`
(stripped) all resolve to `θεός`. Input can be polytonic, monotonic, or
unaccented. MG forms take priority, then Medieval, then AG.

The transformer is a small (~4M param) character-level encoder-decoder,
the standard architecture from
[SIGMORPHON](https://sigmorphon.github.io/) morphological inflection
shared tasks. It learns character-level patterns and generalizes to forms
not in Wiktionary. Training on MG + AG + Medieval data means the model
sees AG augment patterns (`ἔλυσε` → `λύω`) alongside MG stem
transformations (`σκότωσε` → `σκοτώνω`). For katharevousa forms like
`εσκότωσε`, it has both signals to draw from.

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
python build_data.py                        # auto-detects ~/Documents/Klisy/
python build_data.py --klisy /path/to/dumps
```
</details>

### 2. Train

Trains the character-level transformer on the extracted pairs. Use
`--scale` to match your GPU and time budget.

```bash
python train.py --scale 0                   # quick test (15 sec)
python train.py --scale 1                   # default (6.5 min on 2080 Ti)
python train.py --scale 2                   # recommended (13 min on 2080 Ti)
```

### Training scales

Every scale includes **100% of non-standard varieties** (Medieval,
Katharevousa, Cypriot, Cretan, Maniot, Heptanesian, archaic, dialectal).
The remaining budget is split 50/50 between Ancient Greek and standard MG.

| Scale | Training pairs | Varieties | AG | SMG | Time (2080 Ti) | Eval accuracy |
|:-----:|---------------:|----------:|-------:|-------:|:--------------:|:-------------:|
| 0 | 20K | 9K (100%) | 5.5K | 5.5K | 15 sec | 1.6% |
| 1 | 500K | 9K (100%) | 246K | 246K | 6.5 min | 46% |
| 2 | 1M | 9K (100%) | 496K | 496K | 13 min | 63% |
| 3 | 2M | 9K (100%) | 996K | 996K | ~26 min | - |
| 4 | 3.2M (all) | 9K (100%) | 1.4M (100%) | 1.3M (100%) | ~45 min | - |

Eval accuracy is measured on held-out pairs where the model must predict
the lemma without the lookup table. In practice, the lookup resolves
99%+ of forms instantly. The model only handles truly novel words.

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

### GPU quick start

```bash
git clone https://github.com/ciscoriordan/dilemma.git && cd dilemma
pip install -r requirements.txt
python build_data.py --download
python train.py --scale 1
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
covers one period. Dilemma covers all periods in one model - the only tool
that handles Katharevousa, which mixes AG morphology with MG vocabulary.

## Credits

- Training data from [English Wiktionary](https://en.wiktionary.org/) and [Greek Wiktionary](https://el.wiktionary.org/) via [kaikki.org](https://kaikki.org/) JSONL dumps
- Flag icons by [svg-flags](https://github.com/ciscoriordan/svg-flags)

## License

MIT
