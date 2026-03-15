# Dilemma <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/languages/el.svg" width="28" alt="Greek"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/countries/cy.svg" width="28" alt="Cyprus"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/byzantine.svg" width="28" alt="Byzantine">

**Greek lemmatizer** with a 6.5 million form lookup table and a small
character-level transformer trained on Wiktionary inflection pairs
spanning Modern Greek, Ancient Greek, and Medieval Greek.

Handles Standard Modern Greek, Katharevousa, and regional varieties.
Existing lemmatizers (*stanza*, *spaCy*) are trained on small modern news
corpora and fail on anything outside standard SMG. Dilemma trains a single
model on inflection data from all three periods of the language.

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
larger: **4.1M MG forms, 2.4M AG forms, 6.5M combined**.

Beyond the lookup, the transformer model generalizes to forms not in
Wiktionary. Katharevousa forms are the primary non-SMG target - they mix
AG morphology (augments, 3rd declension genitives) with MG vocabulary.
The strong Epic/Homeric coverage (3,755 entries) is directly relevant for
literary texts based on Homer.

> `εσκότωσε` → `σκοτώνω` · `πολεμούσαν` → `πολεμώ` · `δώση` → `δίνω`

---

## Quick Start

```python
from dilemma import Dilemma

d = Dilemma(lang="both")                      # combined MG+AG+Med
d.lemmatize("εσκότωσε")                       # "σκοτώνω" (AG augment + MG stem)
d.lemmatize("πάθης")                          # "παθαίνω"
d.lemmatize_batch(["δώση", "σκότωσε"])        # ["δίνω", "σκοτώνω"]

# Single period
d_mg = Dilemma(lang="el")                     # MG only
d_grc = Dilemma(lang="grc")                   # AG only

# Specific model scale
d = Dilemma(lang="both", scale=1)             # use scale 1 model
```

## How It Works

| Layer | Speed | Coverage | Source |
|-------|-------|----------|--------|
| **Lookup table** | instant | 6.5M known forms | Wiktionary inflection paradigms |
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
python train.py --scale 0                   # quick test (~30s)
python train.py --scale 1                   # default (~10 min on RTX 2080)
python train.py --scale 4                   # full data (~1h on 4090)
```

### Training scales

Every scale includes **100% of non-standard varieties** (Medieval,
Katharevousa, Cypriot, Cretan, Maniot, Heptanesian, archaic, dialectal).
The remaining budget is split 50/50 between Ancient Greek and standard MG.

| Scale | Training pairs | Varieties | AG | SMG | Time (2080 Ti) |
|:-----:|---------------:|----------:|-------:|-------:|:--------------:|
| 0 | 20K | 7.7K (100%) | 6.1K | 6.1K | ~30 sec |
| 1 | 500K | 7.7K (100%) | 246K | 246K | ~10 min |
| 2 | 1M | 7.7K (100%) | 496K | 496K | ~20 min |
| 3 | 2M | 7.7K (100%) | 996K | 996K | ~40 min |
| 4 | 3.4M (all) | 7.7K (100%) | 1.48M (100%) | 1.93M (100%) | ~1 hr |

The lookup table is the **same for all scales** (6.5M forms). The model
only handles words not in the lookup, so even scale 0 works well for
most text. Higher scales improve generalization to truly novel forms.

Models are saved to `model/combined-s0/`, `model/combined-s1/`, etc.
`Dilemma()` auto-detects the best available scale, or you can specify one:

```python
d = Dilemma(lang="both", scale=0)     # use scale 0 model explicitly
d = Dilemma(lang="both")              # auto-detect highest available
```

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

No pretrained weights — the model is small enough to train from scratch
on 500K+ pairs in minutes. The character vocabulary covers all Greek
Unicode ranges (monotonic, polytonic, extended).

### Why not ByT5?

An earlier version of Dilemma fine-tuned Google's
[ByT5-small](https://huggingface.co/google/byt5-small) (300M params).
ByT5 processes raw UTF-8 bytes, so a 10-character Greek word becomes
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
is highly pattern-based — a small specialized model matches a large
general-purpose one.

## Data

| Source | Dump size | Training Pairs | Lookup Forms |
|--------|-----------|----------------|--------------|
| EN + EL Wiktionary (MG) | 1.2 GB | 1.9M | 3.5M |
| EN + EL Wiktionary (AG) | 339 MB | 1.5M | 2.4M |
| EL Wiktionary (Medieval) | 3.7 MB | 3.9K | 7.2K |
| **Combined** | **1.5 GB** | **3.4M** | **5.9M** |

Training pairs are filtered to Greek-only characters. Lookup table
includes original, monotonic, and accent-stripped variants of each form
(6.5M total keys mapping to 5.9M unique forms).

All data is extracted automatically from [kaikki.org](https://kaikki.org/)
Wiktionary JSONL dumps.

## Comparison

| Tool | Coverage | Training data | Katharevousa | Updates |
|------|----------|--------------|--------------|---------|
| *spaCy* `el_core_news_sm` | MG only | ~30K tokens (news) | no | static |
| *stanza* `el` | MG only | ~30K tokens (GDT treebank) | fails on augmented forms | static |
| Perseus *Morpheus* | AG only | hand-crafted rules | no | not actively developed |
| **Dilemma** | **MG + AG + Medieval + dialects** | **3.4M pairs (Wiktionary)** | **yes (AG+MG combined)** | **monthly from Wiktionary** |

Dilemma trains on **100x more data** than *stanza* or *spaCy*. *Morpheus*
is more accurate on classical AG (decades of hand-tuned rules), but only
covers one period. Dilemma covers all periods in one model - the only tool
that handles Katharevousa, which mixes AG morphology with MG vocabulary.

## Credits

Flag icon by [svg-flags](https://github.com/ciscoriordan/svg-flags).

## License

MIT
