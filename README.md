# Dilemma <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/ancient-greece.svg" width="28" alt="Ancient Greece"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/languages/el.svg" width="28" alt="Greek"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/countries/cy.svg" width="28" alt="Cyprus"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/byzantine.svg" width="28" alt="Byzantine">

<p align="center">
  <img width="500" alt="dilemma" src="dilemma.png">
</p>

Greek lemmatizer with a **12.3 million form** lookup table and a ~4M
parameter character-level transformer trained on 3.4 million Wiktionary
inflection pairs spanning Modern Greek, Ancient Greek, and Medieval Greek.

Most Greek words resolve instantly via the lookup table. For unseen forms,
Dilemma uses a small encoder-decoder transformer that learns morphological
patterns at the character level, the standard architecture from
[SIGMORPHON](https://sigmorphon.github.io/) shared tasks. At 4M parameters
it trains from scratch in minutes and runs inference in under a millisecond,
compared to fine-tuning approaches like *ByT5-small* (300M params) which take
hours to train and ~10ms per word. Greek lemmatization is highly
pattern-based - a small specialized model matches a large general-purpose
one, and the 12.3M lookup table handles the rest.

**SQLite backend:** The lookup table loads from a pre-built SQLite database
(instant startup, ~0.3s) instead of parsing 600MB of JSON (~11s). Falls
back to JSON if the database isn't present.

**ONNX support:** Dilemma can run without PyTorch. When ONNX model files
are present, inference uses ONNX Runtime (~50 MB) instead of PyTorch (~2 GB).
The lookup table (which handles 95%+ of words) needs neither.

Handles Standard Modern Greek, Katharevousa, Cypriot, Cretan, and other
regional varieties alongside Ancient and Medieval Greek. Existing
lemmatizers (*stanza*, *spaCy*) are trained on ~30K tokens of modern news
and fail on anything outside standard SMG. Dilemma trains on **100x more
data** from all three periods of the language.

### Lookup table sources

The lookup table combines forms from multiple sources:

| Source | Forms | Notes |
|--------|------:|-------|
| **Wiktionary** (EN + EL, all periods) | 5.2M | Baseline from kaikki.org dumps |
| **LSJ** (Liddell-Scott-Jones) | 4.2M | 32K nouns, 22K verbs, 14K adjectives expanded via Wiktionary Lua modules |
| **Sophocles Lexicon** (Byzantine/Patristic) | 1.0M | 13.5K nouns, 4.6K verbs, 1.5K adverbs from OCR'd TEI data |
| **[GLAUx](https://github.com/alekkeersmaekers/glaux)** (Keersmaekers, 2021) | 557K | 17M-token corpus, 8th c. BC - 4th c. AD, 98.8% lemma accuracy |
| **UD Treebanks** (Perseus, PROIEL, DiGreC) | 27K | Gold form-lemma pairs from annotated treebanks |
| Closed-class fixes | ~500 | Articles, pronouns, prepositions mapped to canonical lemmas |

The LSJ and Sophocles expansions use Wiktionary's own
[grc-decl](https://en.wiktionary.org/wiki/Module:grc-decl) and
[grc-conj](https://en.wiktionary.org/wiki/Module:grc-conj) Lua modules
(via [wikitextprocessor](https://github.com/tatuylonen/wikitextprocessor))
to generate inflection paradigms from headwords with grammatical metadata.

### Orthographic normalizer (Byzantine/papyrological texts)

For texts with non-standard spelling, Dilemma includes an optional
orthographic normalizer that generates candidate normalized forms before
lookup. This handles:

- **Itacism**: η/ει/οι/υ all pronounced [i] and interchanged by scribes
- **αι/ε merger**: αι pronounced [e] and confused with ε
- **ο/ω confusion**: loss of vowel length distinction
- **Missing iota subscripta**: ᾳ/ῃ/ῳ written as α/η/ω
- **Spirantization**: β/υ interchange, φ/π, θ/τ, χ/κ confusion
- **Geminate simplification**: λλ→λ, νν→ν, etc.

Period-specific profiles (hellenistic, late_antique, byzantine) weight
rules by historical probability.

```python
d = Dilemma(normalize=True, period="byzantine")
```

### Evaluation

On the [Swaelens et al. (2024)](https://aclanthology.org/2024.lrec-main.899/)
DBBE gold standard (10K tokens of unedited Byzantine Greek epigrams):

| Method | Accuracy |
|--------|:--------:|
| Swaelens et al. best (2024, hybrid) | 65.8% |
| Swaelens et al. best (2025, multi-task) | ~74-75% |
| **Dilemma** | **91.4%** |

The remaining 8.6% errors break down as 3.2% no lookup hit and 5.4%
wrong lemma or convention difference. Compound decomposition (splitting
Byzantine compounds at linking vowels, e.g. θεοφθόγγοις -> θεο+φθόγγος)
reduced the no-lookup rate from 4.4% to 3.0%. The dedicated eval scripts
(`eval_dbbe.py`, `eval_digrec.py`) provide per-POS breakdowns and error
categorization.

On the [DiGreC treebank](https://github.com/mdm33/digrec) (119K tokens,
Homer through 15th century Byzantine Greek):

| Mode | Accuracy |
|------|:--------:|
| Strict match | 90.3% |
| + monotonic normalization | 91.2% |
| + accent stripping | 91.6% |
| **+ lemma equivalences** | **94.0%** |

The equivalence-adjusted score accounts for legitimate convention
differences between annotation schemes (e.g. `εἶπον`/`λέγω`,
`ἐγώ`/`ἡμεῖς`, `πρότερος`/`πρῶτος`).

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
larger.

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

# Byzantine text with orthographic normalization
d_byz = Dilemma(normalize=True, period="byzantine")
d_byz.lemmatize("θεω")                        # "θεός" (restores iota subscriptum)

# LSJ lemma convention: remap output to LSJ dictionary headwords
d_lsj = Dilemma(convention="lsj")
d_lsj.lemmatize("αἰνῶς")                     # "αἰνός" (adverb -> adjective)
d_lsj.lemmatize("εἶπον")                      # "λέγω" (aorist -> present stem)

# Cunliffe convention: remap to Cunliffe Homeric Lexicon headwords
d_cun = Dilemma(convention="cunliffe")
d_cun.lemmatize("γίνεται")                    # "γίγνομαι" (Homeric form)
d_cun.lemmatize("θέλει")                      # "ἐθέλω" (Homeric form)
d_cun.lemmatize("νοῦν")                       # "νόος" (uncontracted Homeric form)
```

By default, articles and pronoun clitics self-map (e.g. `τῆς` returns
`τῆς`). This is better for alignment pipelines where you want
surface-form matching. Set `resolve_articles=True` to resolve them
to canonical lemmas (`ὁ`, `ἐγώ`, `σύ`), matching treebank conventions
(AGDT, DiGreC, PROIEL).

The `convention` parameter remaps output lemmas to match a specific
dictionary or treebank convention. Set `convention="lsj"` for LSJ
dictionary headwords (maps suppletive aorists to present stems, adverbs
to adjective entries, MG -o neuter forms to AG -on, etc.). Set
`convention="cunliffe"` for Cunliffe Homeric Lexicon headwords (prefers
uncontracted Homeric forms like νόος over νοῦς, γίγνομαι over γίνομαι,
ἐθέλω over θέλω, and non-Attic -σσ- over -ττ-). Default is
`None` (Wiktionary convention, no remapping). The mapping is built
automatically from `data/lemma_equivalences.json` cross-referenced
against the convention's headword list, with explicit overrides in
`data/convention_{name}.json`.

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
- `lemma` - the lemma string
- `lang` - `"el"` (SMG), `"grc"` (AG), `"med"` (Medieval)
- `proper` - `True` if lemma is a proper noun (capitalized headword)
- `source` - `"lookup"`, `"elision"`, `"crasis"`, `"compound"`, `"model"`, `"identity"`
- `via` - how it matched: `"exact"`, `"lower"`, `"elision:ε"`, `"θεο+φθόγγος"`, `"+case_alt"`, etc.
- `score` - `1.0` for lookup, `0.5` for model, `0.0` for identity fallback

### POS-aware disambiguation

When a POS tagger (e.g. [Opla](https://github.com/ciscoriordan/opla))
provides UPOS tags, `lemmatize_pos` uses a 85K-entry POS lookup table
to disambiguate:

```python
d = Dilemma()
d.lemmatize_pos("αὐτοῦ", "ADV")    # "αὐτοῦ" (adverb: here/there)
d.lemmatize_pos("αὐτοῦ", "PRON")   # "αὐτός" (pronoun: genitive)
d.lemmatize_pos("ἄκρα", "NOUN")    # "ἄκρον" (noun: summit)
d.lemmatize_pos("ἄκρα", "ADJ")     # "ἄκρος" (adjective: outermost)
```

The POS lookup is built from four sources in priority order: UD treebanks
(gold), GLAUx corpus (8.7K entries), MG Wiktionary, AG Wiktionary.

### Spelling correction

For unknown or misspelled words, `suggest_spelling` returns candidate
corrections from the lookup table ranked by edit distance:

```python
d = Dilemma()
d.suggest_spelling("θεός")       # [("θεός", 0), ...]  (exact match)
d.suggest_spelling("θεος")       # [("θεός", 0), ...]  (diacritic error = free)
d.suggest_spelling("θδός")       # [("θεός", 1), ...]  (letter-level ED1)
```

The approach works in two layers. First, diacritics are stripped from both
the input and the dictionary, collapsing the 9.7M-entry lookup into ~1-3M
unique base forms. ED0/ED1/ED2 matches are found on these stripped forms,
then expanded back to their original polytonic variants and ranked by true
Levenshtein distance. This means accent and breathing errors (wrong accent,
missing breathing mark) are corrected for free, while letter-level errors
(θ/δ, ρ/ν) use standard edit distance. The spell index is built lazily on
first call.

### Elision expansion

Ancient Greek texts frequently elide final vowels before a following
vowel, marking the elision with an apostrophe (U+0313 in polytonic
encoding, U+02B9/U+02BC/U+2019 in other encodings). Dilemma resolves
these by stripping the elision mark and trying each Greek vowel against
the lookup table:

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
table, avoiding false matches from MG monotonic forms. Common function
words (prepositions, particles, conjunctions like ἀλλά, μετά, παρά, κατά,
διά) are prioritized over content words when disambiguating, and proper
nouns are deprioritized. Remaining candidates are ranked by vowel frequency
in elision contexts (ε, α, ο most common).

## How It Works

| Layer | Speed | Coverage | Source |
|-------|-------|----------|--------|
| **Lookup table** | instant | 12.3M known forms | Wiktionary + LSJ + Sophocles + GLAUx + treebanks |
| **Normalizer** | instant | Byzantine orthographic variants | Rule-based candidate generation |
| **Elision expansion** | instant | AG elided forms | Vowel expansion against lookup |
| **Crasis table** | instant | ~50 common crasis forms | Hand-curated |
| **Compound decomposition** | instant | Byzantine compound words | Split at linking vowel, look up base |
| **Spelling correction** | lazy index | ED0-2 suggestions for unknown words | Accent-stripped edit distance |
| **Transformer** | <1ms/word | generalizes to unseen forms | Trained on Wiktionary pairs |

The lookup table is built from Wiktionary [kaikki dumps](https://kaikki.org/)
(EN and EL editions for MG and AG, plus EL Medieval Greek), expanded with
inflected forms from LSJ (via Wiktionary Lua modules) and the Sophocles
lexicon of Roman and Byzantine Greek, then augmented with form-lemma pairs
from gold-standard treebanks (Gorman, AGDT). Each form is indexed under
its original, monotonic, and accent-stripped variants, so `θεοὶ` (polytonic
with grave), `θεοί` (monotonic with acute), and `θεοι` (stripped) all
resolve to `θεός`. Input can be polytonic, monotonic, or unaccented. AG
forms take priority, then Medieval, then MG - this ensures classical lemma
forms (βιβλίον, φύσις, θεῖος) are preferred over their MG equivalents
(βιβλίο, φύση, θείο). For polytonic input (breathings/circumflex), an
additional AG-only lookup pass runs first.

When the transformer handles an unseen form, beam search generates
multiple candidates and picks the first that matches a known headword
from Wiktionary, [LSJ](https://github.com/helmadik/LSJLogeion) (116K
headwords), or [Cunliffe's Homeric Lexicon](https://archive.org/details/lexiconofhomeric0000cunn) (12K headwords). If nothing
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
python build_lookup_db.py              # builds SQLite DB for instant startup (optional)
```

The lookup table handles 95%+ of words with no model at all. The SQLite
step is optional but recommended - it reduces startup time from ~11s to
~0.3s. Without it, Dilemma falls back to loading JSON files. For the
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

### LSJ/Sophocles expansion (optional, requires wikitextprocessor)

To regenerate the expanded lookup table from LSJ and Sophocles sources:

```bash
pip install --force-reinstall --no-deps git+https://github.com/tatuylonen/wikitextprocessor.git
python expand_lsj.py --setup           # build Wiktionary Lua module database
python expand_lsj.py --expand          # expand LSJ nouns
python expand_lsj.py --expand-verbs    # expand LSJ verbs
python expand_sophocles.py --expand    # expand Sophocles nouns
python expand_sophocles.py --expand-verbs  # expand Sophocles verbs
```

This requires the LSJ XML files from [LSJLogeion](https://github.com/helmadik/LSJLogeion)
and the Sophocles TEI data (included in `data/sophocles/`).

---

## Training

### 1. Build data

Downloads all 5 kaikki dumps and extracts every form-lemma pair from
inflection tables. Non-Greek characters are filtered out.

```bash
pip install -r requirements.txt
python build_data.py --download             # downloads + extracts (~1.5GB total)
```

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
Perfect tense verb forms are oversampled 3x, following
[Swaelens et al. (2025)](https://aclanthology.org/2025.acl-long.430/)'s
finding that perfects are underrepresented in training data (2%) relative
to Byzantine text (11.4%).

| Scale | Training pairs | Varieties | AG | SMG | Time (2080 Ti) | Eval | Tests |
|:-----:|---------------:|----------:|-------:|-------:|:--------------:|:----:|:-----:|
| 1 | 20K | 9K (100%) | 5.5K | 5.5K | 16 sec | 2.6% | 53/55 |
| 2 | 1M | 9K (100%) | 496K | 496K | 13 min | 62% | 54/55 |
| 3 | 3.4M (all) | 9K (100%) | 1.5M (100%) | 1.7M (100%) | 95 min | 71.5% | 55/55 |

Eval accuracy is the model's score on held-out pairs *without* the
lookup table. In practice, the lookup resolves most forms instantly
and the model only handles truly novel words. When the model is used,
beam search generates 4 candidates and the first one that matches a
known headword in the lookup wins. If none match, the input is returned
unchanged (safe fallback).

### Multi-task learning

When training pairs include POS tags (from Wiktionary) and morphological
features (from GLAUx), the model jointly predicts POS, nominal morphology
(gender/number/case, 45 labels), and verbal morphology (tense/mood/voice,
69 labels) alongside the lemma via auxiliary classification heads on the
encoder output. This follows
[Swaelens et al. (2025)](https://aclanthology.org/2025.acl-long.430/)'s
finding that multi-task learning (joint POS + morphology + lemma)
improved Byzantine Greek lemmatization by ~9 percentage points. Each
auxiliary loss is weighted at 0.1x relative to the lemmatization loss.
At scale 3, the heads reach 90.4% POS, 81.5% nominal, and 91.2% verbal
accuracy on the held-out set.

Training uses a linear warmup LR scheduler (500 steps warmup, then linear
decay) and gradient clipping (max norm 1.0) for stable convergence.

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
python build_lookup_db.py              # SQLite for instant startup
python train.py --scale 2
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
| POS head | Linear (256 -> 10 tags), auxiliary task |
| Nominal head | Linear (256 -> 45 labels), gender/number/case |
| Verbal head | Linear (256 -> 69 labels), tense/mood/voice |
| FFN | 512 dim |
| Vocabulary | ~381 Greek characters + special tokens |
| Parameters | ~4.2M |
| Inference | <1ms/word (GPU), ~2ms/word (CPU) |

No pretrained weights - the model is small enough to train from scratch
on 500K+ pairs in minutes. The character vocabulary covers all Greek
Unicode ranges (monotonic, polytonic, extended). Three auxiliary
classification heads (POS, nominal morphology, verbal morphology) share
the encoder and improve representations via multi-task learning.

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
| Training (3.4M pairs, 3 epochs) | ~20 hours | ~95 min |
| Inference | ~10ms/word | <1ms/word |
| Dependencies | torch + transformers | torch only |

The custom model trains **10-20x faster** and runs **10x faster** at
inference, with no loss in accuracy for this task. Greek lemmatization
is highly pattern-based - a small specialized model matches a large
general-purpose one.

## Data

| Source | Forms | Notes |
|--------|------:|-------|
| EN + EL Wiktionary (MG) | 2.8M | From kaikki.org dumps |
| EN + EL Wiktionary (AG) | 2.4M | From kaikki.org dumps |
| EL Wiktionary (Medieval) | 6.9K | From kaikki.org dumps |
| LSJ noun/verb/adj expansion | 4.2M | Via Wiktionary Lua modules |
| Sophocles lexicon expansion | 1.0M | Byzantine/Patristic vocabulary |
| UD Treebanks (AG) | 27K | Gold annotations from Perseus, PROIEL, DiGreC |
| GLAUx corpus | 557K | 17M tokens, 98.8% accuracy ([Keersmaekers 2021](https://github.com/alekkeersmaekers/glaux)) |
| **Total lookup** | **12.3M** | |

All Wiktionary data is extracted automatically from
[kaikki.org](https://kaikki.org/) JSONL dumps. LSJ and Sophocles
expansions use wikitextprocessor to run Wiktionary's grc-decl and grc-conj
Lua modules on headwords extracted from lexicon XML/TEI files.

The [GLAUx corpus](https://github.com/alekkeersmaekers/glaux) provides
the largest single source of new form-lemma pairs outside Wiktionary.
We chose GLAUx over two larger corpora:

- [Opera Graeca Adnotata](https://doi.org/10.5281/zenodo.14206061)
  (OGA, 40M tokens): lower lemma accuracy (91.4% vs GLAUx's 98.8%),
  standoff PAULA XML format requires complex alignment code, and at
  91.4% accuracy would introduce ~3.4M wrong lemmas into the lookup
  table - more noise than signal for a lookup-first system.
- [Pedalion](https://github.com/perseids-publications/pedalion-trees)
  (5.8M tokens): smaller than GLAUx with similar classical-period
  coverage. Would add few forms not already covered by GLAUx + Wiktionary
  + LSJ, since the remaining lookup gaps are mostly Byzantine compounds
  not found in any classical corpus.

All three are CC BY-SA 4.0. Compound decomposition (added in v1.5)
reduced the no-lookup-hit rate on DBBE from 4.4% to 3.0% by splitting
compound words at linking vowels (ο/ι/υ) and looking up the base
element. The remaining 3.0% are forms where neither lookup, compound
decomposition, nor the seq2seq model can recover the correct lemma.

Each form is indexed under its original, monotonic, and accent-stripped
variants for fuzzy matching.

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
  chain is resolved to the real headword at build time. Fixes ~65K entries
  caused by accent-stripped key collisions and treebank convention differences.
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
| **Dilemma** | **MG + AG + Medieval + dialects** | **3.4M pairs + 12.3M lookup** | **yes (AG+MG combined)** | **monthly from Wiktionary** |

Dilemma trains on **100x more data** than *stanza* or *spaCy*. *Morpheus*
is more accurate on classical AG (decades of hand-tuned rules), but only
covers one period. Dilemma covers all periods in one model - the only tool
that handles Katharevousa, which mixes AG morphology with MG vocabulary.

### Related work

[Vatri & McGillivray (2020)](https://brill.com/view/journals/jgl/20/2/article-p179_4.xml)
assessed the state of the art in Ancient Greek lemmatization via a
blinded evaluation by expert readers. They found that methods using
large lexica combined with POS tagging (CLTK backoff lemmatizer,
Diorisis corpus) consistently outperformed pure ML approaches with
smaller lexica. Dilemma follows the same principle: a large lookup
table (9.7M forms) handles the vast majority of words, with a small
model as fallback.

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
method (transformer embeddings + dictionary lookup) reached 65.8%.
Dilemma achieves 91.4% on the same dataset.

[Swaelens et al. (2025)](https://aclanthology.org/2025.acl-long.430/)
showed that multi-task learning (joint POS + morphology + lemma
prediction) improved Byzantine lemmatization by ~9pp, reaching ~74-75%.
They also demonstrated that subword-tokenizing transformers plateau on
Byzantine Greek due to orthographic inconsistency, and called for
character-level models as the next step. Dilemma's character-level
encoder-decoder is this architecture, and its perfect tense oversampling
and multi-task POS head are directly informed by their findings.

### Known Issues

These are inherent limitations or Wiktionary coverage gaps, not code
bugs. Most can be fixed by editing the relevant Wiktionary entry, which
will propagate into Dilemma via kaikki dumps.

| Issue | Tokens | Notes |
|-------|--------|-------|
| **αὐτοῦ ambiguity** | ~200 | Genuine lexical ambiguity: both an adverb ("here/there") and genitive of αὐτός. Resolved when POS context is available via `lemmatize_pos()`. |
| **μιν → ὅς** | ~340 | Convention difference. Wiktionary maps μιν to the 3rd person pronoun. Perseus treebank uses μιν as its own lemma. |
| **Lemma convention differences** | ~400 | αὐτάρ vs ἀτάρ, κε vs ἄν - Wiktionary and Perseus use different citation forms for some Homeric particles. Handled by lemma equivalence groups for evaluation. |

## Credits

- Training data from [English Wiktionary](https://en.wiktionary.org/) and [Greek Wiktionary](https://el.wiktionary.org/) via [kaikki.org](https://kaikki.org/) JSONL dumps
- LSJ data from [LSJLogeion](https://github.com/helmadik/LSJLogeion) (Helma Dik)
- Sophocles lexicon TEI from [Ionian University / Internet Archive](https://archive.org/details/pateres)
- [GLAUx](https://github.com/alekkeersmaekers/glaux) corpus data (Keersmaekers, 2021) (CC BY-SA 4.0)
- DBBE evaluation data from [Swaelens et al.](https://github.com/coswaele/ByzantineGreekDatasets) (CC BY 4.0)
- Flag icons by [svg-flags](https://github.com/ciscoriordan/svg-flags)

## How to Cite

```
Francisco Riordan, Dilemma [computer software] (2026).
https://github.com/ciscoriordan/dilemma
```

## License

MIT
