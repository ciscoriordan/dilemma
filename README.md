# Dilemma <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/ancient-greece.svg" width="28" alt="Ancient Greece"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/languages/el.svg" width="28" alt="Greek"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/countries/cy.svg" width="28" alt="Cyprus"> <img src="https://raw.githubusercontent.com/ciscoriordan/svg-flags/main/circle/historical/byzantine.svg" width="28" alt="Byzantine"> [![Tests](https://github.com/ciscoriordan/dilemma/actions/workflows/test.yml/badge.svg)](https://github.com/ciscoriordan/dilemma/actions/workflows/test.yml)

<p align="center">
  <img width="500" alt="dilemma" src="dilemma.png">
</p>

Dilemma is a holistic Greek lemmatizer spanning Ancient Greek (Classical,
Homeric, Hellenistic), Medieval/Byzantine Greek (both vernacular and
literary), and Modern Greek (Demotic and Katharevousa). It combines multiple strategies into a unified pipeline:

- A 12.5M-form lookup table built from Wiktionary inflection tables,
  Wiktionary's Lua morphological modules applied to LSJ and Sophocles
  lexicon headwords, gold-standard treebanks (Perseus, PROIEL, Gorman,
  DiGreC), and annotated corpora (GLAUx, Diorisis, HNC)
- Dialect normalization for Ionic, Doric, Aeolic, and Koine orthographic
  variants, mapping dialectal forms to their Attic equivalents for lookup
- Surgical rule-based morphological analysis including augment stripping,
  reduplication removal, particle suffix resolution, elision expansion,
  and crasis decomposition - these handle systematic transformations that
  the lookup table cannot enumerate exhaustively (every word x every
  enclitic particle) and that a transformer might not generalize to for
  rare forms it has never trained on
- A small supervised character-level transformer (~4M parameters) trained
  on 3.5M explicit form-lemma pairs, used only for the ~5% of words not
  resolved by lookup or rules
- Convention remapping to match output lemmas to target dictionaries (LSJ,
  Cunliffe, Triantafyllidis, Wiktionary)

Most Greek words resolve instantly via the lookup table. For unseen forms,
Dilemma falls back through rule-based morphological analysis and dialect
normalization before reaching the transformer, which learns morphological
patterns at the character level, the standard architecture from
[SIGMORPHON](https://sigmorphon.github.io/) shared tasks. At 4M parameters
it trains from scratch in minutes, compared to fine-tuning approaches
like *ByT5-small* (300M params) which take hours to train. Greek
lemmatization is highly pattern-based - a small specialized model matches
a large general-purpose one, and the 12.5M lookup table handles the rest.

### What's new here

Most individual components of Dilemma are established techniques. What's
novel is the combination and the scale:

- **Multi-period Greek in one tool.** No other lemmatizer covers Ancient,
  Byzantine, and Modern Greek in a single system. Tools like Morpheus
  handle only classical AG. Stanza handles AG or MG but not both.
  Dilemma resolves Katharevousa, vernacular medieval, and regional MG
  varieties (Cypriot, Cretan) alongside Homer and Herodotus.
- **12.5M-form lookup table.** The largest compiled for Greek, built by
  applying Wiktionary's Lua inflection modules to LSJ and Sophocles
  headwords, then merging with five gold-standard treebanks and two
  corpus-derived pair sets. This is a data engineering contribution, not
  an algorithmic one.
- **Dialect normalization.** Systematic Ionic, Doric, Aeolic, and Koine
  orthographic mapping to Attic equivalents. No other Greek lemmatizer
  handles dialectal variation this way.
- **Elision with consonant de-assimilation and frequency ranking.**
  Recovers prepositions like κατά from assimilated forms like καθ' by
  reversing the aspiration rule, then ranks candidates by corpus frequency.

### What's established

- The character-level transformer uses the standard SIGMORPHON architecture
  for morphological inflection/reinflection tasks, though applying it to
  Greek lemmatization (rather than inflection) appears to be new.
- The SQLite lookup with monotonic/stripped fallback keys is a
  straightforward hash table approach.
- Edit-distance spelling correction uses a BK-tree, a well-known data
  structure for metric-space nearest-neighbor search.
- Wiktionary as a data source for morphological data is widely used
  (kaikki.org, Lexonomy, etc.).
- Treebank integration (Perseus, PROIEL, Gorman) follows standard
  practice in computational linguistics.

**Note on methodology:** Dilemma is a supervised system. The transformer
trains on 3.5M explicit form-to-lemma pairs from Wiktionary inflection
tables, and the lookup table (which handles 95%+ of words) is literally a
dictionary of correct answers. This is not unsupervised learning (pattern
discovery from raw text with no labels). Some evaluations have incorrectly
categorized Dilemma alongside unsupervised tools.

**SQLite backend:** The lookup table loads from a pre-built SQLite database
(instant startup, ~0.3s) instead of parsing 600MB of JSON (~11s). Falls
back to JSON if the database isn't present.

**ONNX support:** For inference, ONNX Runtime (~50 MB) and PyTorch
(~2 GB) produce identical results. If you already have PyTorch
installed, it works fine. If you're starting fresh, ONNX is the
lighter option. PyTorch is only required for training. The lookup
table (which handles 95%+ of words) needs neither.

## Table of Contents

- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
  - [Conventions](#conventions)
- [Evaluation](#evaluation)
  - [Multi-period benchmarks](#multi-period-benchmarks)
  - [Rare vocabulary coverage](#rare-vocabulary-coverage)
  - [DiGreC treebank](#digrec-treebank)
  - [HNC Modern Greek](#hnc-modern-greek)
- [How It Works](#how-it-works)
  - [Pipeline overview](#pipeline-overview)
  - [Lookup table](#lookup-table)
  - [Rule-based fallback layer](#rule-based-fallback-layer)
  - [Dialect normalization](#dialect-normalization)
  - [Orthographic normalizer](#orthographic-normalizer)
  - [Transformer model](#transformer-model)
- [API Reference](#api-reference)
  - [Language and convention options](#language-and-convention-options)
  - [Verbose mode](#verbose-mode)
  - [Batch processing](#batch-processing)
  - [POS-aware disambiguation](#pos-aware-disambiguation)
  - [Spelling correction](#spelling-correction)
  - [Elision expansion](#elision-expansion)
- [Greek Coverage](#greek-coverage)
  - [Language codes](#language-codes)
  - [Modern Greek varieties](#modern-greek-varieties)
  - [Ancient Greek varieties](#ancient-greek-varieties)
  - [Medieval/Byzantine Greek](#medievalbyzantine-greek)
- [Development](#development)
  - [Full installation](#full-installation)
  - [Training](#training)
  - [LSJ/Sophocles expansion](#lsjsophocles-expansion)
  - [Export to ONNX](#export-to-onnx)
  - [Testing](#testing)
- [Data](#data)
  - [Sources and scale](#sources-and-scale)
  - [Confidence tiers](#confidence-tiers)
  - [Quality controls](#quality-controls)
- [Architecture](#architecture)
- [Credits](#credits)
- [How to Cite](#how-to-cite)
- [License](#license)

<p align="center">
  <img src="https://raw.githubusercontent.com/ciscoriordan/dilemma/main/diagram.svg" width="700" alt="Dilemma architecture">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ciscoriordan/dilemma/main/examples.svg" width="700" alt="Lemmatization examples">
</p>

---

## Quick Start

### Installation

```bash
git clone https://github.com/ciscoriordan/dilemma.git && cd dilemma
pip install onnxruntime huggingface_hub  # lightweight dependencies
huggingface-cli download ciscoriordan/dilemma --local-dir . --include "data/*" "model/*"
```

This downloads the pre-built lookup tables and ONNX model files from
HuggingFace. The lookup table handles 95%+ of words with no model at all.
For the remaining ~5% (unseen forms), the ONNX model files provide
transformer inference. If you already have PyTorch installed, that works
too - both produce identical output.

To build the data from scratch instead of downloading:
```bash
python build_data.py --download        # downloads Wiktionary dumps, builds lookup tables
python build_lookup_db.py              # builds SQLite DB for instant startup (optional)
python fix_selfmaps.py                 # fixes inflected forms that self-map (optional)
```

### Basic Usage

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
d_mg = Dilemma(lang="el")                     # MG only (falls back to combined model if no el-specific model exists)
d_grc = Dilemma(lang="grc")                   # AG only

# Specific model scale
d = Dilemma(scale="test")                     # use test-scale model

# Treebank evaluation mode: resolve articles to ὁ, pronouns to ἐγώ/σύ
d_eval = Dilemma(resolve_articles=True)
d_eval.lemmatize("τῆς")                       # "ὁ" (not "τῆς")
d_eval.lemmatize("μοι")                       # "ἐγώ" (not "μοι")

# Byzantine text with orthographic normalization
d_byz = Dilemma(normalize=True, period="byzantine")
d_byz.lemmatize("θεω")                        # "θεός" (restores iota subscriptum)
```

By default, articles and pronoun clitics self-map (e.g. `τῆς` returns
`τῆς`). This is better for alignment pipelines where you want
surface-form matching. Set `resolve_articles=True` to resolve them
to canonical lemmas (`ὁ`, `ἐγώ`, `σύ`), matching treebank conventions
(AGDT, DiGreC, PROIEL). The `triantafyllidis` convention auto-enables
article resolution (articles to `ο`, skipping AG pronoun resolution
for forms like `σε`/`με` that are MG prepositions).

### Conventions

Different dictionaries and treebanks use different citation forms for
the same word. The `convention` parameter remaps Dilemma's output to
match a specific standard. This matters for benchmarking: a tool that
outputs `εἰμί` and a gold standard that expects `είμαι` will show
as an error even though both are correct for their respective
conventions.

| Convention | Target | Example mappings |
|------------|--------|-----------------|
| `None` (default) | Wiktionary headwords | `εἶπον`→`εἶπον`, `θεούς`→`θεός`, `σπήλαια`→`σπήλαιον` |
| `lsj` | [LSJ](https://github.com/ciscoriordan/lsj9) dictionary | `εἶπον`→`λέγω`, `αἰνῶς`→`αἰνός`, `σπήλαιο`→`σπήλαιον` |
| `cunliffe` | [Cunliffe](https://archive.org/details/lexiconofhomeric0000cunn) Homeric Lexicon | `γίνεται`→`γίγνομαι`, `θέλει`→`ἐθέλω`, `νοῦν`→`νόος` |
| `triantafyllidis` | [Triantafyllidis](http://www.greek-language.gr/greekLang/modern_greek/tools/lexica/triantafyllides/) MG dictionary | `ὁ`→`ο`, `εἰμί`→`είμαι`, `σπήλαιον`→`σπήλαιο`, `εἷς`→`ένας` |

The mapping is built automatically from `data/lemma_equivalences.json`
cross-referenced against the convention's headword list, with explicit
overrides in `data/convention_{name}.json`. Lemma equivalences also
group valid alternative lemmatizations (comparative/positive adjective
forms, active/deponent pairs, spelling variants) so that benchmarks
score them as correct rather than penalizing convention disagreements.

For individual form-to-lemma corrections where the lookup table returns
the wrong lemma due to ambiguity (e.g., proper nouns beating common
verbs), `build_lookup_db.py` has a `_LOOKUP_OVERRIDES` dict that
hard-corrects specific entries in the database.

Other tools (stanza, spaCy, CLTK) have fixed output conventions
matching their training treebanks and cannot be remapped.

```python
# LSJ lemma convention: remap output to LSJ dictionary headwords
d_lsj = Dilemma(convention="lsj")
d_lsj.lemmatize("αἰνῶς")                     # "αἰνός" (adverb -> adjective)
d_lsj.lemmatize("εἶπον")                      # "λέγω" (aorist -> present stem)

# Cunliffe convention: remap to Cunliffe Homeric Lexicon headwords
d_cun = Dilemma(convention="cunliffe")
d_cun.lemmatize("γίνεται")                    # "γίγνομαι" (Homeric form)
d_cun.lemmatize("θέλει")                      # "ἐθέλω" (Homeric form)
d_cun.lemmatize("νοῦν")                       # "νόος" (uncontracted Homeric form)

# Triantafyllidis convention: remap to Modern Greek monotonic forms
d_mg = Dilemma(convention="triantafyllidis")
d_mg.lemmatize("σπήλαια")                     # "σπήλαιο" (not σπήλαιον)
d_mg.lemmatize("Είναι")                       # "είμαι" (not εἰμί)
d_mg.lemmatize("εργαλεία")                    # "εργαλείο" (not ἐργαλεῖον)
d_mg.lemmatize("τα")                          # "ο" (not ὁ)
```

In the benchmark table, the first two Dilemma rows use the Wiktionary
convention. The `convention="triantafyllidis"` row auto-enables article
resolution (articles to `ο`, demonstratives to `αυτός`) and outputs
monotonic MG lemma forms. This is the recommended setting for Modern
Greek text.

## Evaluation

### Multi-period benchmarks

Equiv-adjusted accuracy across four periods of Greek. All tools
evaluated with the same normalization (case-folded, accent-stripped)
and lemma equivalence groups (see `data/benchmarks/bench_all.py`).

**Test sets:**
- **AG Classical**: Sextus Empiricus, *Pyrrhoniae Hypotyposes* 1.1-1.8 (357 tokens, [First1KGreek](https://opengreekandlatin.github.io/First1KGreek/), CC BY-SA). Not in any UD treebank or Gorman.
- **Byzantine**: [Swaelens et al. (2024)](https://aclanthology.org/2024.lrec-main.899/) DBBE gold standard (8,342 tokens of unedited Byzantine epigrams, CC BY 4.0). Not in any tool's training data.
- **Katharevousa**: Konstantinos Sathas, *Neoelliniki Filologia* (1868), biography of Bessarion (318 tokens, [el.wikisource.org](https://el.wikisource.org/), public domain). No Katharevousa treebank exists.
- **Demotic MG**: Greek Wikipedia articles "[Σπήλαιο Πετραλώνων](https://el.wikipedia.org/wiki/Σπήλαιο_Πετραλώνων)" and "[Ελαιόλαδο](https://el.wikipedia.org/wiki/Ελαιόλαδο)" (400 tokens, CC BY-SA). Not in any MG treebank. A separate dev set (251 tokens from "[Μέλισσα](https://el.wikipedia.org/wiki/Μέλισσα)" and "[Σαμοθράκη](https://el.wikipedia.org/wiki/Σαμοθράκη)") is also available.

| Tool | AG Classical | Byzantine (literary) | Katharevousa | Demotic MG |
|------|:--------:|:--------:|:--------:|:--------:|
| [spaCy](https://spacy.io/) `el` | -- | 31.7% | 44.6% | 79.9% |
| [stanza](https://stanfordnlp.github.io/stanza/) `el` | -- | 37.4% | 48.4% | 87.0% |
| [Swaelens et al. (2024)](https://aclanthology.org/2024.lrec-main.899/) | -- | 65.8% | -- | -- |
| [CLTK](https://github.com/cltk/cltk) | 81.2% | 66.6% | 74.8% | -- |
| [Morpheus](https://github.com/perseids-tools/morpheus-perseids-api) (oracle) | -- | 71.1% | -- | -- |
| [stanza](https://stanfordnlp.github.io/stanza/) `grc` | 92.2% | 71.3% | 85.2% | -- |
| [Swaelens et al. (2025)](https://aclanthology.org/2025.acl-long.430/) | -- | ~74-75% | -- | -- |
| **Dilemma** (best convention per period) | **99.7%** | **92.7%** | **95.6%** | **96.0%**† |

<sub>†`lang="el"` with `triantafyllidis` scores 95.8%, nearly matching `lang="all"` (96.0%). For MG-only workloads, `lang="el"` with `triantafyllidis` is recommended since it avoids AG false matches.</sub>

Cells marked `--` indicate the tool doesn't support that period or
wasn't tested. Morpheus "oracle" picks the best candidate from all
its analyses, representing the ceiling for rule-based morphology.

**Dilemma detail by convention:**

| Lang | Convention | POS | AG Classical | Byzantine (literary) | Katharevousa | Demotic MG |
|------|------------|-----|:--------:|:--------:|:--------:|:--------:|
| `all` | `wiktionary` (default) | -- | 99.7% | 92.7% | 95.6% | 79.0%* |
| `all` | `wiktionary` (default) | gold | -- | 92.6% | -- | -- |
| `all` | `triantafyllidis` | -- | 85.4% | 83.4% | 90.9% | 96.0%† |
| `grc` | `wiktionary` (default) | -- | 99.7% | 92.3% | 94.3% | 79.0%* |
| `grc` | `triantafyllidis` | -- | 87.4% | 86.9% | 89.9% | 90.0% |
| `el` | `wiktionary` (default) | -- | 93.3% | 86.7% | 92.5% | 73.0%* |
| `el` | `triantafyllidis` | -- | 85.4% | 82.6% | 89.9% | 95.8% |

<sub>\*Demotic MG scores with `wiktionary` convention are convention mismatches, not real accuracy gaps: AG citation forms like `σπήλαιον` don't match the MG gold standard `σπήλαιο`. Using `convention="triantafyllidis"` fixes this.</sub>

`lang="all"` searches both AG and MG lookup tables for every token.
Other tools in the comparison table are locked to a single language.
The `wiktionary` convention outputs polytonic AG citation forms. The
`triantafyllidis` convention outputs monotonic MG lemma forms and is
the recommended setting for Modern Greek text (see
[Conventions](#conventions)).

POS column: `--` means Dilemma disambiguates on its own (default).
`gold` means gold-standard POS tags from the dataset are fed in. Only
DBBE provides gold POS; the negligible difference (92.7% vs 92.6%)
confirms POS ambiguity is not a significant error source.

The eval scripts (`eval/eval_dbbe.py`, `eval/eval_digrec.py`,
`eval/eval_hnc.py`, `eval/bench_dbbe.py`) provide per-POS breakdowns
and error categorization.

### Rare vocabulary coverage

Following [SIGMORPHON](https://sigmorphon.github.io/) shared task
methodology for out-of-vocabulary evaluation, we exclude the 3,000 most
frequent Greek forms and capitalized words, then check whether the output
lemma is a valid LSJ/Wiktionary headword. This tests the hard tail that
matters for real texts.

| Text | Period | Morpheus | Stanza | Dilemma |
|------|--------|:--------:|:------:|:-------:|
| Xenophon, *Cyropaedia* | Attic | 99.5% | 84% | **99.6%** |
| Kresadlo, *Astronautilia* 13 | Epic | 74% | 74% | **84%** |
| Herodotus, *Histories* | Ionic | 99.5% | 88% | **99.9%** |

<sub>On Cyropaedia, gold accuracy vs Gorman treebank annotations is
93.2%. The remaining gap is convention differences (e.g. κτάομαι vs
κτέομαι, ᾄδω vs ἀείδω), not missing forms. Herodotus gold-match
accuracy (vs PROIEL annotations) is 95.3%, where the gap is almost
entirely convention differences (Ionic vs Attic spelling, plural
ethnonym lemmas, voice conventions), not missing forms or
disambiguation failures.</sub>

### DiGreC treebank

On the [DiGreC treebank](https://github.com/mdm33/digrec) (119K tokens,
Homer through 15th century Byzantine Greek), Dilemma reaches 93.7%
equiv-adjusted (90.3% strict). The gap accounts for convention
differences between annotation schemes (e.g. `εἶπον`/`λέγω`,
`ἐγώ`/`ἡμεῖς`).

### HNC Modern Greek

`eval_hnc.py` evaluates against the
[HNC Golden Corpus](https://inventory.clarin.gr/corpus/870) (88K tokens
of gold-standard Modern Greek from CLARIN:EL).

## How It Works

### Pipeline overview

| Layer | Speed | Coverage | Source |
|-------|-------|----------|--------|
| **Lookup table** | hash lookup `O(1)` | 12.5M known forms | Wiktionary + LSJ + Sophocles + GLAUx + treebanks |
| **Normalizer** | k candidates `O(k)` | Byzantine orthographic variants | Rule-based candidate generation |
| **Elision expansion** | v=7 vowels `O(v)` | AG elided forms | Vowel expansion against lookup |
| **Crasis table** | hash lookup `O(1)` | ~50 common crasis forms | Hand-curated |
| **Particle suffix stripping** | suffix check `O(1)` | AG enclitic forms (-per, -ge, -de, deictic -i) | Strip suffix, re-lookup base form |
| **Verb morphology stripping** | prefix check `O(1)` | Unseen augmented/reduplicated verb forms | Strip augment/reduplication, re-lookup |
| **Dialect normalization** | k candidates `O(k)` | Ionic, Doric, Aeolic, Koine dialect forms | Map dialect forms to Attic equivalents |
| **Compound decomposition** | n=word length `O(n)` | Byzantine compound words | Split at linking vowel, look up base |
| **Spelling correction** | BK-tree `O(d·m)` | ED0-2 suggestions for unknown words | Accent-stripped edit distance |
| **Transformer** | beam search `O(b·n²)` | generalizes to unseen forms | Trained on Wiktionary pairs |

### Lookup table

The lookup table combines forms from multiple sources:

| Source | Forms | Notes |
|--------|------:|-------|
| **Wiktionary** (EN + EL, all periods) | 5.2M | Baseline from kaikki.org dumps |
| **LSJ** (Liddell-Scott-Jones) | 4.2M | 32K nouns, 22K verbs, 14K adjectives expanded via Wiktionary Lua modules |
| **Sophocles Lexicon** (Byzantine/Patristic) | 1.0M | 13.5K nouns, 4.6K verbs, 1.5K adverbs from OCR'd TEI data |
| **[GLAUx](https://github.com/alekkeersmaekers/glaux)** (Keersmaekers, 2021) | 557K | 17M-token corpus, 8th c. BC - 4th c. AD, 98.8% lemma accuracy |
| **[Diorisis](https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256)** (Vatri & McGillivray, 2018) | 76K new | 10M-token corpus, Homer - 5th c. AD, 91.4% lemma accuracy. Low-priority pairs (only added when no conflict with existing sources). Also provides frequency data (27M combined tokens with GLAUx). |
| **[HNC Golden Corpus](https://inventory.clarin.gr/corpus/870)** (CLARIN:EL) | 1K new | 88K-token gold-standard MG corpus, 11K unique form-lemma pairs. Low priority (only added when not in Wiktionary). Also used for MG evaluation. |
| **[PROIEL](https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL)** (UD treebank) | 33K | Herodotus gold-standard form-lemma pairs (expert-verified) |
| **[Perseus](https://github.com/UniversalDependencies/UD_Ancient_Greek-Perseus)** (UD treebank) | 42K | 178K tokens: Sophocles, Aeschylus, Homer, Hesiod, Herodotus, Thucydides, Plutarch, Polybius, Athenaeus |
| **[Gorman Treebanks](https://github.com/UD-Greek/UD_Ancient_Greek-Gorman)** (Gorman) | 79K | 687K-token corpus across Herodotus, Thucydides, Xenophon, Demosthenes, Lysias, Polybius, etc. Gold-standard single annotator. |
| Closed-class fixes | ~500 | Articles, pronouns, prepositions mapped to canonical lemmas |

The LSJ and Sophocles expansions use Wiktionary's own
[grc-decl](https://en.wiktionary.org/wiki/Module:grc-decl) and
[grc-conj](https://en.wiktionary.org/wiki/Module:grc-conj) Lua modules
(via [wikitextprocessor](https://github.com/tatuylonen/wikitextprocessor))
to generate inflection paradigms from headwords with grammatical metadata.
Cunliffe's Homeric Lexicon (~12K headwords) is not expanded this way
because its headwords are a subset of LSJ and already covered by the
LSJ expansion plus GLAUx Homeric corpus data (557K pairs).

The lookup table is built from Wiktionary [kaikki dumps](https://kaikki.org/)
(EN and EL editions for MG and AG, plus EL Medieval Greek), expanded with
inflected forms from LSJ (via Wiktionary Lua modules) and the Sophocles
lexicon of Roman and Byzantine Greek, then augmented with form-lemma pairs
from gold-standard treebanks (PROIEL, Gorman, AGDT). Each form is indexed under
its original, monotonic, and accent-stripped variants, so `θεοὶ` (polytonic
with grave), `θεοί` (monotonic with acute), and `θεοι` (stripped) all
resolve to `θεός`. Input can be polytonic, monotonic, or unaccented. AG
forms take priority over MG, ensuring classical lemma forms (βιβλίον,
φύσις, θεῖος) are preferred over their MG equivalents (βιβλίο, φύση,
θείο). Medieval Wiktionary entries are merged into the MG table at
build time. When `lang="el"` is used, 150K MG-specific entries
override the AG-first defaults with MG lemma forms (ο instead of ὁ,
είμαι instead of εἰμί). For polytonic input (breathings/circumflex),
an additional AG-only lookup pass runs first.

When the transformer handles an unseen form, beam search generates
multiple candidates and picks the first that matches a known headword
from the combined filter (~740K headwords from Wiktionary self-maps,
[LSJ9](https://github.com/ciscoriordan/lsj9) (119K entries + variants),
and [Cunliffe's Homeric Lexicon](https://archive.org/details/lexiconofhomeric0000cunn) (12K entries)).
If nothing matches, the input is returned unchanged.

**Wiktionary as upstream:** Because Dilemma's lookup tables are built
directly from Wiktionary, any missing or incorrect lemmatization can
often be fixed by editing the Wiktionary entry itself. When the kaikki
dumps are next regenerated and `build_data.py` re-run, the fix flows
into Dilemma automatically. This means the coverage and accuracy of
Dilemma improve over time as Wiktionary's Greek coverage improves,
without any changes to Dilemma's code.

### Rule-based fallback layer

The rule-based morphological analysis fills the gap between the lookup
table and the transformer:

**Particle stripping** (-περ, -γε, -δε, -ι): These are appended particles
that create forms the lookup table may never have seen. `ὅσπερ` is not a
separate word in Wiktionary - it is `ὅς` + `-περ`. The lookup table would
need to store every word x every enclitic particle combination. Stripping
is simpler.

**Augment/reduplication stripping**: A rare verb's aorist (e.g.,
`ἐμόρμυρσεν`) might not appear in any corpus or Wiktionary table, but the
present stem `μορμύρω` is there. Stripping the augment and tense markers
recovers the connection. The transformer might learn this pattern, but an
explicit rule is more reliable for rare verbs it has never trained on.

**Elision/crasis**: `δ'` needs to expand to `δέ`, `κἀγώ` needs to
decompose to `καὶ ἐγώ`. These are mechanical text artifacts, not
morphology - the transformer would waste capacity learning them.

In short: the rules handle systematic, predictable transformations that the
lookup table cannot enumerate exhaustively and the transformer might not
generalize to for rare forms. They are the cheap, reliable middle layer
between brute-force lookup and expensive ML inference.

### Dialect normalization

For Ancient Greek dialect texts (Herodotus, Pindar, Sappho, etc.),
the normalizer maps dialect-specific forms to their Attic equivalents
so the Attic-heavy lookup table can match them.

**Ionic** (highest coverage): η/ᾱ alternation after ε, ι, ρ
(ἱστορίης → ἱστορίας), uncontracted vowels (ποιέειν → ποιεῖν,
τιμέω → τιμῶ), κ/π interrogative interchange (κῶς → πῶς,
ὅκου → ὅπου), σσ/ττ alternation (θάλασσα → θάλαττα),
ρσ/ρρ alternation (θάρσος → θάρρος), and common word mappings
(μοῦνος → μόνος, ξεῖνος → ξένος, κεῖνος → ἐκεῖνος).

**Doric**: ᾱ/η alternation (Ἀθάνα → Ἀθήνη), word mappings
(ποτί → πρός, τύ → σύ), Doric futures (-σέω → -σω).

**Aeolic**: psilosis (smooth → rough breathing normalization).

**Koine**: σσ/ττ alternation (overlaps with Ionic and period rules).

```python
d = Dilemma(dialect="ionic")                              # Ionic texts
d = Dilemma(dialect="doric")                              # Doric texts
d = Dilemma(dialect="auto")                               # try all dialects
d = Dilemma(dialect="ionic", period="hellenistic")        # combined
```

Dialects can be combined with period profiles. Setting `dialect`
implicitly enables the normalizer (no need for `normalize=True`).

### Orthographic normalizer

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

### Transformer model

The transformer is a small (~4M param) character-level encoder-decoder,
the standard architecture from
[SIGMORPHON](https://sigmorphon.github.io/) morphological inflection
shared tasks. It learns character-level patterns and generalizes to forms
not in Wiktionary. Training on MG + AG + Medieval data means the model
sees AG augment patterns (`ἔλυσε` → `λύω`) alongside MG stem
transformations (`σκότωσε` → `σκοτώνω`). For Katharevousa forms like
`εσκότωσε`, it has both signals to draw from.

## API Reference

### Language and convention options

```python
from dilemma import Dilemma

d = Dilemma()                                  # all periods (default)
d_mg = Dilemma(lang="el")                     # MG only
d_grc = Dilemma(lang="grc")                   # AG only

# LSJ lemma convention
d_lsj = Dilemma(convention="lsj")

# Cunliffe convention
d_cun = Dilemma(convention="cunliffe")

# Triantafyllidis convention (recommended for MG)
d_mg = Dilemma(convention="triantafyllidis")
```

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

**Article-agreement disambiguation:** When multiple candidates exist, pass
the preceding word to rank by gender/number agreement with a Greek article:

```python
# Prefer candidates matching masculine article τόν
candidates = d.lemmatize_verbose("λόγου", prev_word="τοῦ")
# -> masculine λόγος ranked before proper Λόγος
```

This only re-ranks candidates, never excludes them. If the preceding word is
not a recognized article form, it has no effect.

Each `LemmaCandidate` has:
- `lemma` - the lemma string
- `lang` - `"el"` (MG, including medieval), `"grc"` (AG), `"med"` (medieval provenance label in output)
- `proper` - `True` if lemma is a proper noun (capitalized headword)
- `source` - `"lookup"`, `"elision"`, `"crasis"`, `"particle_strip"`, `"verb_morphology"`, `"compound"`, `"model"`, `"identity"`
- `via` - how it matched: `"exact"`, `"lower"`, `"elision:ε"`, `"suffix_strip"`, `"augment_strip"`, `"θεο+φθόγγος"`, `"+case_alt"`, etc.
- `score` - `1.0` for lookup, `0.5` for model, `0.0` for identity fallback

### Batch processing

When processing a large corpus (thousands of words), call `preload()` to
enable query-level caching on the SQLite lookup tables. This avoids
repeated SQLite round trips for forms that appear multiple times:

```python
d = Dilemma()
d.preload()  # enable query cache - ~40x faster for repeated lookups

for word in corpus:
    d.lemmatize_verbose(word)  # second lookup of same form is instant
```

`preload()` is safe to call multiple times (idempotent) and does not
change output - it only affects performance. It caches query results
on demand rather than loading the full 12M-entry table into memory.

### POS-aware disambiguation

When a POS tagger (e.g. [Opla](https://github.com/ciscoriordan/opla))
provides UPOS tags, `lemmatize_pos` uses POS to disambiguate between
multiple candidates from the regular lookup:

```python
d = Dilemma()
d.lemmatize_pos("αὐτοῦ", "ADV")    # "αὐτοῦ" (adverb: here/there)
d.lemmatize_pos("αὐτοῦ", "PRON")   # "αὐτός" (pronoun: genitive)
d.lemmatize_pos("ἄκρα", "NOUN")    # "ἄκρον" (noun: summit)
d.lemmatize_pos("ἄκρα", "ADJ")     # "ἄκρος" (adjective: outermost)
```

POS disambiguates rather than overrides: the regular lookup runs first to
produce all valid candidates, and POS selects among them only when there
are multiple options. When a form has just one candidate, POS is ignored,
ensuring POS-aware lemmatization never produces worse results than the
baseline.

With `convention="triantafyllidis"` or `lang="el"`, POS tags also fix MG
self-map issues for adjective and verb inflections. MG lookup tables
sometimes return self-maps for inflected forms (e.g. `ανθρώπινα` maps to
itself instead of `ανθρώπινος`). When POS is ADJ, the masculine nominative
citation form (-ος, -ής, -ύς) is preferred. When POS is VERB, the
infinitive/1sg form (-ω, -ώ, -μαι) is preferred. Adverbs and nouns keep
their MG self-maps unchanged.

The POS lookup tables (435K AG-only entries, 482K combined) are built
from six sources in priority order: UD treebanks (gold), LSJ9
indeclinables (2.2K adverbs, prepositions, conjunctions, particles,
interjections with unambiguous POS), GLAUx corpus (8.7K entries), MG
Wiktionary, AG Wiktionary, LSJ9 grammar. For polytonic input (breathing
marks, circumflex), the AG-only POS entries are checked first to avoid
MG lemma overrides on Ancient Greek text, mirroring the main lookup's
AG-first logic.

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
the input and the dictionary, collapsing the 12.5M-entry lookup into ~1-3M
unique base forms. ED0/ED1/ED2 matches are found on these stripped forms,
then expanded back to their original polytonic variants and ranked by true
Levenshtein distance. This means accent and breathing errors (wrong accent,
missing breathing mark) are corrected for free, while letter-level errors
(θ/δ, ρ/ν) use standard edit distance. The spell index is built lazily on
first call.

By default, suggestions include all forms in the lookup table (inflected
forms and lemmata from all sources). Two filtering options reduce false
positives when resolving to a specific dictionary:

```python
# Only return known LSJ headwords (strictest - 152K entries)
d.suggest_spelling("ἀγωνιστήριον", max_distance=1, headwords_only="lsj")

# Only return lemmata/citation forms (less strict - ~700K entries)
d.suggest_spelling("ἀγωνιστήριον", max_distance=1, lemmata_only=True)
```

You can also check headword membership directly:

```python
d.is_headword("θεός")              # True  (LSJ headword)
d.is_headword("θεοί")              # False (inflected form, not a headword)
d.is_headword("θεός", "cunliffe")  # check against Cunliffe headwords
```

### Elision expansion

Ancient Greek texts frequently elide final vowels before a following
vowel, marking the elision with an apostrophe (U+0313 in polytonic
encoding, U+02B9/U+02BC/U+1FBF/U+2019 in other encodings). Dilemma
resolves these by stripping the elision mark and trying each Greek vowel
against the lookup table:

| Elided | Expanded | Lemma |
|--------|----------|-------|
| `ἀλλ̓` | `ἀλλά` | `ἀλλά` |
| `δ̓` | `δέ` | `δέ` |
| `τ̓` | `τε` | `τε` |
| `ἐπ̓` | `ἐπί` | `ἐπί` |
| `ἔφατ̓` | `ἔφατο` | `φημί` |
| `κατ̓` | `κατά` | `κατά` |
| `καθ᾿` | `κατά` | `κατά` |
| `ἀφ᾿` | `ἀπό` | `ἀπό` |
| `βάλλ̓` | `βάλλε` | `βάλλω` |

**Consonant de-assimilation:** Before rough breathing, Greek assimilates
voiceless stops to aspirates (τ->θ, π->φ, κ->χ). The elision expander
reverses this: `καθ᾿` tries both `καθ-` and `κατ-`, `ἀφ᾿` tries both
`ἀφ-` and `ἀπ-`, recovering prepositions like κατά and ἀπό.

**Frequency ranking:** When multiple expansions match the lookup table,
candidates are ranked by corpus frequency (from GLAUx), so common
prepositions like κατά always beat obscure verbs like κάθω. Function
words are further prioritized when the stem matches a known elision
pattern, and proper nouns are deprioritized.

Polytonic input automatically restricts expansion to the AG lookup
table, avoiding false matches from MG monotonic forms.

## Greek Coverage

### Language codes

| Code | Period | ISO standard |
|------|--------|-------------|
| `el` | Modern Greek (including vernacular medieval, Katharevousa, regional) | ISO 639-1 |
| `grc` | Ancient Greek (Homer through Byzantine literary Greek) | ISO 639-2 |

Code and API calls use ISO 639 language codes: **`el`** for Modern Greek
and **`grc`** for Ancient Greek. In English text we often use the
shorthands **MG** (Modern Greek) and **AG** (Ancient Greek).

For Dilemma's purposes, MG (`el`) includes Katharevousa, even though
Katharevousa often benefits from AG lemmatization due to its archaizing
vocabulary and morphology. Medieval/Byzantine Greek has two components:
vernacular medieval Greek (ancestor of Modern Greek, merged into `el`)
and literary Byzantine Greek (classicizing, Atticist-influenced, resolved
via the AG lookup under `grc`).

For lemmatization, the two-way split works because Byzantine literary
Greek is classicizing (handled by `grc`), while vernacular medieval
Greek is the ancestor of Modern Greek (handled by `el`). The `med`
label still appears in `LemmaCandidate.lang` for forms from the
medieval Wiktionary dump, but these are merged into the `el` lookup
at build time.

Note: [Opla](https://github.com/ciscoriordan/opla) (POS tagging +
dependency parsing) uses `lang="grc"` for Byzantine text. Byzantine
literary syntax (polytonic, full case system, optative mood) is closer
to Ancient Greek, so the AG-trained POS tagger handles it well.

### Modern Greek varieties

| Variety | Wiktionary-tagged headwords |
|---------|---------------|
| **Standard Modern Greek (SMG/Demotic)** | 877K entries (core) |
| **Katharevousa** | 283+ tagged, hundreds more formal/place terms |
| **Cretan** | 273 |
| **Cypriot** | 199 |
| **Heptanesian (Ionian)** | 18 |
| **Maniot** | 3 |
| **Medieval/Byzantine (vernacular)** | 3K ([merged into MG](#medievalbyzantine-greek) - vernacular medieval is the ancestor of MG; literary Byzantine is Atticist-influenced and resolves via the AG lookup, not this table) |

### Ancient Greek varieties

| Variety | Wiktionary-tagged headwords |
|---------|---------------|
| **Epic/Homeric** | 3,755 |
| **Ionic** | 1,638 |
| **Attic** | 1,279 |
| **Koine** | 1,209 |
| **Byzantine (literary)** | 496 |
| **Doric** | 456 |
| **Aeolic** | 163 |
| **Laconian** | 52 |
| **Boeotian** | 15 |
| **Arcadocypriot** | 11 |

The counts above are Wiktionary headwords explicitly labeled with a
dialect tag. Each headword generates a full inflection paradigm (10-40
forms for verbs, 4-8 for nouns), so Wiktionary-derived form coverage is
much larger than the headword count suggests.

However, Wiktionary tags are only a fraction of Dilemma's actual dialect
coverage. Corpus-derived form-lemma pairs add substantially more:
GLAUx contributes 76K Ionic pairs from Herodotus and the Hippocratic
corpus, PROIEL adds 33K gold-standard Herodotus pairs, and Gorman adds
79K pairs across Herodotus, Thucydides, Xenophon, Demosthenes, and
others. The dialect normalization layer (Ionic, Doric, Aeolic, Koine)
then maps remaining dialectal forms to their Attic equivalents for
lookup, catching forms that no corpus or dictionary has catalogued.

Katharevousa forms are the primary non-SMG target for Modern Greek -
they mix AG morphology (augments, 3rd declension genitives) with MG
vocabulary. The strong Epic/Homeric coverage (3,755 tagged headwords
plus extensive GLAUx corpus data) is directly relevant for literary
texts based on Homer.

### Medieval/Byzantine Greek

<a id="why-medieval-is-mg"></a>
Medieval/Byzantine Greek has two distinct registers that Dilemma handles
differently. Vernacular medieval forms are merged into Modern Greek
(`el`) since they are the direct ancestor of MG. Literary Byzantine
forms are classicizing and resolve via the AG (`grc`) lookup.
EL Wiktionary's "Medieval Greek"
category (6,735 entries, 2,685 headwords) is roughly 71% vernacular
and 29% literary Byzantine, based on presence of polytonic diacritics:

- **Vernacular** (~71%): δέρνω, θυμώνω, χτενίζω, βρίσκω, γούνα,
  ναράντζι, βουρκόλακας, ξεχαρβαλώνω - early MG vocabulary
- **Literary Byzantine** (~29%): ἀποφθέγγομαι, αἰθεροπόρος,
  περικαλλής, κριθάλευρον - Atticist-influenced forms
- **Medieval-specific**: μαξιλάριν, ἀδελφάτον, κασσίδιον, ἴνδικτος,
  γαστάλδος - neither pure AG nor modern MG

Merging all into `el` works because the AG lookup runs first. The 29%
literary forms typically already exist in the AG table and resolve
there; only the vernacular and medieval-specific forms actually fall
through to the MG lookup. On the DBBE benchmark, only 2 of 8,342
tokens resolved via the medieval table, while 92.8% came from the AG
lookup.

## Development

### Full installation

```bash
git clone https://github.com/ciscoriordan/dilemma.git && cd dilemma
pip install -r requirements.txt
python build_data.py --download
python build_lookup_db.py              # SQLite for instant startup
python fix_selfmaps.py                 # fixes inflected forms that self-map
python train.py                        # full scale (~45 min on RTX 2080)
python export_onnx.py                  # optional: enable PyTorch-free inference
```

### Training

#### 1. Build data

Downloads all 5 kaikki dumps and extracts every form-lemma pair from
inflection tables. Non-Greek characters are filtered out.

```bash
pip install -r requirements.txt
python build_data.py --download             # downloads + extracts (~1.5GB total)
```

#### 2. Train

Trains the character-level transformer on the extracted pairs. Use
`--scale` to control the training size.

```bash
python train.py --scale test                # quick sanity check (20K pairs, ~15 sec)
python train.py --scale full                # all data (~45 min on RTX 2080, default)
python train.py                             # same as --scale full
```

Legacy `--scale 1/2/3` flags are still accepted for compatibility.

#### Training scales

Every scale includes **100% of non-standard varieties** (Medieval,
Katharevousa, Cypriot, Cretan, Maniot, Heptanesian, archaic, dialectal).
The remaining budget is split 50/50 between Ancient Greek and standard MG.
Underrepresented tense categories are oversampled to compensate for
their rarity in Wiktionary's paradigm tables, following
[Swaelens et al. (2025)](https://aclanthology.org/2025.acl-long.430/)'s
finding that perfects are underrepresented in training data relative
to Byzantine text. Aorist forms (3x, critical for stem-changing 2nd
aorist), perfect (3x), future (3x), imperfect (2x), and pluperfect
(5x, rarest at 0.15% of pool) are oversampled proportionally to
their rarity and the degree of stem change from the present form.

| Scale | Training pairs | Varieties | AG | SMG | Time (RTX 2080) |
|:-----:|---------------:|----------:|-------:|-------:|:--------------:|
| test | 20K | 9K (100%) | 5.5K | 5.5K | ~15 sec |
| full | 3.5M (all) | 9K (100%) | 1.5M (100%) | 1.7M (100%) | ~95 min |

Models save to `model/{lang}-test/` (test scale) or `model/{lang}/`
(full scale).

Eval accuracy is the model's score on held-out pairs *without* the
lookup table. In practice, the lookup resolves most forms instantly
and the model only handles truly novel words. When the model is used,
beam search generates 4 candidates and the first one that matches a
known headword in the lookup wins. If none match, the input is returned
unchanged (safe fallback).

#### Multi-task learning

When training pairs include POS tags (from Wiktionary) and morphological
features (from GLAUx), the model jointly predicts POS, nominal morphology
(gender/number/case, 45 labels), and verbal morphology (tense/mood/voice,
69 labels) alongside the lemma via auxiliary classification heads on the
encoder output. This follows
[Swaelens et al. (2025)](https://aclanthology.org/2025.acl-long.430/)'s
finding that multi-task learning (joint POS + morphology + lemma)
improved Byzantine Greek lemmatization by ~9 percentage points. Each
auxiliary loss is weighted at 0.1x relative to the lemmatization loss.
At full scale, the heads reach 90.4% POS, 81.5% nominal, and 91.2% verbal
accuracy on the held-out set.

Training uses a linear warmup LR scheduler (500 steps warmup, then linear
decay) and gradient clipping (max norm 1.0) for stable convergence.

### LSJ/Sophocles expansion

To regenerate the expanded lookup table from LSJ and Sophocles sources:

```bash
pip install --force-reinstall --no-deps git+https://github.com/tatuylonen/wikitextprocessor.git
python build/expand_lsj.py --setup           # build Wiktionary Lua module database
python build/expand_lsj.py --expand          # expand LSJ nouns
python build/expand_lsj.py --expand-verbs    # expand LSJ verbs
python build/expand_sophocles.py --expand    # expand Sophocles nouns
python build/expand_sophocles.py --expand-verbs  # expand Sophocles verbs
```

This requires LSJ9 data from [lsj9](https://github.com/ciscoriordan/lsj9)
(included in `data/lsjgr_bridges.json` and `data/lsj9_frequency.json`) and
the Sophocles TEI data (included in `data/sophocles/`).

### Export to ONNX

Generates ONNX model files so inference works without PyTorch.

```bash
python export_onnx.py                  # exports encoder.onnx + decoder_step.onnx
```

### Testing

Tests run automatically via GitHub Actions on push and pull request to
`main`, using a self-hosted runner with GPU access. CI downloads data
files from HuggingFace (`lookup.db`, `spell_index.db`, model weights).

```bash
python -m pytest tests/ -v                  # run all tests via pytest (recommended)
python tests/test_integrity.py              # data integrity + model inference checks
python tests/test_dilemma.py                # lookup table + end-to-end lemmatization tests
python tests/test_dilemma.py --lookup-only  # skip model tests
```

`tests/test_comprehensive.py` is the main pytest test suite (263 tests)
covering core lemmatization, particle suffix stripping, verb morphology
stripping, article-agreement disambiguation, crasis resolution, elision
handling, orthographic normalization, dialect normalization (Ionic, Doric,
Aeolic, Koine), convention switching, language filtering, spelling
suggestions, batch operations, PROIEL/Gorman treebank pairs, and edge
cases.

`tests/test_integrity.py` runs 7 structural checks: ONNX/vocab dimension
match, DB table presence, model load, inference, and ONNX/PyTorch
parity. `tests/test_dilemma.py` validates lookup correctness and known
form-lemma pairs across Greek varieties.

## Data

### Sources and scale

| Source | Forms | Notes |
|--------|------:|-------|
| EN + EL Wiktionary (MG) | 2.8M | From kaikki.org dumps |
| EN + EL Wiktionary (AG) | 2.4M | From kaikki.org dumps |
| EL Wiktionary (Medieval) | 6.9K | From kaikki.org dumps |
| LSJ noun/verb/adj expansion | 4.2M | Via Wiktionary Lua modules |
| Sophocles lexicon expansion | 1.0M | Byzantine/Patristic vocabulary |
| UD Treebanks (DiGreC) | 27K | Gold annotations from DiGreC treebank |
| [PROIEL](https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL) (gold) | 33K | Herodotus gold-standard form-lemma pairs (expert-verified) |
| [Perseus](https://github.com/UniversalDependencies/UD_Ancient_Greek-Perseus) (gold) | 42K | 178K tokens: Sophocles, Aeschylus, Homer, Hesiod, Herodotus, Thucydides, Plutarch, Polybius, Athenaeus |
| [Gorman Treebanks](https://github.com/UD-Greek/UD_Ancient_Greek-Gorman) | 79K | 687K tokens across Herodotus, Thucydides, Xenophon, Demosthenes, Lysias, Polybius, etc. |
| GLAUx corpus | 557K | 17M tokens, 98.8% accuracy ([Keersmaekers 2021](https://github.com/alekkeersmaekers/glaux)) |
| Diorisis corpus | 76K new | 10M tokens, 91.4% accuracy ([Vatri & McGillivray 2018](https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256)) |
| HNC Golden Corpus | 1K new | 88K-token gold MG corpus ([CLARIN:EL](https://inventory.clarin.gr/corpus/870), openUnder-PSI) |
| **Total lookup** | **12.5M** | |

All Wiktionary data is extracted automatically from
[kaikki.org](https://kaikki.org/) JSONL dumps. LSJ and Sophocles
expansions use wikitextprocessor to run Wiktionary's grc-decl and grc-conj
Lua modules on headwords extracted from lexicon XML/TEI files.

The [GLAUx corpus](https://github.com/alekkeersmaekers/glaux) provides
the largest single source of new form-lemma pairs outside Wiktionary.
GLAUx is the primary corpus source due to its 98.8% lemma accuracy.
The [Diorisis corpus](https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256)
(Vatri & McGillivray, 2018; 10M tokens, Homer - 5th c. AD) is used
as a secondary source: its 456K form-lemma pairs add 76K new entries
not found in GLAUx, and its token frequencies are merged with GLAUx
for 27M combined tokens. Because Diorisis has lower lemma accuracy
(91.4%), its pairs are only added when they don't conflict with existing
entries from Wiktionary, LSJ, or GLAUx.

We chose not to integrate one other large corpus:

- [Opera Graeca Adnotata](https://doi.org/10.5281/zenodo.14206061)
  (OGA, 40M tokens): standoff PAULA XML format requires complex
  alignment code, and at 91.4% accuracy with 4x the size of Diorisis,
  the noise-to-signal ratio is worse for lookup purposes.
- [Pedalion](https://github.com/perseids-publications/pedalion-trees)
  (5.8M tokens): smaller than GLAUx with similar classical-period
  coverage. Would add few forms not already covered by GLAUx + Wiktionary
  + LSJ, since the remaining lookup gaps are mostly Byzantine compounds
  not found in any classical corpus.

All three are CC BY-SA 4.0. Compound decomposition (added in v1.5)
reduced the no-lookup-hit rate on DBBE from 4.4% to 2.5% by splitting
compound words at linking vowels (ο/ι/υ), stripping known prefixes,
and applying Byzantine-specific normalizations. The remaining 2.5%
are forms where neither lookup, compound decomposition, nor the
seq2seq model can recover the correct lemma.

Each form is indexed under its original, monotonic, and accent-stripped
variants for fuzzy matching.

#### Extraction sources

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

#### Dialect tagging

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

#### Related work

[Vatri & McGillivray (2020)](https://brill.com/view/journals/jgl/20/2/article-p179_4.xml)
assessed the state of the art in Ancient Greek lemmatization via a
blinded evaluation by expert readers. They found that methods using
large lexica combined with POS tagging (CLTK backoff lemmatizer,
Diorisis corpus) consistently outperformed pure ML approaches with
smaller lexica. Dilemma follows the same principle: a large lookup
table (12.5M forms) handles the vast majority of words, with a small
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
Dilemma achieves 92.7% on the same dataset (equiv-adjusted).

[Swaelens et al. (2025)](https://aclanthology.org/2025.acl-long.430/)
showed that multi-task learning (joint POS + morphology + lemma
prediction) improved Byzantine lemmatization by ~9pp, reaching ~74-75%.
They also demonstrated that subword-tokenizing transformers plateau on
Byzantine Greek due to orthographic inconsistency, and called for
character-level models as the next step. Dilemma's character-level
encoder-decoder is this architecture, and its perfect tense oversampling
and multi-task POS head are directly informed by their findings.

#### Known issues

These are inherent limitations or Wiktionary coverage gaps, not code
bugs. Most can be fixed by editing the relevant Wiktionary entry, which
will propagate into Dilemma via kaikki dumps.

| Issue | Tokens | Notes |
|-------|--------|-------|
| **αὐτοῦ ambiguity** | ~200 | Genuine lexical ambiguity: both an adverb ("here/there") and genitive of αὐτός. Resolved when POS context is available via `lemmatize_pos()`. |
| **μιν → ὅς** | ~340 | Convention difference. Wiktionary maps μιν to the 3rd person pronoun. Perseus treebank uses μιν as its own lemma. |
| **Lemma convention differences** | ~400 | αὐτάρ vs ἀτάρ, κε vs ἄν - Wiktionary and Perseus use different citation forms for some Homeric particles. Handled by lemma equivalence groups for evaluation. |

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
| Inference | ONNX or PyTorch, beam search with headword filter |

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
| Approach | Subword tokenizer (UTF-8 bytes) | Character vocabulary (~381 Greek chars) |
| Parameters | 300M | 4M |
| Training (3.5M pairs, 3 epochs) | ~20 hours | ~95 min |
| Dependencies | torch + transformers | torch only (or ONNX only) |

## Credits

- Training data from [English Wiktionary](https://en.wiktionary.org/) and [Greek Wiktionary](https://el.wiktionary.org/) via [kaikki.org](https://kaikki.org/) JSONL dumps
- LSJ headwords, forms, and POS data from [LSJ9](https://github.com/ciscoriordan/lsj9) processed exports (`lsj9_headwords_flat.json`, `lsj9_headword_pos.json`, `lsj9_frequency.json`, `lsj9_indeclinables.json`)
- Sophocles lexicon TEI from [Ionian University / Internet Archive](https://archive.org/details/pateres)
- [GLAUx](https://github.com/alekkeersmaekers/glaux) corpus data (Keersmaekers, 2021) (CC BY-SA 4.0)
- [Diorisis](https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256) corpus data (Vatri & McGillivray, 2018) (CC BY-SA 3.0)
- [PROIEL Treebank](https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL) gold-standard annotations (CC BY-NC-SA 3.0)
- [Perseus Treebank](https://github.com/UniversalDependencies/UD_Ancient_Greek-Perseus) (AGDT) gold-standard annotations (CC BY-NC-SA 3.0)
- [Gorman Treebanks](https://github.com/UD-Greek/UD_Ancient_Greek-Gorman) (Gorman) (CC BY-NC-SA 4.0)
- [HNC Golden Corpus](https://inventory.clarin.gr/corpus/870) from CLARIN:EL (openUnder-PSI)
- DBBE evaluation data from [Swaelens et al.](https://github.com/coswaele/ByzantineGreekDatasets) (CC BY 4.0)
- Flag icons by [svg-flags](https://github.com/ciscoriordan/svg-flags)

## How to Cite

```
Francisco Riordan, Dilemma [computer software] (2026).
https://github.com/ciscoriordan/dilemma
```

## Upcoming

- `pip install dilemma` - PyPI package for easy installation

## License

MIT
