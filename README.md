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
it trains from scratch in minutes, compared to fine-tuning approaches
like *ByT5-small* (300M params) which take hours to train. Greek lemmatization is highly
pattern-based - a small specialized model matches a large general-purpose
one, and the 12.3M lookup table handles the rest.

**SQLite backend:** The lookup table loads from a pre-built SQLite database
(instant startup, ~0.3s) instead of parsing 600MB of JSON (~11s). Falls
back to JSON if the database isn't present.

**ONNX support:** Dilemma can run without PyTorch. When ONNX model files
are present, inference uses ONNX Runtime (~50 MB) instead of PyTorch (~2 GB).
The lookup table (which handles 95%+ of words) needs neither.

Handles Standard Modern Greek (Demotic), Katharevousa, Cypriot, Cretan,
and other regional varieties alongside Ancient and Medieval Greek.

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

Equiv-adjusted accuracy across four periods of Greek. All tools
evaluated with the same normalization (case-folded, accent-stripped)
and lemma equivalence groups (see `data/benchmarks/bench_all.py`).

**Test sets:**
- **AG Classical**: Sextus Empiricus, *Pyrrhoniae Hypotyposes* 1.1-1.8 (323 tokens, [First1KGreek](https://opengreekandlatin.github.io/First1KGreek/), CC BY-SA). Not in any UD treebank or Gorman.
- **Byzantine**: [Swaelens et al. (2024)](https://aclanthology.org/2024.lrec-main.899/) DBBE gold standard (8,342 tokens of unedited Byzantine epigrams, CC BY 4.0). Not in any tool's training data.
- **Katharevousa**: Konstantinos Sathas, *Neoelliniki Filologia* (1868), biography of Bessarion (283 tokens, [el.wikisource.org](https://el.wikisource.org/), public domain). No Katharevousa treebank exists.
- **Demotic MG**: Greek Wikipedia, "[Σπήλαιο Πετραλώνων](https://el.wikipedia.org/wiki/Σπήλαιο_Πετραλώνων)" (242 tokens, CC BY-SA). Not in any MG treebank.

| Tool | AG Classical | Byzantine | Katharevousa | Demotic MG |
|------|:--------:|:--------:|:--------:|:--------:|
| [spaCy](https://spacy.io/) `el` | -- | 31.7% | 44.6% | 79.9% |
| [stanza](https://stanfordnlp.github.io/stanza/) `el` | -- | 37.4% | 48.4% | 87.0% |
| [Swaelens et al. (2024)](https://aclanthology.org/2024.lrec-main.899/) | -- | 65.8% | -- | -- |
| [CLTK](https://github.com/cltk/cltk) | 81.2% | 66.6% | 74.8% | -- |
| [Morpheus](https://github.com/perseids-tools/morpheus-perseids-api) (oracle) | -- | 71.1% | -- | -- |
| [stanza](https://stanfordnlp.github.io/stanza/) `grc` | 92.2% | 71.3% | 85.2% | -- |
| [Swaelens et al. (2025)](https://aclanthology.org/2025.acl-long.430/) | -- | ~74-75% | -- | -- |
| **Dilemma** | **96.1%** | **91.7%** | **93.1%** | 78.1% |
| **Dilemma** (gold POS) | -- | **92.0%** | -- | -- |
| **Dilemma** MG | -- | -- | 89.3% | **94.0%** |

Dilemma is the only tool that covers all four periods. On Demotic MG,
using `convention="triantafyllidis"` (the "Dilemma MG" row) reaches
94.0%, well ahead of stanza `el` (87.0%). The default Dilemma row
(78.1%) uses AG conventions and is penalized by convention mismatch
(see [Lemma conventions](#lemma-conventions)). Morpheus "oracle" picks
the best candidate from all its analyses, representing the ceiling for
rule-based morphology. Cells marked `--` indicate the tool doesn't
support that period or wasn't tested.

Dilemma's remaining ~8.5% errors on DBBE break down as 3.1% no
lookup hit and 5.4% wrong lemma or convention difference. The eval
scripts (`eval/eval_dbbe.py`, `eval/eval_digrec.py`,
`eval/bench_dbbe.py`) provide per-POS breakdowns and error
categorization. `eval/eval_dbbe.py --use-pos gold` evaluates with
POS-aware disambiguation.

On the [DiGreC treebank](https://github.com/mdm33/digrec) (119K tokens,
Homer through 15th century Byzantine Greek), Dilemma reaches 94.0%
equiv-adjusted (90.3% strict). The gap accounts for convention
differences between annotation schemes (e.g. `εἶπον`/`λέγω`,
`ἐγώ`/`ἡμεῖς`).

### Modern Greek varieties

| Variety | Tagged entries |
|---------|---------------|
| **Standard Modern Greek (SMG/Demotic)** | 877K entries (core) |
| **Katharevousa** | 283+ tagged, hundreds more formal/place terms |
| **Cretan** | 273 |
| **Cypriot** | 199 |
| **Heptanesian (Ionian)** | 18 |
| **Maniot** | 3 |
| **Medieval/Byzantine** | 3K ([merged into MG](#why-medieval-is-mg) - vernacular medieval is the ancestor of MG; Byzantine literary Greek is Atticist-influenced and resolved via AG-first lookup, then MG fallback) |

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

By default, articles and pronoun clitics self-map (e.g. `τῆς` returns
`τῆς`). This is better for alignment pipelines where you want
surface-form matching. Set `resolve_articles=True` to resolve them
to canonical lemmas (`ὁ`, `ἐγώ`, `σύ`), matching treebank conventions
(AGDT, DiGreC, PROIEL). The `triantafyllidis` convention auto-enables
article resolution (articles to `ο`, skipping AG pronoun resolution
for forms like `σε`/`με` that are MG prepositions).

### Lemma conventions

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
overrides in `data/convention_{name}.json`. Other tools (stanza, spaCy,
CLTK) have fixed output conventions matching their training treebanks
and cannot be remapped.

In the benchmark table, the main Dilemma row uses the default
(Wiktionary) convention. The "Dilemma MG" row uses
`convention="triantafyllidis"`, which auto-enables article resolution
(articles to `ο`, demonstratives to `αυτός`) and outputs monotonic MG
lemma forms. This is the recommended setting for Modern Greek text.

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
- `lang` - `"el"` (MG, including medieval), `"grc"` (AG), `"med"` (medieval provenance label in output)
- `proper` - `True` if lemma is a proper noun (capitalized headword)
- `source` - `"lookup"`, `"elision"`, `"crasis"`, `"compound"`, `"model"`, `"identity"`
- `via` - how it matched: `"exact"`, `"lower"`, `"elision:ε"`, `"θεο+φθόγγος"`, `"+case_alt"`, etc.
- `score` - `1.0` for lookup, `0.5` for model, `0.0` for identity fallback

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

The POS lookup tables (435K AG-only entries, 482K combined) are built
from five sources in priority order: UD treebanks (gold), GLAUx corpus
(8.7K entries), MG Wiktionary, AG Wiktionary, LSJ9 grammar. For
polytonic input (breathing marks, circumflex), the AG-only POS entries
are checked first to avoid MG lemma overrides on Ancient Greek text,
mirroring the main lookup's AG-first logic.

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
the input and the dictionary, collapsing the 12.3M-entry lookup into ~1-3M
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
| **Lookup table** | `O(1)` hash lookup | 12.3M known forms | Wiktionary + LSJ + Sophocles + GLAUx + treebanks |
| **Normalizer** | `O(k)` k candidates | Byzantine orthographic variants | Rule-based candidate generation |
| **Elision expansion** | `O(v)` v=7 vowels | AG elided forms | Vowel expansion against lookup |
| **Crasis table** | `O(1)` hash lookup | ~50 common crasis forms | Hand-curated |
| **Compound decomposition** | `O(n)` n=word length | Byzantine compound words | Split at linking vowel, look up base |
| **Spelling correction** | `O(d·m)` BK-tree | ED0-2 suggestions for unknown words | Accent-stripped edit distance |
| **Transformer** | `O(b·n²)` beam search | generalizes to unseen forms | Trained on Wiktionary pairs |

The lookup table is built from Wiktionary [kaikki dumps](https://kaikki.org/)
(EN and EL editions for MG and AG, plus EL Medieval Greek), expanded with
inflected forms from LSJ (via Wiktionary Lua modules) and the Sophocles
lexicon of Roman and Byzantine Greek, then augmented with form-lemma pairs
from gold-standard treebanks (Gorman, AGDT). Each form is indexed under
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
from the combined filter (~820K headwords from Wiktionary self-maps,
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

The transformer is a small (~4M param) character-level encoder-decoder,
the standard architecture from
[SIGMORPHON](https://sigmorphon.github.io/) morphological inflection
shared tasks. It learns character-level patterns and generalizes to forms
not in Wiktionary. Training on MG + AG + Medieval data means the model
sees AG augment patterns (`ἔλυσε` → `λύω`) alongside MG stem
transformations (`σκότωσε` → `σκοτώνω`). For Katharevousa forms like
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
`decoder_step.onnx`) in the model directory provide transformer
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

### Testing

```bash
python tests/test_integrity.py              # data integrity + model inference checks
python tests/test_dilemma.py                # lookup table + end-to-end lemmatization tests
python tests/test_dilemma.py --lookup-only  # skip model tests
```

`tests/test_integrity.py` runs 7 structural checks: ONNX/vocab dimension
match, DB table presence, model load, inference, and ONNX/PyTorch
parity. `tests/test_dilemma.py` validates lookup correctness and known
form-lemma pairs across Greek varieties.

### LSJ/Sophocles expansion (optional, requires wikitextprocessor)

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
(via lsjpre export) and the Sophocles TEI data (included in `data/sophocles/`).

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
`--scale` to control the training size.

```bash
python train.py --scale test                # quick sanity check (20K pairs, ~15 sec)
python train.py --scale full                # all data (~45 min on RTX 2080, default)
python train.py                             # same as --scale full
```

Legacy `--scale 1/2/3` flags are still accepted for compatibility.

### Training scales

Every scale includes **100% of non-standard varieties** (Medieval,
Katharevousa, Cypriot, Cretan, Maniot, Heptanesian, archaic, dialectal).
The remaining budget is split 50/50 between Ancient Greek and standard MG.
Perfect tense verb forms are oversampled 3x, following
[Swaelens et al. (2025)](https://aclanthology.org/2025.acl-long.430/)'s
finding that perfects are underrepresented in training data (2%) relative
to Byzantine text (11.4%).

| Scale | Training pairs | Varieties | AG | SMG | Time (RTX 2080) |
|:-----:|---------------:|----------:|-------:|-------:|:--------------:|
| test | 20K | 9K (100%) | 5.5K | 5.5K | ~15 sec |
| full | 3.4M (all) | 9K (100%) | 1.5M (100%) | 1.7M (100%) | ~45 min |

Models save to `model/{lang}-test/` (test scale) or `model/{lang}/`
(full scale).

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
At full scale, the heads reach 90.4% POS, 81.5% nominal, and 91.2% verbal
accuracy on the held-out set.

Training uses a linear warmup LR scheduler (500 steps warmup, then linear
decay) and gradient clipping (max norm 1.0) for stable convergence.

Tests are a 55-case suite covering SMG, Epic, Attic, Koine, Byzantine,
Katharevousa, crasis, and model fallback across all resolution paths.

`Dilemma()` auto-detects the best available model:

```python
d = Dilemma()                         # auto-detect best available
```

<a id="why-medieval-is-mg"></a>
Medieval/Byzantine Greek forms are merged into Modern Greek (`el`),
not treated as a separate language. EL Wiktionary's "Medieval Greek"
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

### Language codes

| Code | Period | ISO standard |
|------|--------|-------------|
| `el` | Modern Greek (including vernacular medieval, Katharevousa, regional) | ISO 639-1 |
| `grc` | Ancient Greek (Homer through Byzantine literary Greek) | ISO 639-2 |

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
python train.py                        # full scale (~45 min on RTX 2080)
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
| Training (3.4M pairs, 3 epochs) | ~20 hours | ~45 min |
| Dependencies | torch + transformers | torch only (or ONNX only) |

ByT5 processes raw UTF-8 bytes, so a 10-character Greek word becomes
~20 encoder steps. Dilemma uses a Greek character vocabulary, so the
same word is ~10 steps. Combined with 75x fewer parameters, the
custom model trains much faster. Greek lemmatization is highly
pattern-based - a small specialized model matches a large
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
reduced the no-lookup-hit rate on DBBE from 4.4% to 3.1% by splitting
compound words at linking vowels (ο/ι/υ) and looking up the base
element. The remaining 3.1% are forms where neither lookup, compound
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

### Related work

[Vatri & McGillivray (2020)](https://brill.com/view/journals/jgl/20/2/article-p179_4.xml)
assessed the state of the art in Ancient Greek lemmatization via a
blinded evaluation by expert readers. They found that methods using
large lexica combined with POS tagging (CLTK backoff lemmatizer,
Diorisis corpus) consistently outperformed pure ML approaches with
smaller lexica. Dilemma follows the same principle: a large lookup
table (12.3M forms) handles the vast majority of words, with a small
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
Dilemma achieves 91.7% on the same dataset (equiv-adjusted).

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
- LSJ headwords and forms from [LSJ9](https://github.com/ciscoriordan/lsj9) (OCR-corrected LSJ base text, CC BY 4.0)
- LSJ grammar and indeclinables data from [LSJ9](https://github.com/ciscoriordan/lsj9) exports (lsj9_headwords.json, lsj9_forms.tsv, lsj9_indeclinables.json)
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
