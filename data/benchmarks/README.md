# Lemmatization Benchmark Datasets

Three small benchmark datasets for evaluating Greek lemmatizers, covering three distinct registers of Greek: Classical Ancient Greek, Katharevousa, and Modern Demotic Greek.

## 1. Classical Ancient Greek (ag)

**Source:** Sextus Empiricus, *Pyrrhoniae Hypotyposes* (Outlines of Pyrrhonism) 1.1-1.8
**Edition:** First1KGreek / OpenGreekAndLatin (Mutschmann 1912 edition)
**License:** CC BY-SA 3.0
**Tokens:** 357 total (323 content + 34 punctuation), 9 sentences

**Why uncontaminated:**
- Sextus Empiricus is not in UD AG-Perseus (Homer, Sophocles, Plato, Herodotus, Hesiod, Aeschylus)
- Not in UD AG-PROIEL (New Testament, Herodotus)
- Not in the Gorman treebank (which covers Aeschines, Andocides, Antiphon, Appian, Aristotle, Athenaeus, Demosthenes, Diodorus Siculus, Dionysius of Halicarnassus, Herodotus, Isaeus, Isocrates, Josephus, Lysias, Plato, Plutarch, Polybius, Pseudo-Xenophon, Thucydides, Xenophon)
- DiGreC includes Sextus Empiricus with scattered sentences (from its collection of 655 texts/56K tokens), but only verb-selected sentences, not contiguous passages. This contiguous passage from the opening of PH is unlikely to overlap.

**Lemma conventions:** LSJ-style dictionary headwords. Verbs use the present active indicative 1sg as lemma (e.g., λέγω, not ἐρέω, for future forms). Adverbs in -ως derived from adjectives map to the adjective lemma (e.g., εὐλόγως -> εὔλογος, ἱστορικῶς -> ἱστορικός). Stand-alone adverbs with their own LSJ entry keep their form (ἴσως, πάντως, οὕτως, ἰδίως). Article lemma is ὁ. Indefinite τις vs. interrogative τίς are distinguished by accent.

## 2. Katharevousa (katharevousa)

**Source:** Konstantinos Sathas, *Neoelliniki Filologia* (1868), biography of Bessarion
**Edition:** Greek Wikisource transcription (el.wikisource.org), from PDF pages 31-32
**License:** Public domain (published 1868, author died 1914)
**Tokens:** 318 total (283 content + 35 punctuation), 9 sentences

**Why uncontaminated:**
- No known Katharevousa treebank exists in any dependency treebank collection
- DiGreC does not cover 19th-century Katharevousa scholarship
- This is a niche 19th-century biographical text about a 15th-century scholar
- The text mixes pure Katharevousa morphology with proper names from multiple languages (Niceron, Bandini, Baerner)

**Lemma conventions:** Standard AG dictionary headwords are used for classical vocabulary (the Katharevousa lexicon largely overlaps with AG). Proper names of modern scholars use the nominative form attested in the text as lemma (Νικέρων, Βανδίνης, Βαίρνερος). Place names use their standard dictionary forms (Τραπεζοῦς, Κωνσταντινούπολις, Πελοπόννησος). Numbers are kept as-is. Uncertain cases are marked with `?`.

## 3. Modern Greek / Demotic (demotic)

**Source:** Greek Wikipedia article "Σπήλαιο Πετραλώνων" (Petralona Cave)
**Revision:** Retrieved 2026-03-23 via MediaWiki API
**License:** CC BY-SA 4.0
**Tokens:** 269 total (242 content + 27 punctuation), 10 sentences

**Why uncontaminated:**
- No Modern Greek treebank covers Wikipedia articles about Petralona Cave
- The text covers a niche archaeological topic (cave archaeology, paleoanthropology)
- Greek Wikipedia is CC BY-SA, not typically used directly as treebank training data

**Lemma conventions:** Modern Greek monotonic dictionary forms. Article lemma is ο (not ὁ). Verbs use 1sg present active (βρίσκομαι for passive/deponent forms). Contracted preposition+article forms (στη, στους, στα, στο, στην, στις) lemmatize to σε. Numbers kept as-is. Proper nouns use their nominative form. Uncertain compound terms marked with `?`.

## File format

- `{register}.txt` - raw source text
- `{register}_gold.tsv` - gold standard: `form\tlemma` (one token per line, blank lines between sentences)
- `{register}_dilemma.tsv` - initial Dilemma predictions (for comparison)

Uncertain lemmas are marked with `?` suffix on the lemma (e.g., `περιθρυλλούμενον\tπεριθρυλλέω?`).

## Key corrections from Dilemma baseline

### AG corrections
- `εἰκὸς` -> `εἰκός` (adjective/substantive, not perfect of ἔοικα)
- `ἐροῦμεν` -> `λέγω` (Attic future of λέγω, not εἴρω "to string")
- `προειπόντες` -> `προλέγω` (aorist stem προεῖπον belongs to προλέγω)
- `φασιν` -> `φημί` (3pl present, not proper noun Φᾶσις)
- `Ἔστι` -> `εἰμί` (3sg present, not imperative Ἔσθι)
- `σκέψει` -> `σκέψις` (dative of noun, not verb σκέπτομαι in context)
- `οἷον` -> `οἷος` (relative pronoun, "such as")
- `τινές` -> `τις` (indefinite, not interrogative τίς)
- `τί`/`τίς`/`τίνες` -> `τίς` (interrogative in this context)
- `γινομένου` -> `γίγνομαι` (standard LSJ lemma, not non-reduplicated γίνομαι)
- `ἱστορικῶς` -> `ἱστορικός` (adverb mapped to adjective lemma)
- `ὑποτυπωτικῶς` -> `ὑποτυπωτικός` (adverb mapped to adjective lemma)

### Katharevousa corrections
- `ἠθῶν` -> `ἦθος` (gen pl of ἦθος "character/custom", not ἠθέω "to strain")
- `θρυλλουμένου` -> `θρυλέω` (Dilemma failed on compound analysis)
- `κατήρτισεν` -> `καταρτίζω` (Dilemma returned identity form)
- `διεσάλπιζε` -> `διασαλπίζω` (Dilemma returned identity form)
- `λατινισμῷ` -> `λατινισμός` (Dilemma failed on compound analysis)
- `διέψευσεν` -> `διαψεύδω` (Dilemma gave incorrect compound split)
- `καταψηφίζων` -> `καταψηφίζω` (Dilemma failed)
- `ἀναδειχθέντες` -> `ἀναδείκνυμι` (Dilemma returned identity)
- `ἐνδυθεὶς` -> `ἐνδύω` (Dilemma returned monotonic identity)
- Various proper name corrections for foreign scholars

### Demotic corrections
- Article/preposition conventions: ο (not ὁ), σε for στη/στους/στα/στο/στην/στις
- `βρίσκεται` -> `βρίσκομαι` (deponent in MG, separate dictionary entry from βρίσκω)
- `χαρακτηρίζεται` -> `χαρακτηρίζω` (passive voice of active verb, not a separate deponent)
- `συνεχίζεται` -> `συνεχίζω` (passive voice of active verb)
- `με` -> `με` (preposition, not ἐγώ)
- `η` -> `ο` (article, not ὅς)
- `καλύτερα` -> `καλός` (comparative, not adverb καλά)
- `διατηρημένα` -> `διατηρώ` (not participle-as-adjective)
- `κρανία` -> `κρανίο` (Modern Greek neuter, not ancient κρανίον)
- `οστά` -> `οστό` (Modern Greek form, not ancient ὀστέον)
- `περιπτώσεις` -> `περίπτωση` (MG form, not ancient περίπτωσις)
- Various MG vs AG lemma form differences

## Verification methodology

1. Dilemma was run on each text to get initial predictions
2. Every token was manually reviewed against:
   - LSJ (for AG and Katharevousa)
   - Wiktionary Greek entries (for all three registers)
   - Knowledge of Greek morphology (declension/conjugation patterns)
3. Ambiguous cases where the lemma depends on interpretation are resolved from context
4. Genuinely uncertain cases (rare words, unclear etymology) are marked with `?`
5. **Final human verification is still needed** - the user should review all entries

## Token count summary

| Benchmark     | Content tokens | Punctuation | Sentences | Uncertain |
|--------------|---------------|-------------|-----------|-----------|
| AG           | 323           | 34          | 9         | 0         |
| Katharevousa | 283           | 35          | 9         | 3         |
| Demotic      | 242           | 27          | 10        | 1         |
