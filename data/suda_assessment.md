# Suda Assessment for Dilemma ag_lookup.json

**Date:** 2026-03-19
**Question:** Would the Suda add significant new vocabulary to Dilemma's lookup table?
**Answer:** No. The yield is modest - roughly 2,000-3,000 genuinely new lemmas, a ~1.5% increase over the existing 173K unique lemmas. Not worth the effort given available data extraction options.

## Numbers

- **ag_lookup.json:** 8,123,996 keys mapping to 173,529 unique lemmas (from Wiktionary + LSJ + Sophocles)
- **Suda entries:** 31,341 total (per Adler numbering, confirmed via dcthree/ancient-greek-lexica headword list)

### Breakdown of 31,341 Suda entries

| Category | Count | Notes |
|----------|-------|-------|
| Single-word entries | 27,361 | The only ones relevant for lemma lookup |
| Two-word entries | 2,354 | Mostly names ("Καῖσαρ Τιβέριος") or short phrases |
| Three+ word phrases | 1,626 | Proverbs, quotations, idioms - not useful for form lookup |

### Single-word overlap with ag_lookup (27,361 entries)

| Status | Count | % |
|--------|-------|---|
| Already in ag_lookup (exact lowercase match) | 3,901 | 14.3% |
| Match after stripping diacritics | 15,429 | 56.4% |
| **Total already covered** | **19,330** | **70.6%** |
| Not in ag_lookup | 8,031 | 29.4% |

The high diacritics-only mismatch (56.4%) reflects the Suda's polytonic accentuation conventions differing slightly from the forms already in ag_lookup. These are not new vocabulary - the same words are already covered.

### Classification of 8,031 not-in-lookup entries

Based on manual classification of a 100-entry random sample, extrapolated:

| Category | Estimated count | % of not-found |
|----------|----------------|----------------|
| Inflected forms (aorists, participles, subjunctives, etc.) | ~3,200 | 40% |
| Genuine new vocabulary lemmas | ~3,400 | 43% |
| Proper nouns (people, places, ethnic names) | ~1,250 | 16% |
| Biblical/Semitic names | ~140 | 2% |

The Suda frequently headwords inflected forms as they appear in classical texts (e.g., Κεκλάγξω, Ἀπεμορξάμην, Ἐφείλιξαν), explaining why so many entries are inflected rather than lemmatized.

Of the ~3,400 "genuine vocabulary" entries, many are:
- Hapax legomena or very rare poetic forms (Ἀκροκελαινιόων, Ἐνδιαεριανερινηχέτους)
- Compound words whose components are already known (Θαλαττοκοπεῖς, Τραπεζολοιχός)
- Variant/dialectal spellings of known words (Ἀμφισβατεῖν for ἀμφισβητεῖν)
- Late Greek / Byzantine coinages and Latin loanwords (Μαγκίπατος, Σπεκουλάτωρος)

**Realistic yield: ~2,000-3,000 genuinely new, useful lemmas** that would generate new inflected forms for the lookup table.

## Data Availability

### Available structured data

1. **dcthree/ancient-greek-lexica** (GitHub): Has `suda-headwords.csv` with all 31,341 Adler-numbered headwords in Unicode Greek. This is a clean headword list but contains **no definitions, no part-of-speech tags, and no morphological information**. Without POS tags, you cannot generate inflected forms for the lookup table.

2. **Suda On Line** (cs.uky.edu/~raphael/sol/sol-html/): Full text with translations, but only accessible through a web interface (CGI search) or static HTML pages. No API, no database dump, no XML/TEI export. The site structure has entry URLs like `sol-entries/alpha-1.html` through `omega-295.html`, but these 404 as of March 2026 - the static HTML mirror may be offline.

3. **ptrourke/sudareader** (GitHub): A Python scraper intended to extract SOL entries to JSON, but the core extraction scripts (`extract_entries.py`, `main.py`) are empty/unfinished. The `extract_entry.py` parser exists and can parse individual SOL HTML pages, extracting headword, Greek text, translation, and betacode-to-Unicode conversion. Not usable for bulk extraction as-is.

### What's missing

There is **no freely available structured dump** of the Suda with part-of-speech information. The dcthree headword list gives you the 31K headwords but not whether each is a noun, verb, adjective, proper noun, etc. Without POS, you cannot:
- Generate inflected forms (need to know declension/conjugation class)
- Filter out proper nouns vs. vocabulary
- Add entries to ag_lookup in the existing pipeline

The SOL translations could theoretically be scraped to infer POS, but this would require:
1. Scraping 31K HTML pages (if the static mirror comes back online)
2. NLP on the English translations to guess POS
3. Manual morphological classification for each new lemma
4. Integration into the Wiktionary-based form generation pipeline

## Verdict

**Not worth pursuing.** The Suda would add ~2,000-3,000 new lemmas (a ~1.5% increase) to a lookup table that already has 173K lemmas and 8.1M form keys. The vocabulary it uniquely contributes is heavily weighted toward:
- Rare/poetic forms and hapax legomena that users are unlikely to look up
- Byzantine-era coinages and Latin loanwords
- Highly specific compound words

The effort required to extract, classify, and generate forms for these entries is disproportionate to the benefit. Better sources of new vocabulary for Dilemma would be:
- Completing the LSJ Lua morphology expansion (covers the core classical vocabulary more systematically)
- Adding Hesychius glosses (many rare dialectal words with brief definitions that hint at POS)
- Expanding the Wiktionary coverage (continuously growing, already structured with POS and inflection tables)
