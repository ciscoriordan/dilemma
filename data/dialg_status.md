# DIAL-G Status Check - 2026-03-19

## Overview

DIAL-G (Diachronic Interactive Lexicon of Greek) is a lexicographic database project
at the Department of Greek Philology, Democritus University of Thrace (DUTH), Komotini.
It aims to integrate all previous scientific dictionaries of Greek across all periods
(ancient, medieval, modern) into a single searchable database.

## Current Status: Active but stagnant

The website at http://dialg.helit.duth.gr/ is live and functional (Drupal 7, PHP 5.5.9
on Ubuntu - a very dated stack). The copyright footer reads 2026, and the "Free DIALG"
tier offers a search interface. However:

- The **last news entry** on the site is from **May 2018** (students checking the Lampe
  Patristic lexicon). No news posted in nearly 8 years.
- The **last conference presentation** listed is EUROMED 2017 in Volos.
- The "Full DIALG" is behind a login wall (username/password required).
- No API, no bulk download, no open data release, no GitHub presence, no Zenodo deposit.

## Database Size

- **1,632,498 entries** as of the current statistics page (up from 1,600,977 in Nov 2016,
  so about 31,500 entries added in ~9 years - very slow growth).

## What's Actually In It

The statistics page lists 50+ source dictionaries being digitized. Key ones with
significant entry counts:

| Source | Entries | Notes |
|--------|---------|-------|
| Grammar Recognition (DIALG team) | 413,544 | Morphological forms, not lemmas |
| LSJ | 116,461 | Liddell-Scott-Jones |
| LSK (Greek LSJ translation) | 112,185 | Moscho/Konstantinidis translation |
| LKN (Modern Greek) | 46,693 | Standard Modern Greek dictionary |
| LSJ Intermediate | 36,456 | Middle Liddell |
| Babiniotis 2nd ed. | 36,187 | Modern Greek |
| Lampe (Patristic) | 22,456 | Patristic Greek Lexicon |
| Koumanoudis | 7,487 | Neologisms from the Fall to modern times |
| Demetrakos | 1,143 | Mega Lexikon (barely started) |

Many dictionaries show 0 entries - listed as planned but not yet digitized. Cunliffe,
Kriaras (Medieval Greek), Trapp/LBG (Byzantine), DGE, and others are at 0.

## Supplementary Tools ("Apps")

The site lists several indices beyond the main lexicon:
- Manuscript index (Ευρετηριο Χειρογραφων)
- Hymn index (Ευρετηριο Υμνων)
- Hapax legomena index
- Historical persons index
- Chant terminology index
- Learned expressions index

These reflect the team's strength in Byzantine hymnography/musicology rather than
general lexicography.

## Contact Information

- **Email:** dialg.lexico@gmail.com
- **Project directors:**
  - Grigorios Papagiannis (gpapapia@helit.duth.gr) - Assoc. Prof. of Byzantine Philology
  - Nikolaos Siklafidis (siklafidis@gmail.com) - Philologist, MEd Theology
- **Facebook:** https://www.facebook.com/dialgreek
- **Address:** Dept. of Greek Philology, DUTH, University Campus, 69100 Komotini

Papagiannis's Academia.edu profile was last updated Feb 2025, with 13 papers listed
(mostly Byzantine philology, not lexicography specifically).

## Relevance to Dilemma

**Low.** The project has several issues that make it unlikely to be useful for us:

1. **No open data.** The full database is behind a login. There is no downloadable
   dataset, no API, no open-access dump. The "Free DIALG" tier lets you search one term
   at a time through a web form.

2. **Largely duplicates freely available sources.** The biggest components by entry count
   are LSJ (already available via Perseus/Logeion), LSJ Intermediate, and LKN - all of
   which we already have or can get from better-structured digital sources.

3. **Slow progress and apparent dormancy.** Only ~31K entries added in 9 years. Many
   planned dictionaries remain at 0 entries. No public-facing updates since 2018.

4. **Byzantine/hymnographic focus.** The team's expertise and the supplementary tools
   suggest the project leans heavily toward Byzantine musicology, which is tangential
   to our needs.

5. **Ancient infrastructure.** PHP 5.5.9 reached end-of-life in 2015. Drupal 7 reached
   end-of-life in Jan 2025. The SSL certificate is misconfigured. This does not suggest
   active development or maintenance.

## Worth Following Up?

**Probably not**, unless they eventually open-source their morphological recognition data
(the 413K "Grammar Recognition" entries). That could potentially be interesting for
form generation. But given the project's pace and closed nature, it would be more
productive to focus on sources that actually publish their data openly (Perseus, Logeion,
Wiktionary, Open Greek and Latin on GitHub).

If you want to reach out anyway, a short email to dialg.lexico@gmail.com asking whether
they plan to release data under an open license would be the lowest-effort way to check.

## References

- Digital Classicist Wiki: https://wiki.digitalclassicist.org/Diachronic_Interactive_Lexicon_of_Greek
- AWOL blog post (2019): http://ancientworldonline.blogspot.com/2019/07/dialg-diachronic-interactive-lexicon-of.html
- DUTH research programs page: https://helit.duth.gr/research-kedivim/research/
- Papagiannis on Academia.edu: https://duth.academia.edu/GrigoriosPapagiannis
