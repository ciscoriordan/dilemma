"""Pinned HuggingFace Hub revisions for reproducible tagger installs.

Each tagger release pins the exact commit SHA of every external weight
source it downloads. This means:

- a tagger install today pulls the same blobs as the same version in a
  year, even if upstream HF repos have been updated.
- Security: a compromised HF account can't silently swap weights under us.
- Reproducibility: benchmark numbers stay meaningful across time.

To cut a new release that tracks updated upstream weights:
  1. Fetch the latest SHA for each repo (HF UI or
     `huggingface_hub.list_repo_refs`).
  2. Update the constants below.
  3. Re-run the slow test suite to confirm behavior.
  4. Bump dilemma.tagger's __version__.
"""

# AUEB-NLP's gr-nlp-toolkit checkpoint (pos_processor, dp_processor)
GR_NLP_TOOLKIT_REV = "5ddcf577a976b3402ba8810f181eea9ec202a70e"

# AUEB-NLP's GreekBERT backbone (Modern Greek)
GREEK_BERT_REV = "ec2b8f88dd215b5246f2f850413d5bff90d7540d"

# Pranaydeep Singh's Ancient-Greek-BERT backbone (grc + med)
ANCIENT_GREEK_BERT_REV = "5e3e29ece1d63029baa226f11105b1e8277c4f07"

# Our own trained AG/med heads, mirrored under ciscoriordan/dilemma at
# tagger/<lang>/tagger_<lang>.pt (legacy ciscoriordan/morphy is retained
# read-only for older installs).
TAGGER_WEIGHTS_REV = "35c500cc8de197e66a4eacc89d1e798e7bfa3980"


BERT_REVISIONS = {
    "nlpaueb/bert-base-greek-uncased-v1": GREEK_BERT_REV,
    "pranaydeeps/Ancient-Greek-BERT": ANCIENT_GREEK_BERT_REV,
}
