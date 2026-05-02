"""Batched tokenization with subword-to-word mapping.

Replicates gr-nlp-toolkit's preprocessing exactly:
- NFD normalization + strip combining marks (category Mn) + lowercase
- HuggingFace BERT tokenizer for nlpaueb/bert-base-greek-uncased-v1
"""

import unicodedata
from typing import NamedTuple

import torch
from transformers import AutoTokenizer

from ._revisions import GREEK_BERT_REV


_TOKENIZER_NAME = "nlpaueb/bert-base-greek-uncased-v1"


def strip_accents_and_lowercase(s: str) -> str:
    """Exact replica of gr-nlp-toolkit's preprocessing."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).lower()


class BatchEncoding(NamedTuple):
    """Result of batch tokenization."""
    input_ids: torch.Tensor       # (batch, padded_seq_len)
    attention_mask: torch.Tensor  # (batch, padded_seq_len)
    word_masks: list[list[bool]]  # per-sentence first-subword masks
    subword2word: list[dict]      # per-sentence {subword_idx -> word_idx}
    word_forms: list[list[str]]   # per-sentence normalized word forms (stripped+lowered)
    raw_forms: list[list[str]]    # per-sentence original word forms (polytonic)


def _get_tokenizer():
    """Lazy-load and cache the tokenizer."""
    if not hasattr(_get_tokenizer, "_tok"):
        _get_tokenizer._tok = AutoTokenizer.from_pretrained(
            _TOKENIZER_NAME, revision=GREEK_BERT_REV
        )
    return _get_tokenizer._tok


def batch_tokenize(
    sentences: list[str],
    max_length: int = 512,
) -> BatchEncoding:
    """Tokenize a batch of sentences with padding and word mapping.

    Args:
        sentences: List of raw text sentences.
        max_length: Maximum subword sequence length (BERT limit).

    Returns:
        BatchEncoding with input_ids, attention_mask, word_masks,
        subword2word mappings, and original word forms.
    """
    tokenizer = _get_tokenizer()

    # Preprocess: strip accents + lowercase (matching gr-nlp-toolkit)
    normalized = [strip_accents_and_lowercase(s) for s in sentences]

    # Tokenize with padding
    enc = tokenizer(
        normalized,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )

    special_ids = set(tokenizer.all_special_ids)
    all_word_masks = []
    all_s2w = []
    all_forms = []
    all_raw_forms = []

    for i in range(len(sentences)):
        ids = enc.input_ids[i].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)

        # Split original sentence into whitespace tokens for alignment.
        # BERT's tokenizer splits on whitespace identically for both
        # original and normalized text (strip_accents_and_lowercase
        # preserves spaces), so word boundaries align 1:1.
        orig_words = sentences[i].split()

        mask = []
        s2w = {0: 0}  # CLS -> root (index 0)
        forms = []
        word_idx = 0
        current_ids = []

        for j, (tok, tok_id) in enumerate(zip(tokens, ids)):
            if tok_id in special_ids:
                # Special token ([CLS], [SEP], [PAD])
                mask.append(False)
                s2w[j] = 0
            elif tok.startswith("##"):
                # Continuation subword
                mask.append(False)
                s2w[j] = word_idx
                current_ids.append(tok_id)
            else:
                # First subword of a new word
                if current_ids:
                    # Decode previous word
                    forms.append(tokenizer.decode(current_ids))
                word_idx += 1
                mask.append(True)
                s2w[j] = word_idx
                current_ids = [tok_id]

        # Decode last word
        if current_ids:
            forms.append(tokenizer.decode(current_ids))

        # Build raw_forms from original words, aligned to BERT word indices.
        # If BERT produced fewer words (due to truncation), pad with the
        # normalized form; if more (shouldn't happen), truncate.
        raw_forms = []
        for w_i in range(len(forms)):
            if w_i < len(orig_words):
                raw_forms.append(orig_words[w_i])
            else:
                raw_forms.append(forms[w_i])

        all_word_masks.append(mask)
        all_s2w.append(s2w)
        all_forms.append(forms)
        all_raw_forms.append(raw_forms)

    return BatchEncoding(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        word_masks=all_word_masks,
        subword2word=all_s2w,
        word_forms=all_forms,
        raw_forms=all_raw_forms,
    )
