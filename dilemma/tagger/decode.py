"""Decode model outputs to structured token dicts."""

import torch

from .labels import pos_labels, pos_properties, dp_labels


def decode_batch(
    pos_logits: dict[str, torch.Tensor],
    arc_scores: torch.Tensor,
    rel_scores: torch.Tensor,
    word_masks: list[list[bool]],
    subword2word: list[dict],
    word_forms: list[list[str]],
    raw_forms: list[list[str]] | None = None,
) -> list[list[dict]]:
    """Decode a batch of model outputs into per-sentence token dicts.

    Args:
        pos_logits: {feature_name: (batch, seq, n_labels)} from model
        arc_scores: (batch, seq, seq) head scores
        rel_scores: (batch, seq, seq, numrels) relation scores
        word_masks: per-sentence first-subword boolean masks
        subword2word: per-sentence {subword_idx -> word_idx} mappings
        word_forms: per-sentence normalized word forms (stripped+lowered)
        raw_forms: per-sentence original word forms (polytonic), or None

    Returns:
        List of sentence results, each a list of token dicts:
        {"form": str, "upos": str, "feats": dict, "head": int, "deprel": str}
        If raw_forms is provided, each dict also includes "raw_form".
    """
    batch_size = arc_scores.shape[0]

    # Precompute POS predictions: argmax over last dim
    pos_preds = {}
    for feat, logits in pos_logits.items():
        pos_preds[feat] = torch.argmax(logits, dim=-1).cpu()  # (batch, seq)

    # DP predictions
    head_preds = torch.argmax(arc_scores, dim=-1).cpu()  # (batch, seq)

    # Gather relation scores at predicted heads, then argmax
    # rel_scores: (batch, seq, seq, numrels)
    # We need rel_scores[b, t, head_preds[b,t], :] for each b, t
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(head_preds)
    seq_idx = torch.arange(head_preds.shape[1]).unsqueeze(0).expand_as(head_preds)
    # gathered: (batch, seq, numrels)
    gathered_rels = rel_scores.cpu()[batch_idx, seq_idx, head_preds]
    deprel_preds = torch.argmax(gathered_rels, dim=-1)  # (batch, seq)

    results = []
    for b in range(batch_size):
        tokens = []
        mask = word_masks[b]
        s2w = subword2word[b]
        forms = word_forms[b]
        rforms = raw_forms[b] if raw_forms else None

        word_i = 0  # 0-indexed into forms list
        for j, is_first_subword in enumerate(mask):
            if not is_first_subword:
                continue

            # j is the subword position of this word's first subtoken
            # Skip [CLS] (j=0 is always False for special tokens)

            if word_i >= len(forms):
                break

            form = forms[word_i]

            # UPOS
            upos_idx = pos_preds["upos"][b, j].item()
            upos = pos_labels["upos"][upos_idx]

            # Morphological features (filtered by pos_properties)
            feats = {}
            valid_feats = pos_properties.get(upos, [])
            for feat in valid_feats:
                if feat not in pos_preds:
                    continue
                feat_idx = pos_preds[feat][b, j].item()
                feat_val = pos_labels[feat][feat_idx]
                if feat_val != "_":
                    feats[feat] = feat_val

            # DP: head and deprel
            head_subword = head_preds[b, j].item()
            head_word = s2w.get(head_subword, 0)

            deprel_idx = deprel_preds[b, j].item()
            deprel = dp_labels[deprel_idx]

            tok = {
                "form": form,
                "upos": upos,
                "feats": feats,
                "head": head_word,
                "deprel": deprel,
            }
            if rforms and word_i < len(rforms):
                tok["raw_form"] = rforms[word_i]
            tokens.append(tok)
            word_i += 1

        results.append(tokens)

    return results
