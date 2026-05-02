"""Load and remap gr-nlp-toolkit checkpoint weights into TaggerModel."""

import torch
from huggingface_hub import hf_hub_download

from ._revisions import GR_NLP_TOOLKIT_REV


_REPO_ID = "AUEB-NLP/gr-nlp-toolkit"
_POS_FILENAME = "pos_processor"
_DP_FILENAME = "dp_processor"


def _download_weights(filename: str, cache_dir: str | None = None) -> str:
    """Download a weight file from HuggingFace if not already cached."""
    return hf_hub_download(
        repo_id=_REPO_ID,
        filename=filename,
        revision=GR_NLP_TOOLKIT_REV,
        cache_dir=cache_dir,
    )


def load_weights(
    model,
    pos_path: str | None = None,
    dp_path: str | None = None,
    device: str = "cpu",
):
    """Load gr-nlp-toolkit POS + DP weights into a TaggerModel (dual BERT).

    Remaps key names:
    - POS: _bert_model.* -> pos_bert.*, _linear_dict.{feat}.* -> pos_heads.{feat}.*
    - DP: _bert_model.* -> dp_bert.*, arc_head.* etc. load directly
    """
    if pos_path is None:
        pos_path = _download_weights(_POS_FILENAME)
    if dp_path is None:
        dp_path = _download_weights(_DP_FILENAME)

    pos_sd = torch.load(pos_path, map_location=device, weights_only=True)
    dp_sd = torch.load(dp_path, map_location=device, weights_only=True)

    new_sd = {}

    # POS BERT weights
    for k, v in pos_sd.items():
        if k.startswith("_bert_model."):
            new_sd["pos_bert." + k[len("_bert_model."):]] = v

    # POS heads
    for k, v in pos_sd.items():
        if k.startswith("_linear_dict."):
            new_sd[k.replace("_linear_dict.", "pos_heads.")] = v

    # DP BERT weights
    for k, v in dp_sd.items():
        if k.startswith("_bert_model."):
            new_sd["dp_bert." + k[len("_bert_model."):]] = v

    # DP heads (names match directly)
    skip_prefixes = ("_bert_model.", "_dp.")
    for k, v in dp_sd.items():
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        new_sd[k] = v

    # Filter out non-parameter buffers (e.g. position_ids)
    model_keys = set(model.state_dict().keys())
    new_sd = {k: v for k, v in new_sd.items() if k in model_keys}

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in TaggerModel: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in TaggerModel: {unexpected}")
