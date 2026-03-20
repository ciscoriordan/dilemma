#!/usr/bin/env python3
"""Export Dilemma's transformer model to ONNX format.

Produces two ONNX files:
  - encoder.onnx: encodes source character sequence -> memory
  - decoder_step.onnx: single decoder step (memory + partial output -> logits)

The beam search loop stays in Python (onnx_inference.py), calling these
ONNX models instead of PyTorch. This eliminates the PyTorch dependency
for inference.

Usage:
    python export_onnx.py                    # export combined-s3 (default)
    python export_onnx.py --scale 2          # export specific scale
    python export_onnx.py --lang grc         # export specific language
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

from model import CharVocab, LemmaTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "model"


class EncoderWrapper(nn.Module):
    """Wraps the encoder for clean ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.embedding = model.embedding
        self.pos_enc = model.pos_enc
        self.encoder = model.encoder
        self.d_model = model.d_model

    def forward(self, src, src_key_padding_mask):
        x = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class DecoderStepWrapper(nn.Module):
    """Wraps decoder for ONNX export with fixed tgt length.

    Takes padded target sequence and memory, returns logits for ALL
    positions. The caller selects the position it needs. Causal mask
    is built inside so future padding doesn't affect earlier positions.
    """

    def __init__(self, model):
        super().__init__()
        self.embedding = model.embedding
        self.pos_enc = model.pos_enc
        self.decoder = model.decoder
        self.output_proj = model.output_proj
        self.d_model = model.d_model

    def forward(self, tgt, memory, memory_key_padding_mask):
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt.device)
        x = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, tgt_mask=tgt_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        return self.output_proj(x)  # (batch, tgt_len, vocab_size)


def export(lang="all", scale=None):
    lang_dir = {"el": "el", "grc": "grc", "all": "combined"}[lang]

    if scale is not None:
        model_path = MODEL_DIR / f"{lang_dir}-s{scale}"
    else:
        model_path = None
        for s in [3, 2, 1]:
            candidate = MODEL_DIR / f"{lang_dir}-s{s}"
            if (candidate / "model.pt").exists():
                model_path = candidate
                break
        if model_path is None:
            model_path = MODEL_DIR / lang_dir

    pt_path = model_path / "model.pt"
    if not pt_path.exists():
        print(f"No model at {pt_path}")
        sys.exit(1)

    print(f"Loading {pt_path}...")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    vocab = CharVocab()
    vocab.load_state_dict(checkpoint["vocab"])
    cfg = checkpoint["config"]
    model = LemmaTransformer(**cfg)
    # strict=False: skip auxiliary heads (POS, nom, verb) not needed for ONNX inference
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Save vocab separately (needed by ONNX inference)
    vocab_path = model_path / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab.state_dict(), f, ensure_ascii=False)
    print(f"  Saved vocab to {vocab_path}")

    # Export encoder
    encoder = EncoderWrapper(model)
    encoder.eval()

    # Use fixed sequence length of 48 - matches ONNX_MAX_LEN in dilemma.py.
    # ONNX MHA reshape ops aren't fully dynamic, so we pad all inputs to this.
    ONNX_SEQ_LEN = 48
    dummy_src = torch.randint(3, len(vocab), (1, ONNX_SEQ_LEN), dtype=torch.long)
    dummy_mask = torch.zeros(1, ONNX_SEQ_LEN, dtype=torch.bool)

    enc_path = model_path / "encoder.onnx"
    torch.onnx.export(
        encoder,
        (dummy_src, dummy_mask),
        str(enc_path),
        input_names=["src", "src_key_padding_mask"],
        output_names=["memory"],
        dynamic_axes={
            "src": {0: "batch"},
            "src_key_padding_mask": {0: "batch"},
            "memory": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"  Exported encoder to {enc_path} ({enc_path.stat().st_size / 1024:.0f} KB)")

    # Export decoder step
    decoder = DecoderStepWrapper(model)
    decoder.eval()

    # Fixed tgt length for decoder - must match ONNX_MAX_OUT in dilemma.py
    ONNX_TGT_LEN = 32
    dummy_tgt = torch.randint(1, len(vocab), (1, ONNX_TGT_LEN), dtype=torch.long)
    dummy_memory = torch.randn(1, ONNX_SEQ_LEN, cfg["d_model"])
    dummy_mem_mask = torch.zeros(1, ONNX_SEQ_LEN, dtype=torch.bool)

    dec_path = model_path / "decoder_step.onnx"
    torch.onnx.export(
        decoder,
        (dummy_tgt, dummy_memory, dummy_mem_mask),
        str(dec_path),
        input_names=["tgt", "memory", "memory_key_padding_mask"],
        output_names=["logits"],
        dynamic_axes={
            "tgt": {0: "batch"},
            "memory": {0: "batch"},
            "memory_key_padding_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"  Exported decoder to {dec_path} ({dec_path.stat().st_size / 1024:.0f} KB)")

    # Verify
    import onnxruntime as ort
    enc_sess = ort.InferenceSession(str(enc_path))
    dec_sess = ort.InferenceSession(str(dec_path))

    import numpy as np
    src_np = dummy_src.numpy()
    mask_np = dummy_mask.numpy()
    mem_out = enc_sess.run(None, {"src": src_np, "src_key_padding_mask": mask_np})[0]

    tgt_np = dummy_tgt.numpy()
    mem_mask_np = dummy_mem_mask.numpy()
    logits_out = dec_sess.run(None, {
        "tgt": tgt_np, "memory": mem_out, "memory_key_padding_mask": mem_mask_np
    })[0]

    print(f"  Verification: encoder output {mem_out.shape}, decoder output {logits_out.shape}")
    print("  ONNX export successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="all", choices=["el", "grc", "all"])
    parser.add_argument("--scale", type=int, default=None)
    args = parser.parse_args()
    export(args.lang, args.scale)
