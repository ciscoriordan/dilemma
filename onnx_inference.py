"""ONNX inference backend for Dilemma.

Drop-in replacement for the PyTorch model's generate() method.
Uses ONNX Runtime instead of PyTorch, eliminating the ~2GB dependency.

Usage (called automatically by Dilemma when ONNX files exist):
    from onnx_inference import OnnxLemmaModel
    model = OnnxLemmaModel("/path/to/model_dir")
    results = model.generate(src_ids, src_pad_mask, num_beams=4)
"""

import json
import numpy as np
from pathlib import Path


class CharVocabLight:
    """Minimal vocab class that doesn't need PyTorch."""

    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self, vocab_path: Path):
        with open(vocab_path, encoding="utf-8") as f:
            d = json.load(f)
        self.char2id = d["char2id"]
        self.id2char = {int(k): v for k, v in d["id2char"].items()}

    def encode(self, text: str, add_bos=False, add_eos=False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.BOS)
        for ch in text:
            ids.append(self.char2id.get(ch, self.PAD))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids) -> str:
        chars = []
        for i in ids:
            i = int(i)
            if i == self.EOS:
                break
            if i in (self.PAD, self.BOS):
                continue
            chars.append(self.id2char.get(i, ""))
        return "".join(chars)

    def __len__(self):
        return len(self.char2id)


class OnnxLemmaModel:
    """ONNX-backed inference for the Dilemma transformer.

    Loads encoder.onnx and decoder_step.onnx, implements beam search
    in pure Python/NumPy. No PyTorch required.

    The ONNX models use fixed sequence lengths to avoid MHA reshape issues:
    - Encoder: ONNX_MAX_LEN (48) for source
    - Decoder: ONNX_MAX_OUT (32) for target
    Inputs are padded to these lengths; the causal mask handles the rest.
    """

    ONNX_MAX_OUT = 32  # must match export_onnx.py ONNX_TGT_LEN

    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        self.encoder = ort.InferenceSession(
            str(model_dir / "encoder.onnx"),
            providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(
            str(model_dir / "decoder_step.onnx"),
            providers=["CPUExecutionProvider"])

    def _encode(self, src: np.ndarray, src_mask: np.ndarray) -> np.ndarray:
        return self.encoder.run(None, {
            "src": src,
            "src_key_padding_mask": src_mask,
        })[0]

    def _decode_step(self, tgt_ids: list[int], memory: np.ndarray,
                     mem_mask: np.ndarray) -> np.ndarray:
        """Run one decoder step. Pads tgt to fixed length, returns logits
        for the last real token position.

        Args:
            tgt_ids: list of token IDs (the partial output so far)
            memory: (1, src_len, d_model) encoder output
            mem_mask: (1, src_len) bool encoder padding mask

        Returns:
            logits: (vocab_size,) for the next token prediction
        """
        real_len = len(tgt_ids)
        padded = tgt_ids + [0] * (self.ONNX_MAX_OUT - real_len)
        tgt = np.array([padded[:self.ONNX_MAX_OUT]], dtype=np.int64)

        # Decoder returns (1, ONNX_MAX_OUT, vocab_size)
        all_logits = self.decoder.run(None, {
            "tgt": tgt,
            "memory": memory,
            "memory_key_padding_mask": mem_mask,
        })[0]

        # Extract logits at the last real position (causal mask ensures
        # this position only sees tokens before it, not the padding)
        return all_logits[0, real_len - 1, :]

    def generate(self, src: np.ndarray, src_key_padding_mask: np.ndarray = None,
                 max_len=32, bos_id=1, eos_id=2, num_beams=1):
        """Beam search decoding using ONNX sessions.

        Args:
            src: (batch, src_len) int64 array of source token IDs
            src_key_padding_mask: (batch, src_len) bool array (True = padding)
            num_beams: beam width (1 = greedy)

        Returns:
            list of lists of (token_ids, score) tuples, matching the
            PyTorch model's beam search return format.
        """
        if isinstance(src, list):
            src = np.array(src, dtype=np.int64)
        if src_key_padding_mask is None:
            src_key_padding_mask = (src == 0)
        if isinstance(src_key_padding_mask, list):
            src_key_padding_mask = np.array(src_key_padding_mask, dtype=bool)

        memory = self._encode(src, src_key_padding_mask)
        batch_size = src.shape[0]

        all_results = []
        for i in range(batch_size):
            mem_i = memory[i:i+1]
            mask_i = src_key_padding_mask[i:i+1]

            beams = [([bos_id], 0.0)]
            complete = []

            for _ in range(max_len):
                candidates = []
                for ids, score in beams:
                    if ids[-1] == eos_id:
                        complete.append((ids, score))
                        continue

                    logits = self._decode_step(ids, mem_i, mask_i)

                    # Log softmax
                    max_val = logits.max()
                    log_probs = logits - max_val - np.log(
                        np.exp(logits - max_val).sum())

                    # Top-k
                    top_indices = np.argpartition(log_probs, -num_beams)[-num_beams:]
                    top_indices = top_indices[np.argsort(log_probs[top_indices])[::-1]]

                    for idx in top_indices:
                        token = int(idx)
                        new_score = score + float(log_probs[idx])
                        candidates.append((ids + [token], new_score))

                if not candidates:
                    break

                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:num_beams]

                if all(ids[-1] == eos_id for ids, _ in beams):
                    complete.extend(beams)
                    break

            complete.extend(b for b in beams if b[0][-1] != eos_id)
            complete.sort(key=lambda x: x[1], reverse=True)
            all_results.append(complete[:num_beams])

        return all_results
