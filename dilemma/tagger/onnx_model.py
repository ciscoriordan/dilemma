"""ONNX inference backend for the tagger.

Replaces PyTorch model with onnxruntime for CPU-only deployment.
Requires: pip install onnxruntime (~50MB vs ~2GB for torch+transformers)
"""

import json
import numpy as np
from pathlib import Path


class TaggerONNX:
    """ONNX-backed model that mimics TaggerModel's forward() output."""

    def __init__(self, onnx_dir: Path):
        import onnxruntime as ort

        meta_path = onnx_dir / "meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.feat_names = self.meta["feat_names"]

        # Find the ONNX model file
        joint_path = onnx_dir / "tagger_joint.onnx"
        if joint_path.exists():
            self.session = ort.InferenceSession(
                str(joint_path),
                providers=["CPUExecutionProvider"],
            )
        else:
            raise FileNotFoundError(f"No ONNX model found in {onnx_dir}")

        # Map output names
        self._output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, input_ids, attention_mask):
        """Run inference, returning outputs matching TaggerModel.forward().

        Processes each sentence individually to avoid ONNX dynamic shape
        issues with BERT's internal buffers, then stacks results.

        Args:
            input_ids: torch.Tensor (batch, seq_len)
            attention_mask: torch.Tensor (batch, seq_len)

        Returns:
            pos_logits: dict {feat_name: torch.Tensor (batch, seq, n_labels)}
            arc_scores: torch.Tensor (batch, seq, seq)
            rel_scores: torch.Tensor (batch, seq, seq, numrels)
        """
        import torch

        if hasattr(input_ids, "numpy"):
            input_ids_np = input_ids.cpu().numpy()
            attention_mask_np = attention_mask.cpu().numpy()
        else:
            input_ids_np = np.asarray(input_ids)
            attention_mask_np = np.asarray(attention_mask)

        batch_size = input_ids_np.shape[0]

        # Run each sentence separately to avoid dynamic shape issues
        all_outputs = []
        for i in range(batch_size):
            ids_i = input_ids_np[i:i+1].astype(np.int64)
            mask_i = attention_mask_np[i:i+1].astype(np.int64)
            out = self.session.run(
                None,
                {"input_ids": ids_i, "attention_mask": mask_i},
            )
            all_outputs.append(out)

        # Stack results across batch
        pos_logits = {}
        arc_scores_list = []
        rel_scores_list = []

        for name_idx, name in enumerate(self._output_names):
            if name.startswith("pos_"):
                feat = name[4:]
                stacked = np.concatenate(
                    [out[name_idx] for out in all_outputs], axis=0)
                pos_logits[feat] = torch.from_numpy(stacked)
            elif name == "arc_scores":
                arc_scores_list = [out[name_idx] for out in all_outputs]
            elif name == "rel_scores":
                rel_scores_list = [out[name_idx] for out in all_outputs]

        # Pad arc/rel scores to same seq_len before stacking
        max_seq = max(a.shape[1] for a in arc_scores_list)
        padded_arc = []
        padded_rel = []
        for arc, rel in zip(arc_scores_list, rel_scores_list):
            seq = arc.shape[1]
            if seq < max_seq:
                pad_width = max_seq - seq
                arc = np.pad(arc, ((0,0),(0,pad_width),(0,pad_width)))
                rel = np.pad(rel, ((0,0),(0,pad_width),(0,pad_width),(0,0)))
            padded_arc.append(arc)
            padded_rel.append(rel)

        arc_scores = torch.from_numpy(np.concatenate(padded_arc, axis=0))
        rel_scores = torch.from_numpy(np.concatenate(padded_rel, axis=0))

        return pos_logits, arc_scores, rel_scores

    def to(self, device):
        """No-op for ONNX (always runs on CPU)."""
        return self

    def eval(self):
        """No-op for ONNX (always in eval mode)."""
        return self
