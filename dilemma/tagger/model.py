"""Dual-BERT POS + DP model. Two BERT forward passes (down from 19)."""

import torch
from torch import nn
from torch.nn import LeakyReLU

from .labels import pos_labels, dp_labels


class TaggerModel(nn.Module):
    """POS tagger + dependency parser with dual BERT backbones.

    Uses separate BERT instances for POS and DP (matching gr-nlp-toolkit's
    trained weights which diverged during separate training). Two forward
    passes total, down from gr-nlp-toolkit's 19.

    For AG models trained jointly, pos_bert and dp_bert can be the same
    instance (single BERT, single forward pass).
    """

    def __init__(self, pos_bert, dp_bert=None, feat_sizes=None, num_deprels=None):
        """
        Args:
            pos_bert: BERT model for POS features.
            dp_bert: BERT model for DP. If None, shares pos_bert.
            feat_sizes: Dict mapping feature name -> num labels. If None,
                uses full unified label set from labels.py.
            num_deprels: Number of dependency relation labels. If None,
                uses full unified set from labels.py.
        """
        super().__init__()
        self.pos_bert = pos_bert
        self.dp_bert = dp_bert if dp_bert is not None else pos_bert
        self.shared_bert = dp_bert is None
        self.dropout = nn.Dropout(0.0)

        # POS heads
        if feat_sizes is None:
            feat_sizes = {k: len(v) for k, v in pos_labels.items()}
        self.pos_heads = nn.ModuleDict({
            feat: nn.Linear(768, size)
            for feat, size in feat_sizes.items()
        })

        # DP heads: biaffine attention
        if num_deprels is None:
            num_deprels = len(dp_labels)
        self.numrels = num_deprels

        self.arc_head = nn.Linear(768, 768)
        self.arc_dep = nn.Linear(768, 768)
        self.rel_head = nn.Linear(768, 768)
        self.rel_dep = nn.Linear(768, 768)

        self.arc_bias = nn.Parameter(torch.zeros(1, 768, 1))
        self.rel_bias = nn.Parameter(torch.zeros(1, 1, 1, self.numrels))
        self.u_rel = nn.Parameter(torch.zeros(1, 768, self.numrels * 768))
        self.w_arc = nn.Parameter(torch.zeros(1, 768, 768))
        self.w_rel_head = nn.Parameter(torch.zeros(1, 1, 768, self.numrels))
        self.w_rel_dep = nn.Parameter(torch.zeros(1, 1, 768, self.numrels))

        self.deprel_linear_2 = nn.Linear(768, self.numrels * 768)
        self.relu = LeakyReLU(1)

    def forward(self, input_ids, attention_mask):
        """Forward pass producing POS logits and DP scores.

        If pos_bert and dp_bert are the same instance, only one BERT
        forward pass is needed.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask

        Returns:
            pos_logits: dict mapping feature name -> (batch, seq_len, n_labels)
            arc_scores: (batch, seq_len, seq_len) head scores
            rel_scores: (batch, seq_len, seq_len, numrels) relation scores
        """
        # POS: one BERT pass for all features
        pos_out = self.dropout(
            self.pos_bert(input_ids, attention_mask=attention_mask)[0]
        )
        pos_logits = {
            feat: head(pos_out) for feat, head in self.pos_heads.items()
        }

        # DP: reuse pos_out if shared, otherwise separate pass
        if self.shared_bert:
            dp_out = pos_out
        else:
            dp_out = self.dropout(
                self.dp_bert(input_ids, attention_mask=attention_mask)[0]
            )
        bs, mseq = dp_out.shape[:2]

        arc_h = self.relu(self.arc_head(dp_out))
        arc_d = self.relu(self.arc_dep(dp_out))
        rel_h = self.relu(self.rel_head(dp_out))
        rel_d = self.relu(self.rel_dep(dp_out))

        # Arc scores: (bs, mseq, mseq)
        arc_scores = (
            arc_h @ (arc_d @ self.w_arc).transpose(1, 2)
            + arc_h @ self.arc_bias
        )

        # Relation scores: biaffine
        label_biaffine = rel_d @ self.u_rel
        label_biaffine = label_biaffine.reshape(bs, mseq, self.numrels, 768)
        label_biaffine = label_biaffine @ rel_h.transpose(1, 2).unsqueeze(1)
        label_biaffine = label_biaffine.transpose(2, 3)

        label_head_affine = rel_h.unsqueeze(2) @ self.w_rel_head
        label_dep_affine = rel_d.unsqueeze(2) @ self.w_rel_dep

        rel_scores = (
            label_biaffine + label_head_affine + label_dep_affine
            + self.rel_bias
        )

        return pos_logits, arc_scores, rel_scores.reshape(bs, mseq, mseq, self.numrels)
