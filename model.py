"""Character-level encoder-decoder transformer for Greek lemmatization.

Small (~5M param) transformer trained from scratch on Greek character
sequences. Standard architecture from SIGMORPHON morphological
inflection/reinflection shared tasks.

Input: inflected Greek form as character sequence
Output: lemma as character sequence

Example:
    εσκότωσε -> σκοτώνω
    πολεμούσαν -> πολεμώ
"""

import math
import torch
import torch.nn as nn


class CharVocab:
    """Character-level vocabulary for Greek text."""

    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self):
        self.char2id = {"<pad>": self.PAD, "<bos>": self.BOS, "<eos>": self.EOS}
        self.id2char = {self.PAD: "", self.BOS: "", self.EOS: ""}
        self._frozen = False

    def fit(self, texts: list[str]):
        """Build vocabulary from a list of strings."""
        chars = set()
        for t in texts:
            chars.update(t)
        for ch in sorted(chars):
            idx = len(self.char2id)
            self.char2id[ch] = idx
            self.id2char[idx] = ch
        self._frozen = True
        return self

    def encode(self, text: str, add_bos=False, add_eos=False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.BOS)
        for ch in text:
            ids.append(self.char2id.get(ch, self.PAD))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i == self.EOS:
                break
            if i in (self.PAD, self.BOS):
                continue
            chars.append(self.id2char.get(i, ""))
        return "".join(chars)

    def __len__(self):
        return len(self.char2id)

    def state_dict(self):
        return {"char2id": self.char2id, "id2char": {int(k): v for k, v in self.id2char.items()}}

    def load_state_dict(self, d):
        self.char2id = d["char2id"]
        self.id2char = {int(k): v for k, v in d["id2char"].items()}
        self._frozen = True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LemmaTransformer(nn.Module):
    """Small encoder-decoder transformer for character-level lemmatization.

    Architecture:
        - Character embedding (vocab_size -> d_model)
        - Sinusoidal positional encoding
        - N-layer transformer encoder
        - N-layer transformer decoder with cross-attention
        - Linear output projection -> vocab_size

    Default config (~5M params):
        d_model=256, nhead=4, num_layers=3, dim_feedforward=512
    """

    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3,
                 dim_feedforward=512, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def _make_causal_mask(self, sz, device):
        return nn.Transformer.generate_square_subsequent_mask(sz, device=device)

    def encode(self, src, src_key_padding_mask=None):
        x = self.dropout(self.pos_enc(self.embedding(src) * math.sqrt(self.d_model)))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_key_padding_mask=None,
               memory_key_padding_mask=None):
        tgt_mask = self._make_causal_mask(tgt.size(1), tgt.device)
        x = self.dropout(self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model)))
        x = self.decoder(x, memory, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        return self.output_proj(x)

    def forward(self, src, tgt, src_key_padding_mask=None,
                tgt_key_padding_mask=None):
        """Forward pass for training. Returns logits over vocabulary."""
        memory = self.encode(src, src_key_padding_mask)
        return self.decode(tgt, memory, tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask)

    @torch.no_grad()
    def generate(self, src, src_key_padding_mask=None, max_len=32,
                 bos_id=1, eos_id=2):
        """Greedy autoregressive decoding."""
        memory = self.encode(src, src_key_padding_mask)
        batch_size = src.size(0)
        device = src.device

        # Start with BOS token
        ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.decode(ys, memory,
                                 memory_key_padding_mask=src_key_padding_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)  # (batch,)
            next_token = next_token.masked_fill(finished, 0)  # pad finished seqs
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        return ys  # (batch, seq_len) including BOS
