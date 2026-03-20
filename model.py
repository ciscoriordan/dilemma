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

    # UPOS tags for multi-task POS prediction head
    UPOS_TAGS = [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
        "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB", "X",
    ]

    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3,
                 dim_feedforward=512, max_len=64, dropout=0.1,
                 num_pos_tags=0):
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

        # Optional multi-task prediction heads
        self.pos_head = None
        if num_pos_tags > 0:
            self.pos_head = nn.Linear(d_model, num_pos_tags)
        self.nom_head = None  # Gender+Number+Case
        self.verb_head = None  # Tense+Mood+Voice

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

    def _pool_encoder(self, src, src_key_padding_mask=None):
        """Mean-pool encoder output over non-padding positions."""
        memory = self.encode(src, src_key_padding_mask)
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask.unsqueeze(-1)
            return (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return memory.mean(dim=1)

    def predict_pos(self, src, src_key_padding_mask=None):
        """Predict POS tag from encoder output (multi-task head).

        Returns logits of shape (batch, num_pos_tags).
        """
        if self.pos_head is None:
            return None
        pooled = self._pool_encoder(src, src_key_padding_mask)
        return self.pos_head(pooled)

    def predict_morph(self, src, src_key_padding_mask=None):
        """Predict nominal and verbal morphology groups.

        Returns (nom_logits, verb_logits), either can be None if
        the corresponding head isn't initialized.
        """
        pooled = self._pool_encoder(src, src_key_padding_mask)
        nom = self.nom_head(pooled) if self.nom_head is not None else None
        verb = self.verb_head(pooled) if self.verb_head is not None else None
        return nom, verb

    @torch.no_grad()
    def generate(self, src, src_key_padding_mask=None, max_len=32,
                 bos_id=1, eos_id=2, num_beams=1):
        """Autoregressive decoding with optional beam search.

        Args:
            num_beams: 1 for greedy, >1 for beam search.
                Returns top-num_beams candidates per input.

        Returns:
            If num_beams == 1: (batch, seq_len) tensor
            If num_beams > 1: list of lists of (token_ids, score) tuples,
                one list per batch item, sorted best-first.
        """
        memory = self.encode(src, src_key_padding_mask)
        batch_size = src.size(0)
        device = src.device

        if num_beams == 1:
            # Greedy
            ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for _ in range(max_len):
                logits = self.decode(ys, memory,
                                     memory_key_padding_mask=src_key_padding_mask)
                next_token = logits[:, -1, :].argmax(dim=-1)
                next_token = next_token.masked_fill(finished, 0)
                ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
                finished = finished | (next_token == eos_id)
                if finished.all():
                    break
            return ys

        # Beam search (per-item, not batched, for simplicity)
        all_results = []
        for i in range(batch_size):
            mem_i = memory[i:i+1]  # (1, seq, d)
            mask_i = src_key_padding_mask[i:i+1] if src_key_padding_mask is not None else None

            # Each beam: (token_ids_list, cumulative_log_prob)
            beams = [([bos_id], 0.0)]
            complete = []

            for _ in range(max_len):
                candidates = []
                for ids, score in beams:
                    if ids[-1] == eos_id:
                        complete.append((ids, score))
                        continue
                    tgt = torch.tensor([ids], dtype=torch.long, device=device)
                    logits = self.decode(tgt, mem_i, memory_key_padding_mask=mask_i)
                    log_probs = logits[0, -1, :].log_softmax(dim=-1)
                    topk = log_probs.topk(num_beams)
                    for k in range(num_beams):
                        token = topk.indices[k].item()
                        new_score = score + topk.values[k].item()
                        candidates.append((ids + [token], new_score))

                if not candidates:
                    break
                # Keep top beams
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:num_beams]

                # Early stop if all beams ended
                if all(ids[-1] == eos_id for ids, _ in beams):
                    complete.extend(beams)
                    break

            # Add any incomplete beams
            complete.extend(b for b in beams if b[0][-1] != eos_id)
            complete.sort(key=lambda x: x[1], reverse=True)
            all_results.append(complete[:num_beams])

        return all_results
