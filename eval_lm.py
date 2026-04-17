#!/usr/bin/env python3
"""Evaluate the next-word LM artifact ``build/lm/grc_ngram.bin``.

Reproduces the exact lookup the Swift reader in Tonos will perform:
binary search in the sorted vocab, binary search in the trigram index,
bigram fallback, unigram fallback. This gives us confidence that the
accuracy numbers we publish match what Tonos will actually see on
device.

Measures, over the held-out dev sentences from ``build/lm/dev_sentences.txt``:

  - top-1 next-word accuracy
  - top-3 next-word accuracy
  - top-5 next-word accuracy
  - UNK rate (share of gold next-words that are out of vocab)
  - perplexity (Stupid Backoff with α=0.4)
  - breakdown of backoff levels hit (trigram / bigram / unigram)

Writes ``build/lm/eval_results.txt``.

Usage
-----

    python eval_lm.py                    # default: whole dev set
    python eval_lm.py --max 5000         # limit sentences for a quick read
    python eval_lm.py --bin path/to.bin  # eval a non-default artifact
"""

from __future__ import annotations

import argparse
import bisect
import math
import mmap
import struct
import sys
import time
import unicodedata
from bisect import bisect_left
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_DIR = SCRIPT_DIR / "build" / "lm"

ALPHA = 0.4   # Stupid Backoff discount per level
LOGP_SCALE = 1024.0


class NgramLM:
    """mmap-backed reader for the grc_ngram.bin binary format.

    Mirrors the Swift reader's access pattern so eval numbers
    accurately predict on-device behaviour.
    """

    def __init__(self, path: Path):
        self.path = path
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._read_header()
        self._build_vocab_index()

    def close(self):
        self._mm.close()
        self._f.close()

    def _read_header(self):
        h = self._mm[:128]
        assert h[:4] == b"GNLM", "bad magic"
        self.format_version = struct.unpack_from("<I", h, 4)[0]
        self.order = struct.unpack_from("<I", h, 8)[0]
        self.top_k_uni = struct.unpack_from("<I", h, 12)[0]
        # legacy alias
        self.top_k = self.top_k_uni
        self.V = struct.unpack_from("<I", h, 16)[0]
        self.id_pad = struct.unpack_from("<I", h, 20)[0]
        self.id_unk = struct.unpack_from("<I", h, 24)[0]
        self.id_bos = struct.unpack_from("<I", h, 28)[0]
        self.id_eos = struct.unpack_from("<I", h, 32)[0]
        # v2: header offset 36 now carries top_k_bi. v1 had a reserved
        # 0 there; treat that as "same as top_k" so old artifacts keep
        # working in the reference reader.
        self.top_k_bi = struct.unpack_from("<I", h, 36)[0] or self.top_k_uni
        self.total_tokens = struct.unpack_from("<Q", h, 40)[0]
        self.vocab_offsets_off = struct.unpack_from("<Q", h, 48)[0]
        self.string_pool_off = struct.unpack_from("<Q", h, 56)[0]
        self.string_pool_size = struct.unpack_from("<Q", h, 64)[0]
        self.unigram_topk_off = struct.unpack_from("<Q", h, 72)[0]
        self.bigram_index_off = struct.unpack_from("<Q", h, 80)[0]
        self.bigram_index_len = struct.unpack_from("<I", h, 88)[0]
        self.bigram_suggestions_len = struct.unpack_from("<I", h, 92)[0]
        self.bigram_suggestions_off = struct.unpack_from("<Q", h, 96)[0]
        self.trigram_index_off = struct.unpack_from("<Q", h, 104)[0]
        self.trigram_index_len = struct.unpack_from("<I", h, 112)[0]
        self.trigram_suggestions_len = struct.unpack_from("<I", h, 116)[0]
        self.trigram_suggestions_off = struct.unpack_from("<Q", h, 120)[0]
        # v2: per-vocab counts sits between the string pool and the
        # unigram top-K table. Present only when format_version >= 2.
        if self.format_version >= 2:
            self.vocab_counts_off = (
                self.string_pool_off + self.string_pool_size
            )
        else:
            self.vocab_counts_off = None

    def _build_vocab_index(self):
        """Materialize token strings for binary search.

        The Swift reader binary-searches directly on the mmap'd bytes.
        For eval we decode the UTF-8 pool once since Python hits a
        comparable speed via ``bisect`` on a list.
        """
        self._vocab = [None] * self.V
        off_base = self.vocab_offsets_off
        pool_off = self.string_pool_off
        pool = self._mm[pool_off: pool_off + self.string_pool_size]
        offsets = struct.unpack_from(
            f"<{self.V + 1}I", self._mm, off_base
        )
        for i in range(self.V):
            self._vocab[i] = pool[offsets[i]: offsets[i + 1]].decode("utf-8")

    # ---- lookup ----

    def id_of(self, token: str) -> int:
        """Binary search for the token id. Returns UNK if not present."""
        i = bisect.bisect_left(self._vocab, token)
        if i < self.V and self._vocab[i] == token:
            return i
        return self.id_unk

    def _load_sug(self, base_off: int, sug_offset: int, count: int):
        """Load ``count`` (word_id, logprob) suggestions from the flat table."""
        start = base_off + sug_offset * 6
        out = []
        for j in range(count):
            wid, q = struct.unpack_from("<Ih", self._mm, start + j * 6)
            out.append((wid, q / LOGP_SCALE))
        return out

    def _trigram_lookup(self, w1: int, w2: int):
        """Binary search trigram index for (w1, w2)."""
        lo, hi = 0, self.trigram_index_len
        base = self.trigram_index_off
        target = (w1, w2)
        while lo < hi:
            mid = (lo + hi) // 2
            mw1, mw2 = struct.unpack_from("<II", self._mm, base + mid * 16)
            if (mw1, mw2) < target:
                lo = mid + 1
            else:
                hi = mid
        if lo >= self.trigram_index_len:
            return None
        mw1, mw2, sug_off, sug_cnt, _res = struct.unpack_from(
            "<IIIHH", self._mm, base + lo * 16
        )
        if (mw1, mw2) != target:
            return None
        return self._load_sug(self.trigram_suggestions_off, sug_off, sug_cnt)

    def _bigram_lookup(self, w1: int):
        """Binary search bigram index for w1."""
        lo, hi = 0, self.bigram_index_len
        base = self.bigram_index_off
        while lo < hi:
            mid = (lo + hi) // 2
            mw1 = struct.unpack_from("<I", self._mm, base + mid * 12)[0]
            if mw1 < w1:
                lo = mid + 1
            else:
                hi = mid
        if lo >= self.bigram_index_len:
            return None
        mw1, sug_off, sug_cnt, _res = struct.unpack_from(
            "<IIHH", self._mm, base + lo * 12
        )
        if mw1 != w1:
            return None
        return self._load_sug(self.bigram_suggestions_off, sug_off, sug_cnt)

    def _unigram_fallback(self):
        out = []
        for i in range(self.top_k):
            wid, q = struct.unpack_from(
                "<Ih", self._mm, self.unigram_topk_off + i * 6
            )
            out.append((wid, q / LOGP_SCALE))
        return out

    def predict(self, prev_ids: list[int]):
        """Return (level, [(word_id, logprob), ...]) suggestions.

        level is one of 'trigram' / 'bigram' / 'unigram'.
        """
        if len(prev_ids) >= 2:
            sug = self._trigram_lookup(prev_ids[-2], prev_ids[-1])
            if sug:
                return "trigram", sug
        if len(prev_ids) >= 1:
            sug = self._bigram_lookup(prev_ids[-1])
            if sug:
                return "bigram", sug
        return "unigram", self._unigram_fallback()

    def score_word(self, prev_ids: list[int], target_id: int) -> float:
        """Stupid Backoff log probability of target_id given context.

        Conservative fallback: if a level has no entries at all for the
        context, we drop down as normal. If a level has entries but
        target_id isn't among its top-K, we don't assume it's
        impossible; we fall down to the next level and apply α^depth.
        """
        logp_target = None
        depth = 0
        if len(prev_ids) >= 2:
            sug = self._trigram_lookup(prev_ids[-2], prev_ids[-1])
            if sug:
                for wid, lp in sug:
                    if wid == target_id:
                        logp_target = lp
                        break
                if logp_target is not None:
                    return logp_target
                depth += 1
        if len(prev_ids) >= 1:
            sug = self._bigram_lookup(prev_ids[-1])
            if sug:
                for wid, lp in sug:
                    if wid == target_id:
                        logp_target = lp + depth * math.log(ALPHA)
                        return logp_target
                depth += 1
        # unigram fallback; scored from top-K table, or a floor prob
        for wid, lp in self._unigram_fallback():
            if wid == target_id:
                return lp + depth * math.log(ALPHA)
        # not even in unigram top-K: assign a floor
        floor = math.log(1.0 / self.total_tokens) + depth * math.log(ALPHA)
        return floor


def load_dev_sentences(path: Path, max_n: int | None = None):
    sents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if toks:
                sents.append(toks)
            if max_n and len(sents) >= max_n:
                break
    return sents


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", default=str(BUILD_DIR / "grc_ngram.bin"))
    ap.add_argument("--dev", default=str(BUILD_DIR / "dev_sentences.txt"))
    ap.add_argument("--max", type=int, default=None,
                    help="limit to N dev sentences for a quick read")
    ap.add_argument("--out", default=str(BUILD_DIR / "eval_results.txt"))
    args = ap.parse_args()

    lm = NgramLM(Path(args.bin))
    sents = load_dev_sentences(Path(args.dev), args.max)
    print(f"Loaded {lm.V:,} vocab, {lm.bigram_index_len:,} bigram ctx, "
          f"{lm.trigram_index_len:,} trigram ctx")
    print(f"Evaluating on {len(sents):,} dev sentences")

    # metrics tracked in two regimes:
    #   "all"     : every scoreable position (incl. </s>, UNK targets)
    #   "kbd"     : keyboard-realistic; skip </s> and UNK targets, which
    #               the keyboard never needs to propose as suggestions
    n_all = n_all_top1 = n_all_top3 = n_all_top5 = n_all_top6 = 0
    n_kbd = n_kbd_top1 = n_kbd_top3 = n_kbd_top5 = n_kbd_top6 = 0
    n_unk_targets = 0
    n_eos_targets = 0
    level_counts = {"trigram": 0, "bigram": 0, "unigram": 0}
    sum_logp = 0.0
    n_logp = 0

    t0 = time.time()
    latencies = []

    for s_i, toks in enumerate(sents):
        # tokens already include <s> and </s> from train_lm.py
        ids = [lm.id_of(nfc(t)) for t in toks]
        # predict position i given ids[:i]; scored positions start at
        # index 1 so we always have some left context
        for i in range(1, len(ids)):
            prev = ids[max(0, i - 2): i]
            target = ids[i]
            ts = time.time()
            level, sug = lm.predict(prev)
            latencies.append(time.time() - ts)
            sug_ids = [w for w, _ in sug]
            level_counts[level] += 1

            is_unk = (target == lm.id_unk)
            is_eos = (target == lm.id_eos)
            if is_unk:
                n_unk_targets += 1
            if is_eos:
                n_eos_targets += 1

            n_all += 1
            if sug_ids and target == sug_ids[0]:
                n_all_top1 += 1
            if target in sug_ids[:3]:
                n_all_top3 += 1
            if target in sug_ids[:5]:
                n_all_top5 += 1
            if target in sug_ids[:6]:
                n_all_top6 += 1

            if not is_unk and not is_eos:
                n_kbd += 1
                if sug_ids and target == sug_ids[0]:
                    n_kbd_top1 += 1
                if target in sug_ids[:3]:
                    n_kbd_top3 += 1
                if target in sug_ids[:5]:
                    n_kbd_top5 += 1
                if target in sug_ids[:6]:
                    n_kbd_top6 += 1

            # perplexity component (all positions)
            lp = lm.score_word(prev, target)
            sum_logp += lp
            n_logp += 1

        if (s_i + 1) % 1000 == 0:
            print(f"  ...{s_i+1:,} sentences ({n_all:,} preds)",
                  flush=True)

    elapsed = time.time() - t0
    ppl = math.exp(-sum_logp / n_logp) if n_logp else float("inf")
    pred_per_sec = n_all / elapsed if elapsed > 0 else 0
    latencies.sort()
    p50 = latencies[len(latencies) // 2] * 1000
    p95 = latencies[int(len(latencies) * 0.95)] * 1000
    p99 = latencies[int(len(latencies) * 0.99)] * 1000

    rows = [
        f"artifact            : {args.bin}",
        f"format version      : {lm.format_version}",
        f"vocab size          : {lm.V:,}",
        f"top_k uni/bi/tri    : "
        f"{lm.top_k_uni}/{lm.top_k_bi}/max per-row",
        f"bigram contexts     : {lm.bigram_index_len:,}",
        f"trigram contexts    : {lm.trigram_index_len:,}",
        f"dev sentences       : {len(sents):,}",
        f"predictions scored  : {n_all:,}",
        "",
        "all-positions (includes </s> and UNK targets, research baseline):",
        f"  top-1 accuracy    : {100 * n_all_top1 / max(1, n_all):.2f}%",
        f"  top-3 accuracy    : {100 * n_all_top3 / max(1, n_all):.2f}%",
        f"  top-5 accuracy    : {100 * n_all_top5 / max(1, n_all):.2f}%",
        f"  top-6 accuracy    : {100 * n_all_top6 / max(1, n_all):.2f}%",
        f"  perplexity (SB)   : {ppl:,.1f}",
        f"  UNK target share  : "
        f"{100 * n_unk_targets / max(1, n_all):.2f}%",
        f"  </s> target share : "
        f"{100 * n_eos_targets / max(1, n_all):.2f}%",
        "",
        "keyboard-realistic (exclude </s> and UNK targets):",
        f"  scored positions  : {n_kbd:,}",
        f"  top-1 accuracy    : {100 * n_kbd_top1 / max(1, n_kbd):.2f}%",
        f"  top-3 accuracy    : {100 * n_kbd_top3 / max(1, n_kbd):.2f}%",
        f"  top-5 accuracy    : {100 * n_kbd_top5 / max(1, n_kbd):.2f}%",
        f"  top-6 accuracy    : {100 * n_kbd_top6 / max(1, n_kbd):.2f}%",
        "",
        "level breakdown (which n-gram table supplied the suggestion):",
        f"  trigram           : {level_counts['trigram']:,} "
        f"({100 * level_counts['trigram'] / max(1, n_all):.1f}%)",
        f"  bigram            : {level_counts['bigram']:,} "
        f"({100 * level_counts['bigram'] / max(1, n_all):.1f}%)",
        f"  unigram fallback  : {level_counts['unigram']:,} "
        f"({100 * level_counts['unigram'] / max(1, n_all):.1f}%)",
        "",
        "per-prediction latency (Python reference reader; Swift ~10x faster):",
        f"  p50               : {p50:.3f} ms",
        f"  p95               : {p95:.3f} ms",
        f"  p99               : {p99:.3f} ms",
        f"  throughput        : {pred_per_sec:,.0f} preds/sec",
    ]
    report = "\n".join(rows)
    print()
    print(report)

    Path(args.out).write_text(report + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
