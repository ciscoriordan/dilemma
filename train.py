#!/usr/bin/env python3
"""Train Dilemma lemmatizer on Wiktionary form->lemma pairs.

Trains a small character-level encoder-decoder transformer (~5M params)
on Greek lemmatization. The model learns morphological patterns like
-ωσε -> -ώνω (aorist -> present stem) and generalizes to unseen forms.

Usage:
    python train.py --scale 0               # quick test (~30s)
    python train.py --scale 1               # default (~10 min on RTX 2080)
    python train.py --scale 4               # full data (~1h on 4090)
    python train.py --eval-only             # evaluate existing model

Prerequisites:
    pip install torch
    python build_data.py                    # must run first to generate pairs
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from model import CharVocab, LemmaTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "model"


class LemmaPairDataset(Dataset):
    def __init__(self, pairs, vocab, max_len=48):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        src = self.vocab.encode(p["form"])[:self.max_len]
        tgt = self.vocab.encode(p["lemma"], add_bos=True, add_eos=True)[:self.max_len]

        # Pad
        src = src + [0] * (self.max_len - len(src))
        tgt = tgt + [0] * (self.max_len - len(tgt))

        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
        }


# Tags that mark non-standard MG varieties — these get priority in training
# so they're never dropped when capping pair count.
PRIORITY_TAGS = {
    "Katharevousa", "archaic", "Cypriot", "Cretan", "Demotic",
    "dialectal", "polytonic", "Maniot", "Heptanesian", "literary",
    "dated", "vernacular", "rare", "formal",
}


def load_pairs(lang: str, max_pairs: int = 0) -> list[dict]:
    """Load training pairs, deduplicate, and apply priority ordering.

    Priority order (always included first, never dropped by max_pairs):
      1. Non-standard MG varieties (Medieval, Katharevousa, Cypriot, etc.)
      2. Remaining budget split 50/50 between AG and standard MG

    lang="both" combines MG and AG pairs for a single model that handles
    katharevousa (which mixes AG morphology with MG vocabulary).
    """
    if lang == "both":
        prefixes = ["mg", "ag", "med"]
    else:
        prefixes = [{"el": "mg", "grc": "ag", "mgr": "med"}[lang]]

    # Load raw pairs by source
    by_source = {}
    for prefix in prefixes:
        path = DATA_DIR / f"{prefix}_pairs.json"
        if not path.exists():
            print(f"Warning: {path} not found. Run build_data.py first.")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {prefix}: {len(data)} raw pairs")
        by_source[prefix] = data

    if not by_source:
        print("Error: no training data found.")
        sys.exit(1)

    # Deduplicate globally
    seen = set()
    def dedup(pairs_list):
        result = []
        for p in pairs_list:
            key = (p["form"], p["lemma"])
            if key not in seen:
                seen.add(key)
                result.append({"form": p["form"], "lemma": p["lemma"]})
        return result

    # Separate pools
    varieties = []  # always 100%: non-SMG MG + Medieval
    ag_pool = []
    smg_pool = []

    # All Medieval
    if "med" in by_source:
        med = dedup(by_source["med"])
        varieties.extend(med)
        print(f"  varieties: {len(med)} Medieval pairs")

    # Non-standard MG (Katharevousa, Cypriot, Cretan, etc.)
    if "mg" in by_source:
        mg_priority = []
        mg_standard = []
        for p in by_source["mg"]:
            tags = set(p.get("tags", []))
            if tags & PRIORITY_TAGS:
                mg_priority.append(p)
            else:
                mg_standard.append(p)
        mg_pri = dedup(mg_priority)
        varieties.extend(mg_pri)
        print(f"  varieties: {len(mg_pri)} non-standard MG pairs")
        smg_pool = dedup(mg_standard)
        print(f"  SMG pool: {len(smg_pool)} pairs")

    # Ancient Greek
    if "ag" in by_source:
        ag_pool = dedup(by_source["ag"])
        print(f"  AG pool: {len(ag_pool)} pairs")

    # Build final set: all varieties + 50/50 AG/SMG for remaining budget
    if max_pairs > 0 and len(varieties) + len(ag_pool) + len(smg_pool) > max_pairs:
        remaining = max(0, max_pairs - len(varieties))
        half = remaining // 2
        random.shuffle(ag_pool)
        random.shuffle(smg_pool)
        ag_sample = ag_pool[:half]
        smg_sample = smg_pool[:remaining - half]
        print(f"  mix: {len(varieties)} varieties + {len(ag_sample)} AG + "
              f"{len(smg_sample)} SMG = {len(varieties) + len(ag_sample) + len(smg_sample)} total")
        pairs = varieties + ag_sample + smg_sample
    else:
        pairs = varieties + ag_pool + smg_pool

    random.shuffle(pairs)
    print(f"Total training pairs: {len(pairs)}")
    return pairs


def evaluate(model, vocab, eval_pairs, device, batch_size=256):
    """Evaluate model accuracy on held-out pairs."""
    model.eval()
    correct = 0
    total = 0

    for i in range(0, len(eval_pairs), batch_size):
        batch = eval_pairs[i:i + batch_size]
        forms = [p["form"] for p in batch]
        expected = [p["lemma"] for p in batch]

        # Encode source
        max_len = max(len(f) for f in forms) + 1
        src_ids = []
        for f in forms:
            ids = vocab.encode(f)
            ids = ids + [0] * (max_len - len(ids))
            src_ids.append(ids)
        src = torch.tensor(src_ids, dtype=torch.long, device=device)
        src_pad_mask = (src == 0)

        with torch.no_grad():
            out_ids = model.generate(src, src_key_padding_mask=src_pad_mask)

        for j, exp in enumerate(expected):
            pred = vocab.decode(out_ids[j].tolist())
            total += 1
            if pred == exp:
                correct += 1

    accuracy = correct / total if total else 0
    return accuracy, correct, total


def train(lang: str, epochs: int, batch_size: int, lr: float, eval_split: float,
          max_pairs: int = 0, scale: int = None):
    """Train the lemmatizer model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pairs = load_pairs(lang, max_pairs=max_pairs)

    # Split train/eval
    random.shuffle(pairs)
    split = int(len(pairs) * (1 - eval_split))
    train_pairs = pairs[:split]
    eval_pairs = pairs[split:]
    print(f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    # Build character vocabulary from all forms and lemmas
    all_texts = [p["form"] for p in pairs] + [p["lemma"] for p in pairs]
    vocab = CharVocab()
    vocab.fit(all_texts)
    print(f"Vocabulary: {len(vocab)} characters")

    # Create model
    model = LemmaTransformer(vocab_size=len(vocab))
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count / 1e6:.1f}M parameters")

    dataset = LemmaPairDataset(train_pairs, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=device == "cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

    total_batches = len(loader)
    print(f"\nTraining for {epochs} epochs ({total_batches} batches/epoch)...",
          flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        batches = 0
        t0 = time.time()

        for batch_data in loader:
            src = batch_data["src"].to(device)
            tgt = batch_data["tgt"].to(device)

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_pad_mask = (src == 0)
            tgt_pad_mask = (tgt_in == 0)

            logits = model(src, tgt_in,
                           src_key_padding_mask=src_pad_mask,
                           tgt_key_padding_mask=tgt_pad_mask)

            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

            if batches % 100 == 0:
                elapsed = time.time() - t0
                rate = batches / elapsed
                eta = (total_batches - batches) / rate
                msg = (f"    [{batches}/{total_batches}] loss={total_loss/batches:.4f} "
                       f"({rate:.1f} batch/s, ETA {eta/60:.0f}m)")
                print(msg, flush=True)
                with open(SCRIPT_DIR / "progress.log", "a") as pf:
                    pf.write(msg + "\n")

        avg_loss = total_loss / batches
        elapsed = time.time() - t0

        # Evaluate every epoch
        accuracy, correct, total = evaluate(
            model, vocab, eval_pairs[:2000], device
        )
        msg = (f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, "
               f"eval={correct}/{total} ({accuracy:.1%}), {elapsed:.0f}s")
        print(msg, flush=True)
        with open(SCRIPT_DIR / "progress.log", "a") as pf:
            pf.write(msg + "\n")

    # Save model
    lang_dir = {"el": "el", "grc": "grc", "mgr": "med", "both": "combined"}[lang]
    if scale is not None:
        lang_dir = f"{lang_dir}-s{scale}"
    out_dir = MODEL_DIR / lang_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab.state_dict(),
        "config": {
            "vocab_size": len(vocab),
            "d_model": 256,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
        },
    }, out_dir / "model.pt")
    print(f"\nModel saved to {out_dir}")

    # Final evaluation on full eval set
    accuracy, correct, total = evaluate(model, vocab, eval_pairs, device)
    print(f"Final eval: {correct}/{total} ({accuracy:.1%})")

    # Test on specific words
    print("\nTest predictions:")
    test_words = ["πάθης", "εσκότωσε", "πολεμούσαν", "δώση", "εσήκωσε",
                   "εφώναξε", "ήρωας", "Ψάλλε", "τρομερό", "ανδρειωμένων",
                   "ἔλυσε", "θεοὶ", "μῆνιν"]
    max_len = max(len(w) for w in test_words) + 1
    src_ids = []
    for w in test_words:
        ids = vocab.encode(w)
        ids = ids + [0] * (max_len - len(ids))
        src_ids.append(ids)
    src = torch.tensor(src_ids, dtype=torch.long, device=device)
    src_pad_mask = (src == 0)
    with torch.no_grad():
        out_ids = model.generate(src, src_key_padding_mask=src_pad_mask)
    for w, ids in zip(test_words, out_ids):
        pred = vocab.decode(ids.tolist())
        try:
            print(f"  {w:25s} -> {pred}")
        except UnicodeEncodeError:
            print(f"  {w} -> {pred}".encode("utf-8", errors="replace").decode())


def main():
    parser = argparse.ArgumentParser(description="Train Dilemma lemmatizer")
    parser.add_argument("--lang", type=str, default="both",
                        choices=["el", "grc", "mgr", "both"],
                        help="Language: el (MG), grc (AG), mgr (Medieval), or both (MG+AG+Med, default)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--eval-split", type=float, default=0.05,
                        help="Fraction held out for eval (default: 0.05)")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Cap training pairs (0 = unlimited, overrides --scale)")
    parser.add_argument("--scale", type=int, default=1, choices=[0, 1, 2, 3, 4],
                        help="GPU scale: 0=20K pairs (~30s), "
                             "1=500K (~10 min), "
                             "2=1M (~20 min), "
                             "3=2M (~40 min), "
                             "4=all (~1h)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate existing model without training")
    args = parser.parse_args()

    if args.eval_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lang_dir = {"el": "el", "grc": "grc", "mgr": "med", "both": "combined"}[args.lang]
        out_dir = MODEL_DIR / lang_dir

        checkpoint = torch.load(out_dir / "model.pt", map_location=device,
                                weights_only=False)
        vocab = CharVocab()
        vocab.load_state_dict(checkpoint["vocab"])
        cfg = checkpoint["config"]
        model = LemmaTransformer(**cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        pairs = load_pairs(args.lang, max_pairs=0)
        random.shuffle(pairs)
        accuracy, correct, total = evaluate(
            model, vocab, pairs[:5000], device
        )
        print(f"Eval: {correct}/{total} ({accuracy:.1%})")
    else:
        # Resolve max_pairs from --scale if not explicitly set
        max_pairs = args.max_pairs
        if max_pairs == 0:
            scale_pairs = {0: 20_000, 1: 500_000, 2: 1_000_000, 3: 2_000_000, 4: 0}
            max_pairs = scale_pairs[args.scale]
            if max_pairs:
                print(f"Scale {args.scale}: capping at {max_pairs:,} pairs")
        train(args.lang, args.epochs, args.batch, args.lr,
              args.eval_split, max_pairs, scale=args.scale)


if __name__ == "__main__":
    main()
