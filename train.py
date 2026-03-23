#!/usr/bin/env python3
"""Train Dilemma lemmatizer on Wiktionary form->lemma pairs.

Trains a small character-level encoder-decoder transformer (~5M params)
on Greek lemmatization. The model learns morphological patterns like
-ωσε -> -ώνω (aorist -> present stem) and generalizes to unseen forms.

Usage:
    python train.py --scale test            # 20K pairs (~15 sec, sanity check)
    python train.py --scale full            # full data (~45 min on RTX 2080, default)
    python train.py --scale 3               # legacy alias for full
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

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import Dataset, DataLoader

SEED = 42

from model import CharVocab, LemmaTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "model"


class LemmaPairDataset(Dataset):
    GENDER = {"masculine", "feminine", "neuter"}
    NUMBER = {"singular", "plural", "dual"}
    CASE = {"nominative", "genitive", "dative", "accusative", "vocative"}
    TENSE = {"present", "imperfect", "aorist", "future", "perfect", "pluperfect"}
    MOOD = {"indicative", "subjunctive", "optative", "imperative", "participle", "infinitive"}
    VOICE = {"active", "middle", "passive"}

    def __init__(self, pairs, vocab, pos_map=None, nom_map=None, verb_map=None, max_len=48):
        self.pairs = pairs
        self.vocab = vocab
        self.pos_map = pos_map
        self.nom_map = nom_map
        self.verb_map = verb_map
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

        result = {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
        }

        # POS label for multi-task learning (-1 = no label)
        if self.pos_map is not None:
            pos_idx = self.pos_map.get(p.get("pos", ""), -1)
            result["pos"] = torch.tensor(pos_idx, dtype=torch.long)

        # Morphology labels (Swaelens nominal/verbal grouping)
        tags = set(p.get("tags", []))
        if self.nom_map is not None:
            g, n, c = tags & self.GENDER, tags & self.NUMBER, tags & self.CASE
            if g and n and c:
                label = f"{next(iter(g))[:1]}.{next(iter(n))[:1]}.{next(iter(c))[:1]}"
                result["nom"] = torch.tensor(self.nom_map.get(label, -1), dtype=torch.long)
            else:
                result["nom"] = torch.tensor(-1, dtype=torch.long)
        if self.verb_map is not None:
            t, m, v = tags & self.TENSE, tags & self.MOOD, tags & self.VOICE
            if t and m and v:
                label = f"{next(iter(t))[:3]}.{next(iter(m))[:3]}.{next(iter(v))[:3]}"
                result["verb"] = torch.tensor(self.verb_map.get(label, -1), dtype=torch.long)
            else:
                result["verb"] = torch.tensor(-1, dtype=torch.long)

        return result


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

    lang="all" combines MG and AG pairs for a single model that handles
    katharevousa (which mixes AG morphology with MG vocabulary).
    """
    # Medieval Greek is folded into MG — same language, earlier stage
    if lang == "all":
        prefixes = ["mg", "med", "ag"]
    elif lang == "el":
        prefixes = ["mg", "med"]
    else:
        prefixes = [{"grc": "ag"}[lang]]

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

    # Deduplicate globally (preserve POS and tags for multi-task learning)
    seen = set()
    def dedup(pairs_list):
        result = []
        for p in pairs_list:
            key = (p["form"], p["lemma"])
            if key not in seen:
                seen.add(key)
                entry = {"form": p["form"], "lemma": p["lemma"]}
                if "pos" in p:
                    entry["pos"] = p["pos"]
                if "tags" in p:
                    entry["tags"] = p["tags"]
                result.append(entry)
        return result

    # Separate pools
    varieties = []  # always 100%: non-standard MG (including Medieval)
    ag_pool = []
    smg_pool = []

    # Medieval is part of MG — treat all Medieval pairs as varieties
    if "med" in by_source:
        med = dedup(by_source["med"])
        varieties.extend(med)
        print(f"  varieties: {len(med)} Medieval MG pairs")

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

    # GLAUx corpus pairs (morphologically tagged AG, folded into AG pool)
    glaux_path = DATA_DIR / "glaux_pairs.json"
    if glaux_path.exists() and lang in ("all", "grc"):
        with open(glaux_path, encoding="utf-8") as f:
            glaux_data = json.load(f)
        glaux = dedup(glaux_data)
        ag_pool.extend(glaux)
        print(f"  GLAUx: {len(glaux)} morphologically tagged AG pairs")

    # Oversample perfect tense forms (underrepresented in training: ~2%
    # vs 11.4% in Byzantine text per Swaelens et al. 2024/2025)
    PERFECT_TAGS = {"perfect", "pluperfect", "future-perfect"}
    perfect_pairs = [p for p in ag_pool
                     if p.get("pos") == "verb" and PERFECT_TAGS & set(p.get("tags", []))]
    if perfect_pairs:
        # Add 2 extra copies to bring perfects from ~2% to ~6%
        ag_pool.extend(perfect_pairs * 2)
        print(f"  Perfect tense oversampling: {len(perfect_pairs)} pairs x3")

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


def evaluate_heads(model, vocab, eval_pairs, device, batch_size=256,
                   pos_map=None, nom_map=None, verb_map=None):
    """Evaluate auxiliary head accuracy on held-out pairs.

    Returns dict of {head_name: (correct, total)} for each active head.
    """
    model.eval()
    stats = {}
    if pos_map:
        stats["pos"] = [0, 0]
    if nom_map:
        stats["nom"] = [0, 0]
    if verb_map:
        stats["verb"] = [0, 0]

    if not stats:
        return {}

    dataset = LemmaPairDataset(eval_pairs, vocab,
                               pos_map=pos_map, nom_map=nom_map,
                               verb_map=verb_map)

    for i in range(0, len(eval_pairs), batch_size):
        batch_indices = range(i, min(i + batch_size, len(eval_pairs)))
        batch_items = [dataset[j] for j in batch_indices]

        src = torch.stack([b["src"] for b in batch_items]).to(device)
        src_pad_mask = (src == 0)

        with torch.no_grad():
            if "pos" in stats:
                pos_logits = model.predict_pos(src, src_pad_mask)
                if pos_logits is not None:
                    pos_labels = torch.stack([b["pos"] for b in batch_items]).to(device)
                    mask = pos_labels != -1
                    if mask.any():
                        preds = pos_logits.argmax(dim=-1)
                        stats["pos"][0] += (preds[mask] == pos_labels[mask]).sum().item()
                        stats["pos"][1] += mask.sum().item()

            if "nom" in stats or "verb" in stats:
                nom_logits, verb_logits = model.predict_morph(src, src_pad_mask)

                if "nom" in stats and nom_logits is not None:
                    nom_labels = torch.stack([b["nom"] for b in batch_items]).to(device)
                    mask = nom_labels != -1
                    if mask.any():
                        preds = nom_logits.argmax(dim=-1)
                        stats["nom"][0] += (preds[mask] == nom_labels[mask]).sum().item()
                        stats["nom"][1] += mask.sum().item()

                if "verb" in stats and verb_logits is not None:
                    verb_labels = torch.stack([b["verb"] for b in batch_items]).to(device)
                    mask = verb_labels != -1
                    if mask.any():
                        preds = verb_logits.argmax(dim=-1)
                        stats["verb"][0] += (preds[mask] == verb_labels[mask]).sum().item()
                        stats["verb"][1] += mask.sum().item()

    return {k: tuple(v) for k, v in stats.items()}


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
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
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

    # Build POS tag mapping for multi-task learning
    WIKT_TO_UPOS = {
        "verb": 0, "noun": 1, "adj": 2, "adv": 3, "name": 4,
        "pron": 5, "num": 6, "prep": 7, "article": 8, "character": 9,
    }
    has_pos = sum(1 for p in train_pairs if p.get("pos") in WIKT_TO_UPOS)
    use_multitask = has_pos > len(train_pairs) * 0.3  # enable if 30%+ have POS
    num_pos_tags = len(WIKT_TO_UPOS) if use_multitask else 0
    if use_multitask:
        print(f"Multi-task learning: {has_pos}/{len(train_pairs)} pairs have POS tags")

    # Build nominal (Gender+Number+Case) and verbal (Tense+Mood+Voice) label maps
    # Following Swaelens et al. (2025) feature grouping approach
    GENDER = {"masculine", "feminine", "neuter"}
    NUMBER = {"singular", "plural", "dual"}
    CASE = {"nominative", "genitive", "dative", "accusative", "vocative"}
    TENSE = {"present", "imperfect", "aorist", "future", "perfect", "pluperfect"}
    MOOD = {"indicative", "subjunctive", "optative", "imperative", "participle", "infinitive"}
    VOICE = {"active", "middle", "passive"}

    nom_labels = {}  # "m.s.n" -> idx
    verb_labels = {}  # "pre.ind.act" -> idx
    for p in train_pairs:
        tags = set(p.get("tags", []))
        g, n, c = tags & GENDER, tags & NUMBER, tags & CASE
        if g and n and c:
            label = f"{next(iter(g))[:1]}.{next(iter(n))[:1]}.{next(iter(c))[:1]}"
            if label not in nom_labels:
                nom_labels[label] = len(nom_labels)
        t, m, v = tags & TENSE, tags & MOOD, tags & VOICE
        if t and m and v:
            label = f"{next(iter(t))[:3]}.{next(iter(m))[:3]}.{next(iter(v))[:3]}"
            if label not in verb_labels:
                verb_labels[label] = len(verb_labels)

    use_morph = len(nom_labels) > 5 and len(verb_labels) > 5
    if use_morph:
        print(f"Morphology heads: {len(nom_labels)} nominal labels, {len(verb_labels)} verbal labels")

    # Create model
    model = LemmaTransformer(vocab_size=len(vocab), num_pos_tags=num_pos_tags)
    if use_morph:
        model.nom_head = torch.nn.Linear(model.d_model, len(nom_labels)).to(device)
        model.verb_head = torch.nn.Linear(model.d_model, len(verb_labels)).to(device)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    heads = []
    if use_multitask:
        heads.append(f"POS: {num_pos_tags}")
    if use_morph:
        heads.append(f"nom: {len(nom_labels)}")
        heads.append(f"verb: {len(verb_labels)}")
    head_str = f" (heads: {', '.join(heads)})" if heads else ""
    print(f"Model: {param_count / 1e6:.1f}M parameters{head_str}")

    dataset = LemmaPairDataset(train_pairs, vocab,
                               pos_map=WIKT_TO_UPOS if use_multitask else None,
                               nom_map=nom_labels if use_morph else None,
                               verb_map=verb_labels if use_morph else None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=device == "cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
    aux_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    AUX_LOSS_WEIGHT = 0.1  # auxiliary tasks, don't dominate
    MAX_GRAD_NORM = 1.0

    # LR scheduler: linear warmup then linear decay
    total_batches = len(loader)
    num_training_steps = total_batches * epochs
    num_warmup_steps = min(500, num_training_steps // 10)

    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        return max(0.0, (num_training_steps - step) /
                   max(1, num_training_steps - num_warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nTraining for {epochs} epochs ({total_batches} batches/epoch, "
          f"warmup={num_warmup_steps} steps, grad_clip={MAX_GRAD_NORM})...",
          flush=True)
    global_step = 0
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

            # Multi-task auxiliary losses
            if use_multitask and "pos" in batch_data:
                pos_labels = batch_data["pos"].to(device)
                pos_logits = model.predict_pos(src, src_pad_mask)
                if pos_logits is not None:
                    loss = loss + AUX_LOSS_WEIGHT * aux_criterion(pos_logits, pos_labels)

            if use_morph:
                if "nom" in batch_data:
                    nom_labels_batch = batch_data["nom"].to(device)
                    nom_logits, _ = model.predict_morph(src, src_pad_mask)
                    if nom_logits is not None:
                        loss = loss + AUX_LOSS_WEIGHT * aux_criterion(nom_logits, nom_labels_batch)
                if "verb" in batch_data:
                    verb_labels_batch = batch_data["verb"].to(device)
                    _, verb_logits = model.predict_morph(src, src_pad_mask)
                    if verb_logits is not None:
                        loss = loss + AUX_LOSS_WEIGHT * aux_criterion(verb_logits, verb_labels_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count = nan_count + 1 if 'nan_count' in dir() else 1
                scheduler.step()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            batches += 1
            global_step += 1

            if batches % 100 == 0:
                elapsed = time.time() - t0
                rate = batches / elapsed
                eta = (total_batches - batches) / rate
                current_lr = scheduler.get_last_lr()[0]
                msg = (f"    [{batches}/{total_batches}] loss={total_loss/batches:.4f} "
                       f"lr={current_lr:.2e} ({rate:.1f} batch/s, ETA {eta/60:.0f}m)")
                print(msg, flush=True)
                with open(SCRIPT_DIR / "progress.log", "a") as pf:
                    pf.write(msg + "\n")

        avg_loss = total_loss / batches if batches > 0 else float('nan')
        elapsed = time.time() - t0
        skipped = nan_count if 'nan_count' in dir() else 0

        # Evaluate lemma accuracy
        accuracy, correct, total = evaluate(
            model, vocab, eval_pairs[:2000], device
        )
        nan_msg = f", {skipped} NaN batches skipped" if skipped else ""
        msg = (f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, "
               f"eval={correct}/{total} ({accuracy:.1%}), {elapsed:.0f}s{nan_msg}")
        nan_count = 0

        # Evaluate morphology head accuracy
        if use_multitask or use_morph:
            head_stats = evaluate_heads(
                model, vocab, eval_pairs[:2000], device,
                pos_map=WIKT_TO_UPOS if use_multitask else None,
                nom_map=nom_labels if use_morph else None,
                verb_map=verb_labels if use_morph else None,
            )
            head_parts = []
            for name, (hcorrect, htotal) in head_stats.items():
                if htotal > 0:
                    head_parts.append(f"{name}={hcorrect}/{htotal} ({hcorrect/htotal:.1%})")
            if head_parts:
                msg += "\n    Heads: " + ", ".join(head_parts)

        print(msg, flush=True)
        with open(SCRIPT_DIR / "progress.log", "a") as pf:
            pf.write(msg + "\n")

    # Save model
    lang_dir = {"el": "el", "grc": "grc", "all": "combined"}[lang]
    if scale == "test":
        lang_dir = f"{lang_dir}-test"
    elif scale and scale not in ("full",):
        # Legacy: --scale 1/2/3 -> -s1/-s2/-s3 dirs
        lang_dir = f"{lang_dir}-s{scale}"
    out_dir = MODEL_DIR / lang_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build label maps for inference (index -> label string)
    pos_labels_inv = {v: k for k, v in WIKT_TO_UPOS.items()} if use_multitask else {}
    nom_labels_inv = {v: k for k, v in nom_labels.items()} if use_morph else {}
    verb_labels_inv = {v: k for k, v in verb_labels.items()} if use_morph else {}

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab.state_dict(),
        "config": {
            "vocab_size": len(vocab),
            "d_model": 256,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "num_pos_tags": num_pos_tags,
        },
        "head_config": {
            "num_nom_labels": len(nom_labels) if use_morph else 0,
            "num_verb_labels": len(verb_labels) if use_morph else 0,
            "pos_labels": pos_labels_inv,
            "nom_labels": nom_labels_inv,
            "verb_labels": verb_labels_inv,
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
    parser.add_argument("--lang", type=str, default="all",
                        choices=["all", "el", "grc"],
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
    parser.add_argument("--scale", type=str, default="full",
                        choices=["test", "full", "1", "2", "3"],
                        help="Training scale: test=20K pairs (~15 sec), "
                             "full=all data (~45 min on RTX 2080). "
                             "Legacy 1/2/3 still accepted for compatibility.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate existing model without training")
    args = parser.parse_args()

    if args.eval_only:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        lang_dir = {"el": "el", "grc": "grc", "all": "combined"}[args.lang]
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
        # Normalize legacy scale numbers to names
        scale = {"1": "test", "2": "full", "3": "full"}.get(args.scale, args.scale)
        if max_pairs == 0:
            scale_pairs = {"test": 20_000, "full": 0}
            max_pairs = scale_pairs[scale]
            if max_pairs:
                print(f"Scale {scale}: capping at {max_pairs:,} pairs")
        train(args.lang, args.epochs, args.batch, args.lr,
              args.eval_split, max_pairs, scale=scale)


if __name__ == "__main__":
    main()
