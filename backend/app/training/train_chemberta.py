"""
train_chemberta.py — Fine-tune ChemBERTa-zinc-base-v1 for DDI classification.

Downloads pretrained weights from HuggingFace automatically on first run.
Pre-tokenizes the dataset once at startup and caches it to disk for fast training.

Usage:
    python -m backend.app.training.train_chemberta
    python -m backend.app.training.train_chemberta --smoke-test   # 3 epochs, 200 samples
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.app.config import settings
from backend.app.data.dataset import DDITextDataset
from backend.app.models.chemberta_model import ChemBERTaDDIClassifier, get_tokenizer


def compute_class_weights(labels, num_classes):
    """Compute inverse-frequency class weights."""
    counts = Counter(labels)
    total = sum(counts.values())
    return torch.tensor(
        [total / (num_classes * max(counts.get(c, 1), 1)) for c in range(num_classes)],
        dtype=torch.float,
    )


def collate_pretokenized(batch):
    """Collate pre-tokenized dictionary tensors into batch formats."""
    input_ids_a = torch.stack([item["input_ids_a"] for item in batch])
    attention_mask_a = torch.stack([item["attention_mask_a"] for item in batch])
    input_ids_b = torch.stack([item["input_ids_b"] for item in batch])
    attention_mask_b = torch.stack([item["attention_mask_b"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return {
        "input_ids_a":      input_ids_a,
        "attention_mask_a": attention_mask_a,
        "input_ids_b":      input_ids_b,
        "attention_mask_b": attention_mask_b,
        "labels":           labels,
    }


def train_chemberta(epochs: int = None, batch_size: int = None, lr: float = None, smoke_test: bool = False):
    epochs     = epochs     or settings.CHEMBERTA_EPOCHS
    # Reduced batch size to 32 to fit inside 4GB GPU VRAM and prevent slow System RAM swapping
    batch_size = batch_size or 32
    lr         = lr         or settings.CHEMBERTA_LR
    device     = torch.device(settings.DEVICE)

    if smoke_test:
        epochs = 3
        batch_size = 32
        print("[SMOKE TEST] Running 3 epochs on 200 samples only.")

    # ── Load data ──────────────────────────────────────────────────────────
    processed_dir = settings.DATA_DIR / "processed"
    train_df = pd.read_pickle(processed_dir / "ddi_train.pkl")
    val_df   = pd.read_pickle(processed_dir / "ddi_val.pkl")

    if smoke_test:
        train_df = train_df.head(200)
        val_df   = val_df.head(100)

    print(f"[TRAIN ChemBERTa] Train: {len(train_df)} pairs, Val: {len(val_df)} pairs")

    tokenizer = get_tokenizer(settings.CHEMBERTA_MODEL_NAME)

    # Pre-tokenize or load pre-tokenized cache
    train_cache = processed_dir / f"train_chemberta_tokenized{'_smoke' if smoke_test else ''}.pkl"
    val_cache   = processed_dir / f"val_chemberta_tokenized{'_smoke' if smoke_test else ''}.pkl"

    train_ds = DDITextDataset(
        train_df["smiles_a"].tolist(),
        train_df["smiles_b"].tolist(),
        train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_len=settings.MAX_SEQ_LEN,
        cache_path=train_cache,
    )
    val_ds = DDITextDataset(
        val_df["smiles_a"].tolist(),
        val_df["smiles_b"].tolist(),
        val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_len=settings.MAX_SEQ_LEN,
        cache_path=val_cache,
    )

    collate_fn = collate_pretokenized

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ── Model — loads pretrained HuggingFace weights automatically ─────────
    print(f"[TRAIN ChemBERTa] Loading pretrained weights: {settings.CHEMBERTA_MODEL_NAME}")
    model = ChemBERTaDDIClassifier(
        model_name=settings.CHEMBERTA_MODEL_NAME,
        num_classes=settings.NUM_CLASSES,
        dropout=settings.GAT_DROPOUT,
    ).to(device)

    # Freeze the transformer encoder layers (Feature Extraction Mode)
    print("[TRAIN ChemBERTa] Freezing base encoder layers (running in fast feature extraction mode)...")
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Re-compute labels list after filtering in dataset
    ds_labels = train_ds.labels.tolist()
    class_weights = compute_class_weights(ds_labels, settings.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only train the classifier head parameters (saves 70% VRAM/RAM overhead)
    optimizer = AdamW(model.classifier.parameters(), lr=lr * 10, weight_decay=1e-2)

    # Linear warmup then cosine decay
    warmup_steps = min(len(train_loader) * 2, 500)
    total_steps  = len(train_loader) * epochs
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1)),
        ],
        milestones=[warmup_steps],
    )

    print(f"\n[TRAIN ChemBERTa] Device: {device}")
    print(f"[TRAIN ChemBERTa] Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"[TRAIN ChemBERTa] Params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── Training loop ──────────────────────────────────────────────────────
    import json
    save_path = settings.CHECKPOINT_DIR / "chemberta_ddi.pt"
    meta_path = settings.CHECKPOINT_DIR / "chemberta_ddi_meta.json"
    
    start_epoch = 1
    best_f1 = 0.0
    patience_counter = 0

    if save_path.exists() and not smoke_test:
        try:
            print(f"\n[TRAIN ChemBERTa] Found existing checkpoint at {save_path}. Loading weights to resume...")
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                start_epoch = meta.get("epoch", 1) + 1
                best_f1 = meta.get("best_f1", 0.0)
                print(f"[TRAIN ChemBERTa] Resuming training from Epoch {start_epoch} (Best Val F1: {best_f1:.4f})")
            else:
                start_epoch = 2
                best_f1 = 0.5726  # Epoch 1 Val F1 score
                print(f"[TRAIN ChemBERTa] Resuming training from Epoch 2...")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint to resume: {e}. Starting from scratch.")

    from tqdm import tqdm

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs} [Train]")
        for batch in pbar:
            logits = model(
                batch["input_ids_a"].to(device),
                batch["attention_mask_a"].to(device),
                batch["input_ids_b"].to(device),
                batch["attention_mask_b"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        avg_loss = total_loss / max(num_batches, 1)

        # Validate
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{epochs} [Val]"):
                logits = model(
                    batch["input_ids_a"].to(device),
                    batch["attention_mask_a"].to(device),
                    batch["input_ids_b"].to(device),
                    batch["attention_mask_b"].to(device),
                )
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_true.extend(batch["labels"].tolist())

        val_acc = accuracy_score(all_true, all_preds)
        val_f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            # Save metadata for resumption
            with open(meta_path, "w") as f:
                json.dump({"epoch": epoch, "best_f1": best_f1}, f, indent=2)
            marker = " [SAVED]"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}{marker}"
        )

        if patience_counter >= settings.PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={settings.PATIENCE})")
            break

    print(f"\n[TRAIN ChemBERTa] Done. Best Val F1: {best_f1:.4f}")
    print(f"[TRAIN ChemBERTa] Checkpoint saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    train_chemberta(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke_test=args.smoke_test,
    )
