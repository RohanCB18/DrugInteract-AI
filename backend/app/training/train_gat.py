"""
train_gat.py — Train the GAT DDI classifier from scratch on DrugBank DDI pairs.

Usage:
    python -m backend.app.training.train_gat
    python -m backend.app.training.train_gat --smoke-test   # 3 epochs, 300 samples
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.app.config import settings
from backend.app.data.featurizer import get_num_atom_features, get_num_bond_features
from backend.app.data.dataset import DDIGraphDataset, ddi_graph_collate
from backend.app.models.gat_model import GATDDIClassifier


def compute_class_weights(labels, num_classes):
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (num_classes * max(counts.get(c, 1), 1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def train_gat(epochs: int = None, batch_size: int = None, lr: float = None, smoke_test: bool = False):
    epochs = epochs or settings.GAT_EPOCHS
    batch_size = batch_size or settings.BATCH_SIZE
    lr = lr or settings.LEARNING_RATE
    device = torch.device(settings.DEVICE)

    if smoke_test:
        epochs = 3
        print("[SMOKE TEST] Running 3 epochs on 300 samples only.")

    # ── Load data ──────────────────────────────────────────────────────────
    processed_dir = settings.DATA_DIR / "processed"
    train_df = pd.read_pickle(processed_dir / "ddi_train.pkl")
    val_df   = pd.read_pickle(processed_dir / "ddi_val.pkl")

    if smoke_test:
        train_df = train_df.head(300)
        val_df   = val_df.head(100)

    print(f"[TRAIN GAT] Train: {len(train_df)} pairs, Val: {len(val_df)} pairs")

    train_ds = DDIGraphDataset(
        train_df["smiles_a"].tolist(),
        train_df["smiles_b"].tolist(),
        train_df["label"].tolist(),
        cache_path=processed_dir / f"train_graphs{'_smoke' if smoke_test else ''}.pkl",
    )
    val_ds = DDIGraphDataset(
        val_df["smiles_a"].tolist(),
        val_df["smiles_b"].tolist(),
        val_df["label"].tolist(),
        cache_path=processed_dir / f"val_graphs{'_smoke' if smoke_test else ''}.pkl",
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=ddi_graph_collate, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=ddi_graph_collate, num_workers=0,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = GATDDIClassifier(
        in_channels=get_num_atom_features(),
        hidden_channels=settings.GAT_HIDDEN,
        num_heads=settings.GAT_HEADS,
        num_layers=settings.GAT_LAYERS,
        num_classes=settings.NUM_CLASSES,
        dropout=settings.GAT_DROPOUT,
        edge_dim=get_num_bond_features(),
    ).to(device)

    class_weights = compute_class_weights(train_df["label"].tolist(), settings.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n[TRAIN GAT] Device: {device}")
    print(f"[TRAIN GAT] Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"[TRAIN GAT] Params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── Training loop ──────────────────────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0
    save_path = settings.CHECKPOINT_DIR / "gat_ddi.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        from tqdm import tqdm

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs} [Train]")
        for batch_a, batch_b, labels in pbar:
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            labels  = labels.to(device)

            logits = model(batch_a, batch_b)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Validate
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch_a, batch_b, labels in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{epochs} [Val]"):
                logits = model(batch_a.to(device), batch_b.to(device))
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_true.extend(labels.tolist())

        val_acc = accuracy_score(all_true, all_preds)
        val_f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            marker = " [SAVED]"
        else:
            patience_counter += 1
            marker = ""

        if epoch % 5 == 0 or epoch == 1 or marker:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}{marker}"
            )

        if patience_counter >= settings.PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={settings.PATIENCE})")
            break

    print(f"\n[TRAIN GAT] Done. Best Val F1: {best_f1:.4f}")
    print(f"[TRAIN GAT] Checkpoint saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    train_gat(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke_test=args.smoke_test,
    )
