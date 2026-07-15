"""
evaluate.py — Evaluate GAT and ChemBERTa on the held-out test set.

Pre-tokenizes test SMILES for ChemBERTa to run in seconds.

Usage:
    python -m backend.app.training.evaluate

Evaluates both trained models on the same test split and saves:
  - evaluation_results.json  (consumed by the /compare API endpoint)
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.app.config import settings
from backend.app.data.featurizer import get_num_atom_features, get_num_bond_features
from backend.app.data.dataset import DDIGraphDataset, DDITextDataset, ddi_graph_collate
from backend.app.models.gat_model import GATDDIClassifier
from backend.app.models.chemberta_model import ChemBERTaDDIClassifier, get_tokenizer


# ── Helper: compute metrics ──────────────────────────────────────────────────

def _compute_metrics(model_name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(settings.NUM_CLASSES)))

    label_map_path = settings.DATA_DIR / "processed" / "label_map.json"
    label_map = json.load(open(label_map_path)) if label_map_path.exists() else {str(i): f"Class {i}" for i in range(settings.NUM_CLASSES)}

    result = {
        "model_name":        model_name,
        "accuracy":          round(acc,  4),
        "precision_macro":   round(prec, 4),
        "recall_macro":      round(rec,  4),
        "f1_macro":          round(f1,   4),
        "per_class_f1":      {label_map.get(str(i), f"Class {i}"): round(float(v), 4) for i, v in enumerate(per_class_f1)},
        "confusion_matrix":  cm.tolist(),
        "num_test_samples":  len(y_true),
    }

    print(f"\n  {model_name}:")
    print(f"    Accuracy:   {acc:.4f}")
    print(f"    Precision:  {prec:.4f}")
    print(f"    Recall:     {rec:.4f}")
    print(f"    F1 (macro): {f1:.4f}")
    return result


# ── GAT evaluation ───────────────────────────────────────────────────────────

def evaluate_gat(test_df: pd.DataFrame, device: torch.device):
    ckpt = settings.CHECKPOINT_DIR / "gat_ddi.pt"
    if not ckpt.exists():
        print(f"  [SKIP] GAT: checkpoint not found at {ckpt}")
        return None

    model = GATDDIClassifier(
        in_channels=get_num_atom_features(),
        hidden_channels=settings.GAT_HIDDEN,
        num_heads=settings.GAT_HEADS,
        num_layers=settings.GAT_LAYERS,
        num_classes=settings.NUM_CLASSES,
        dropout=settings.GAT_DROPOUT,
        edge_dim=get_num_bond_features(),
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    processed_dir = settings.DATA_DIR / "processed"
    ds = DDIGraphDataset(
        test_df["smiles_a"].tolist(),
        test_df["smiles_b"].tolist(),
        test_df["label"].tolist(),
        cache_path=processed_dir / "test_graphs.pkl",
    )
    loader = DataLoader(ds, batch_size=settings.BATCH_SIZE, shuffle=False, collate_fn=ddi_graph_collate, num_workers=0)

    all_preds, all_true = [], []
    with torch.no_grad():
        for batch_a, batch_b, labels in loader:
            logits = model(batch_a.to(device), batch_b.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_true.extend(labels.tolist())

    return _compute_metrics("GAT", all_true, all_preds)


# ── ChemBERTa evaluation ──────────────────────────────────────────────────────

def collate_pretokenized(batch):
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


def evaluate_chemberta(test_df: pd.DataFrame, device: torch.device):
    ckpt = settings.CHECKPOINT_DIR / "chemberta_ddi.pt"
    if not ckpt.exists():
        print(f"  [SKIP] ChemBERTa: checkpoint not found at {ckpt}")
        return None

    model = ChemBERTaDDIClassifier(
        model_name=settings.CHEMBERTA_MODEL_NAME,
        num_classes=settings.NUM_CLASSES,
        dropout=settings.GAT_DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
    model.eval()

    tokenizer = get_tokenizer(settings.CHEMBERTA_MODEL_NAME)
    processed_dir = settings.DATA_DIR / "processed"
    test_cache = processed_dir / "test_chemberta_tokenized.pkl"

    ds = DDITextDataset(
        test_df["smiles_a"].tolist(),
        test_df["smiles_b"].tolist(),
        test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_len=settings.MAX_SEQ_LEN,
        cache_path=test_cache,
    )

    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_pretokenized, num_workers=0)

    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids_a"].to(device),
                batch["attention_mask_a"].to(device),
                batch["input_ids_b"].to(device),
                batch["attention_mask_b"].to(device),
            )
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_true.extend(batch["labels"].tolist())

    return _compute_metrics("ChemBERTa", all_true, all_preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_all():
    device = torch.device(settings.DEVICE)
    processed_dir = settings.DATA_DIR / "processed"

    test_path = processed_dir / "ddi_test.pkl"
    if not test_path.exists():
        print(f"Error: {test_path} not found. Run prepare_data.py first.")
        return

    test_df = pd.read_pickle(test_path)
    print(f"\n{'='*60}")
    print(f"  Evaluating models on {len(test_df)} test pairs")
    print(f"{'='*60}")

    results = []
    for fn in [evaluate_gat, evaluate_chemberta]:
        r = fn(test_df, device)
        if r:
            results.append(r)

    output = {"models": results, "num_classes": settings.NUM_CLASSES, "test_size": len(test_df)}

    out_path = settings.CHECKPOINT_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Evaluation complete. Results saved to {out_path}")
    print(f"{'='*60}")

    if results:
        print(f"\n  {'Model':<20} {'Accuracy':<10} {'F1 (macro)':<12}")
        print(f"  {'-'*42}")
        for r in results:
            print(f"  {r['model_name']:<20} {r['accuracy']:<10} {r['f1_macro']:<12}")

    return output


if __name__ == "__main__":
    evaluate_all()
