"""
model_registry.py — Loads GAT and ChemBERTa checkpoints at startup and provides
a unified prediction interface.

Singleton pattern: call `get_registry()` to get the shared instance.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict

import torch

from backend.app.config import settings
from backend.app.data.featurizer import (
    smiles_to_graph,
    smiles_to_2d_coords,
    get_num_atom_features,
    get_num_bond_features,
)
from backend.app.models.gat_model import GATDDIClassifier
from backend.app.models.chemberta_model import ChemBERTaDDIClassifier, get_tokenizer
from backend.app.explainability.gat_attention import extract_gat_attention
from backend.app.explainability.chemberta_attention import extract_chemberta_attention


_registry_instance = None


class ModelRegistry:
    """
    Holds the GAT and ChemBERTa models and provides a unified predict() interface.
    Models are loaded from checkpoints on startup; missing checkpoints are skipped.
    """

    def __init__(self):
        self.device    = torch.device(settings.DEVICE)
        self.models:   Dict[str, object] = {}
        self.label_map: Dict[str, str]   = {}
        self.tokenizer = None
        self.all_pairs: Dict[tuple, int] = {}

        self._load_label_map()
        self._load_models()
        self._load_ground_truth_pairs()

    def _load_ground_truth_pairs(self):
        try:
            processed_dir = settings.DATA_DIR / "processed"
            dfs = []
            for split in ["train", "val", "test"]:
                path = processed_dir / f"ddi_{split}.pkl"
                if path.exists():
                    dfs.append(pd.read_pickle(path))
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                sa_list = df["smiles_a"].tolist()
                sb_list = df["smiles_b"].tolist()
                lbl_list = df["label"].tolist()
                for sa, sb, lbl in zip(sa_list, sb_list, lbl_list):
                    key = tuple(sorted([sa, sb]))
                    self.all_pairs[key] = int(lbl)
                print(f"  [REGISTRY] Loaded {len(self.all_pairs)} pairs for ground truth lookup.")
        except Exception as e:
            print(f"  [WARNING] Failed to load dataset for ground-truth lookup: {e}")

    # ── Label map ────────────────────────────────────────────────────────────

    def _load_label_map(self):
        path = settings.DATA_DIR / "processed" / "label_map.json"
        if path.exists():
            with open(path) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = {str(i): f"Class {i}" for i in range(settings.NUM_CLASSES)}

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        num_atom_feat = get_num_atom_features()
        num_bond_feat = get_num_bond_features()

        # ── GAT ───────────────────────────────────────────────────────────────
        gat_ckpt = settings.CHECKPOINT_DIR / "gat_ddi.pt"
        if gat_ckpt.exists():
            try:
                model = GATDDIClassifier(
                    in_channels=num_atom_feat,
                    hidden_channels=settings.GAT_HIDDEN,
                    num_heads=settings.GAT_HEADS,
                    num_layers=settings.GAT_LAYERS,
                    num_classes=settings.NUM_CLASSES,
                    dropout=settings.GAT_DROPOUT,
                    edge_dim=num_bond_feat,
                ).to(self.device)
                model.load_state_dict(
                    torch.load(gat_ckpt, map_location=self.device, weights_only=True)
                )
                model.eval()
                self.models["GAT"] = model
                print("  [REGISTRY] Loaded: GAT")
            except Exception as e:
                print(f"  [REGISTRY] Failed to load GAT: {e}")
        else:
            print(f"  [REGISTRY] GAT checkpoint not found at {gat_ckpt} (run train_gat.py first)")

        # ── ChemBERTa ─────────────────────────────────────────────────────────
        chem_ckpt = settings.CHECKPOINT_DIR / "chemberta_ddi.pt"
        if chem_ckpt.exists():
            try:
                model = ChemBERTaDDIClassifier(
                    model_name=settings.CHEMBERTA_MODEL_NAME,
                    num_classes=settings.NUM_CLASSES,
                    dropout=settings.GAT_DROPOUT,
                ).to(self.device)
                model.load_state_dict(
                    torch.load(chem_ckpt, map_location=self.device, weights_only=True),
                    strict=False,
                )
                model.eval()
                self.models["ChemBERTa"] = model
                print("  [REGISTRY] Loaded: ChemBERTa")
            except Exception as e:
                print(f"  [REGISTRY] Failed to load ChemBERTa: {e}")
        else:
            print(f"  [REGISTRY] ChemBERTa checkpoint not found at {chem_ckpt} (run train_chemberta.py first)")

        # ── Tokenizer (shared) ────────────────────────────────────────────────
        try:
            self.tokenizer = get_tokenizer(settings.CHEMBERTA_MODEL_NAME)
        except Exception:
            self.tokenizer = None

        print(f"\n  [REGISTRY] {len(self.models)} model(s) loaded.\n")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, smiles_a: str, smiles_b: str) -> dict:
        """
        Run all loaded models on a drug pair and return combined results.

        Returns:
            {
                "drug_a": { "smiles": str, "structure": {...} },
                "drug_b": { "smiles": str, "structure": {...} },
                "predictions": [
                    {
                        "model_name": str,
                        "predicted_class": int,
                        "predicted_label": str,
                        "confidence": float,
                        "probabilities": list[float],
                        "explainability": {...}
                    },
                    ...
                ]
            }
        """
        key = tuple(sorted([smiles_a, smiles_b]))
        ground_truth = self.all_pairs.get(key, None)
        
        results = {
            "drug_a": {"smiles": smiles_a, "structure": smiles_to_2d_coords(smiles_a)},
            "drug_b": {"smiles": smiles_b, "structure": smiles_to_2d_coords(smiles_b)},
            "predictions": [],
            "ground_truth": ground_truth,
            "ground_truth_label": self.label_map.get(str(ground_truth)) if ground_truth is not None else None
        }

        graph_a = smiles_to_graph(smiles_a)
        graph_b = smiles_to_graph(smiles_b)

        for model_name, model in self.models.items():
            try:
                if model_name == "GAT":
                    pred = self._predict_gat(model, graph_a, graph_b)
                else:
                    pred = self._predict_chemberta(model, smiles_a, smiles_b)

                pred["model_name"]      = model_name
                pred["predicted_label"] = self.label_map.get(
                    str(pred["prediction"]), f"Class {pred['prediction']}"
                )
                results["predictions"].append(pred)

            except Exception as e:
                results["predictions"].append({"model_name": model_name, "error": str(e)})

        return results

    def _predict_gat(self, model, graph_a, graph_b):
        if graph_a is None or graph_b is None:
            raise ValueError("Invalid SMILES — could not construct molecular graph.")
        return extract_gat_attention(model, graph_a, graph_b, self.device)

    def _predict_chemberta(self, model, smiles_a, smiles_b):
        if self.tokenizer is None:
            raise RuntimeError("ChemBERTa tokenizer not loaded.")
        return extract_chemberta_attention(
            model, self.tokenizer, smiles_a, smiles_b,
            self.device, settings.MAX_SEQ_LEN,
        )

    def get_loaded_models(self):
        """Return list of loaded model names."""
        return list(self.models.keys())


def get_registry() -> ModelRegistry:
    """Get or create the singleton ModelRegistry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
