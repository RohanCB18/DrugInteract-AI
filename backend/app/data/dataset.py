"""
dataset.py — PyTorch / PyG dataset classes for Drug-Drug Interaction prediction.

Provides:
  - DDIGraphDataset   : pairs of molecular graphs  (for GAT)
  - DDITextDataset    : pairs of SMILES strings     (for ChemBERTa)
  - UnlabeledGraphs   : single molecular graphs     (for GAT contrastive pretraining)
  - UnlabeledSMILES   : single SMILES strings        (for ChemBERTa MLM pretraining)
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .featurizer import smiles_to_graph


# ══════════════════════════════════════════════════════════════════════════════
#  Labeled DDI datasets
# ══════════════════════════════════════════════════════════════════════════════

class DDIGraphDataset(Dataset):
    """
    Each item is (graph_A, graph_B, label).

    Precomputes all graphs on first load and caches to disk.
    """

    def __init__(self, smiles_a: List[str], smiles_b: List[str],
                 labels: List[int], cache_path: Optional[Path] = None):
        super().__init__()
        self.labels = labels
        self.cache_path = cache_path

        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.graphs_a = cached["graphs_a"]
            self.graphs_b = cached["graphs_b"]
            self.valid_idx = cached["valid_idx"]
        else:
            self.graphs_a = []
            self.graphs_b = []
            self.valid_idx = []
            for i, (sa, sb) in enumerate(zip(smiles_a, smiles_b)):
                ga = smiles_to_graph(sa)
                gb = smiles_to_graph(sb)
                if ga is not None and gb is not None:
                    self.graphs_a.append(ga)
                    self.graphs_b.append(gb)
                    self.valid_idx.append(i)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "graphs_a": self.graphs_a,
                        "graphs_b": self.graphs_b,
                        "valid_idx": self.valid_idx,
                    }, f)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        label = self.labels[self.valid_idx[idx]]
        return self.graphs_a[idx], self.graphs_b[idx], label


class DDITextDataset(Dataset):
    """
    Pre-tokenized dataset for ChemBERTa DDI prediction.
    Tokenizes SMILES in bulk once at startup and caches the tensors to disk.
    """

    def __init__(self, smiles_a: List[str], smiles_b: List[str], labels: List[int],
                 tokenizer=None, max_len: int = 128, cache_path: Optional[Path] = None):
        super().__init__()
        import pickle

        if cache_path and cache_path.exists():
            print(f"  [DATA] Loading pre-tokenized text dataset from {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.input_ids_a = data["input_ids_a"]
            self.attention_mask_a = data["attention_mask_a"]
            self.input_ids_b = data["input_ids_b"]
            self.attention_mask_b = data["attention_mask_b"]
            self.labels = data["labels"]
        else:
            # Filter non-empty smiles pairs
            valid_a, valid_b, valid_lbl = [], [], []
            for sa, sb, lbl in zip(smiles_a, smiles_b, labels):
                if sa and isinstance(sa, str) and sb and isinstance(sb, str):
                    valid_a.append(sa)
                    valid_b.append(sb)
                    valid_lbl.append(lbl)

            if tokenizer is not None:
                print(f"  [DATA] Pre-tokenizing {len(valid_a)} smiles pairs (runs once)...")
                enc_a = tokenizer(valid_a, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                enc_b = tokenizer(valid_b, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                self.input_ids_a = enc_a["input_ids"]
                self.attention_mask_a = enc_a["attention_mask"]
                self.input_ids_b = enc_b["input_ids"]
                self.attention_mask_b = enc_b["attention_mask"]
            else:
                self.input_ids_a = valid_a
                self.attention_mask_a = None
                self.input_ids_b = valid_b
                self.attention_mask_b = None

            self.labels = torch.tensor(valid_lbl, dtype=torch.long)

            if cache_path and tokenizer is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  [DATA] Saving pre-tokenized dataset cache to {cache_path}")
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "input_ids_a": self.input_ids_a,
                        "attention_mask_a": self.attention_mask_a,
                        "input_ids_b": self.input_ids_b,
                        "attention_mask_b": self.attention_mask_b,
                        "labels": self.labels,
                    }, f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.attention_mask_a is not None:
            return {
                "input_ids_a": self.input_ids_a[idx],
                "attention_mask_a": self.attention_mask_a[idx],
                "input_ids_b": self.input_ids_b[idx],
                "attention_mask_b": self.attention_mask_b[idx],
                "label": self.labels[idx],
            }
        else:
            return self.input_ids_a[idx], self.input_ids_b[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  Unlabeled datasets (for self-supervised pretraining)
# ══════════════════════════════════════════════════════════════════════════════

class UnlabeledGraphDataset(Dataset):
    """Single molecular graphs from ZINC-250K (for GAT contrastive pretraining)."""

    def __init__(self, smiles_list: List[str], cache_path: Optional[Path] = None):
        super().__init__()
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                self.graphs = pickle.load(f)
        else:
            self.graphs = []
            for s in smiles_list:
                g = smiles_to_graph(s)
                if g is not None:
                    self.graphs.append(g)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(self.graphs, f)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class UnlabeledSMILESDataset(Dataset):
    """Raw SMILES strings from ZINC-250K (for ChemBERTa MLM pretraining)."""

    def __init__(self, smiles_list: List[str]):
        super().__init__()
        self.smiles = [s for s in smiles_list if s and len(s) > 0]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  Custom collate for graph pairs (needed because PyG Data objects can't go
#  through the default collate)
# ══════════════════════════════════════════════════════════════════════════════

def ddi_graph_collate(batch):
    """
    Collate a list of (graph_a, graph_b, label) into batched format.

    Returns (batch_a, batch_b, labels) where batch_a/batch_b are
    torch_geometric Batch objects.
    """
    from torch_geometric.data import Batch

    graphs_a, graphs_b, labels = zip(*batch)
    batch_a = Batch.from_data_list(list(graphs_a))
    batch_b = Batch.from_data_list(list(graphs_b))
    labels_t = torch.tensor(list(labels), dtype=torch.long)
    return batch_a, batch_b, labels_t
