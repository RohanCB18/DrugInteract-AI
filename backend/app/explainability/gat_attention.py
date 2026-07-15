"""
gat_attention.py — Extract and interpret GAT attention weights for explainability.

Given a drug pair prediction, extracts the attention coefficients from the
last GATConv layer, aggregates them per-atom, and returns importance scores
that can be visualized on the molecule structure.
"""

import torch
import numpy as np
from torch_geometric.data import Batch


def extract_gat_attention(model, data_a, data_b, device):
    """
    Run a forward pass through the GAT DDI classifier with attention extraction.

    Args:
        model: GATDDIClassifier instance (already on device, eval mode)
        data_a: PyG Data object for drug A
        data_b: PyG Data object for drug B
        device: torch device

    Returns:
        dict with:
          - prediction: int (class index)
          - confidence: float
          - probabilities: list of floats per class
          - attention_a: dict {atom_idx: importance_score}
          - attention_b: dict {atom_idx: importance_score}
    """
    model.eval()

    # Batch single graphs
    batch_a = Batch.from_data_list([data_a]).to(device)
    batch_b = Batch.from_data_list([data_b]).to(device)

    with torch.no_grad():
        logits, (att_a, att_b) = model(batch_a, batch_b, return_attention=True)

    probs = torch.softmax(logits, dim=1)[0]
    pred_class = probs.argmax().item()
    confidence = probs[pred_class].item()

    # Process attention weights
    importance_a = _aggregate_attention(att_a, data_a.x.size(0))
    importance_b = _aggregate_attention(att_b, data_b.x.size(0))

    return {
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": [round(p, 4) for p in probs.tolist()],
        "attention_a": importance_a,
        "attention_b": importance_b,
    }


def _aggregate_attention(attention_tuple, num_nodes):
    """
    Aggregate attention weights to per-atom importance scores.

    The GATConv returns (edge_index_with_self_loops, alpha) where alpha
    is [num_edges, num_heads]. We average over heads and sum incoming
    attention per node.
    """
    edge_index, alpha = attention_tuple

    # alpha shape: [num_edges, num_heads] or [num_edges]
    if alpha.dim() > 1:
        alpha = alpha.mean(dim=1)  # Average over heads

    alpha = alpha.cpu().numpy()
    edge_index = edge_index.cpu().numpy()

    # Sum incoming attention for each target node
    importance = np.zeros(num_nodes)
    for i in range(edge_index.shape[1]):
        target = edge_index[1, i]
        if target < num_nodes:
            importance[target] += alpha[i]

    # Normalize to [0, 1]
    if importance.max() > 0:
        importance = importance / importance.max()

    return {int(k): round(float(v), 4) for k, v in enumerate(importance)}
