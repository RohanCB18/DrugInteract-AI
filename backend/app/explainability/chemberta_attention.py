"""
chemberta_attention.py — Extract and interpret ChemBERTa attention weights.

Extracts attention from the last transformer layer, averages across heads,
and maps importance scores back to SMILES token positions.
"""

import torch
import numpy as np


def extract_chemberta_attention(model, tokenizer, smiles_a, smiles_b, device, max_len=128):
    """
    Run a forward pass through ChemBERTa DDI classifier with attention extraction.

    Args:
        model: ChemBERTaDDIClassifier instance (on device, eval mode)
        tokenizer: ChemBERTa tokenizer
        smiles_a: SMILES string for drug A
        smiles_b: SMILES string for drug B
        device: torch device
        max_len: max token length

    Returns:
        dict with:
          - prediction: int (class index)
          - confidence: float
          - probabilities: list of floats per class
          - attention_a: list of {token: str, importance: float}
          - attention_b: list of {token: str, importance: float}
    """
    model.eval()

    # Tokenize
    enc_a = tokenizer(
        smiles_a, padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    enc_b = tokenizer(
        smiles_b, padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt",
    )

    input_ids_a = enc_a["input_ids"].to(device)
    attn_mask_a = enc_a["attention_mask"].to(device)
    input_ids_b = enc_b["input_ids"].to(device)
    attn_mask_b = enc_b["attention_mask"].to(device)

    with torch.no_grad():
        logits, (attentions_a, attentions_b) = model(
            input_ids_a, attn_mask_a,
            input_ids_b, attn_mask_b,
            output_attentions=True,
        )

    probs = torch.softmax(logits, dim=1)[0]
    pred_class = probs.argmax().item()
    confidence = probs[pred_class].item()

    # Process attention for drug A
    tokens_a = tokenizer.convert_ids_to_tokens(input_ids_a[0].cpu())
    importance_a = _process_attention(attentions_a, attn_mask_a[0], tokens_a)

    # Process attention for drug B
    tokens_b = tokenizer.convert_ids_to_tokens(input_ids_b[0].cpu())
    importance_b = _process_attention(attentions_b, attn_mask_b[0], tokens_b)

    return {
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": [round(p, 4) for p in probs.tolist()],
        "attention_a": importance_a,
        "attention_b": importance_b,
    }


def _process_attention(attentions, attention_mask, tokens):
    """
    Process transformer attention weights into per-token importance.

    Takes the last layer's attention, averages over heads, and extracts
    the [CLS] token's attention distribution (how much [CLS] attends to
    each token) as importance scores.
    """
    if attentions is None or len(attentions) == 0:
        # Fallback: return uniform low importance if attentions are not available
        result = []
        for i, token in enumerate(tokens):
            if token in ("<s>", "</s>", "<pad>", "<mask>"):
                continue
            if attention_mask[i] == 0:
                continue
            result.append({
                "token_idx": i,
                "token": token,
                "importance": 0.1,
            })
        return result

    # attentions is a tuple of tensors, one per layer
    # Each: [batch_size, num_heads, seq_len, seq_len]
    last_layer_att = attentions[-1][0]  # [num_heads, seq_len, seq_len]

    # Average over heads
    avg_att = last_layer_att.mean(dim=0)  # [seq_len, seq_len]

    # [CLS] token attention: how much CLS attends to each position
    cls_attention = avg_att[0].cpu().numpy()  # [seq_len]

    # Mask out padding
    mask = attention_mask.cpu().numpy()
    cls_attention = cls_attention * mask

    # Normalize to [0, 1]
    if cls_attention.max() > 0:
        cls_attention = cls_attention / cls_attention.max()

    # Build result: skip special tokens and padding
    result = []
    for i, (token, imp) in enumerate(zip(tokens, cls_attention)):
        # Skip padding, [CLS], [SEP], etc.
        if token in ("<s>", "</s>", "<pad>", "<mask>"):
            continue
        if mask[i] == 0:
            continue
        result.append({
            "token_idx": i,
            "token": token,
            "importance": round(float(imp), 4),
        })

    return result
