"""
chemberta_model.py — ChemBERTa-based Drug-Drug Interaction classifier.

Architecture:
  - Loads seyonec/ChemBERTa-zinc-base-v1 (RoBERTa for SMILES)
  - Encodes SMILES_A and SMILES_B separately via the transformer
  - Extracts [CLS] embeddings for each → concatenate → MLP → num_classes
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class ChemBERTaEncoder(nn.Module):
    """
    Wraps a ChemBERTa (RoBERTa) model to produce [CLS] embeddings for SMILES.
    """

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
                 freeze_base: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        try:
            self.transformer = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        except TypeError:
            self.transformer = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, output_attentions=False):
        """
        Returns:
            cls_embedding: [batch_size, hidden_size]
            (optionally) attentions: tuple of attention tensors from all layers
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        if output_attentions:
            return cls_embedding, outputs.attentions
        return cls_embedding


class ChemBERTaDDIClassifier(nn.Module):
    """
    Drug-Drug Interaction classifier using ChemBERTa.

    Encodes two SMILES strings separately, concatenates their [CLS] embeddings,
    and classifies through an MLP.
    """

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = ChemBERTaEncoder(model_name)
        hidden = self.encoder.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, input_ids_a, attention_mask_a,
                input_ids_b, attention_mask_b,
                output_attentions=False):
        """
        Args:
            input_ids_a, attention_mask_a: tokenized SMILES for drug A
            input_ids_b, attention_mask_b: tokenized SMILES for drug B
            output_attentions: if True, also return attention weights

        Returns:
            logits: [batch_size, num_classes]
            (optionally) (attentions_a, attentions_b)
        """
        if output_attentions:
            emb_a, att_a = self.encoder(
                input_ids_a, attention_mask_a, output_attentions=True
            )
            emb_b, att_b = self.encoder(
                input_ids_b, attention_mask_b, output_attentions=True
            )
        else:
            emb_a = self.encoder(input_ids_a, attention_mask_a)
            emb_b = self.encoder(input_ids_b, attention_mask_b)

        combined = torch.cat([emb_a, emb_b], dim=-1)
        logits = self.classifier(combined)

        if output_attentions:
            return logits, (att_a, att_b)
        return logits


def get_tokenizer(model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
    """Load the ChemBERTa tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)
