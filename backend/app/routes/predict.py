"""
predict.py — POST /predict endpoint.

Accepts a pair of drugs (SMILES), runs all loaded model variants,
returns predictions with confidence and explainability data,
and logs each prediction to the database.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database import get_db, Prediction
from backend.app.models.model_registry import get_registry

router = APIRouter()


class PredictRequest(BaseModel):
    """Request body for /predict."""
    drug_a: str  # SMILES string or drug identifier
    drug_b: str
    # Optional: future support for drug name → SMILES lookup
    drug_a_name: Optional[str] = None
    drug_b_name: Optional[str] = None


class PredictResponse(BaseModel):
    """Response body for /predict."""
    drug_a: dict
    drug_b: dict
    predictions: list
    ground_truth: Optional[int] = None
    ground_truth_label: Optional[str] = None


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Predict drug-drug interaction for a given pair.

    Runs all loaded model variants (GAT-A, GAT-B, ChemBERTa-A, ChemBERTa-B)
    and returns side-by-side predictions with explainability data.
    """
    registry = get_registry()

    if not registry.get_loaded_models():
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Train models first.",
        )

    smiles_a = request.drug_a.strip()
    smiles_b = request.drug_b.strip()

    if not smiles_a or not smiles_b:
        raise HTTPException(status_code=400, detail="Both drug SMILES are required.")

    # Run predictions
    results = registry.predict(smiles_a, smiles_b)

    # Log each successful prediction to DB
    for pred in results["predictions"]:
        if "error" not in pred:
            try:
                db_pred = Prediction(
                    drug_a=request.drug_a_name or smiles_a[:100],
                    drug_b=request.drug_b_name or smiles_b[:100],
                    smiles_a=smiles_a,
                    smiles_b=smiles_b,
                    model_name=pred["model_name"],
                    predicted_class=pred["prediction"],
                    predicted_label=pred.get("predicted_label", ""),
                    confidence=pred["confidence"],
                    timestamp=datetime.now(timezone.utc),
                )
                db.add(db_pred)
            except Exception:
                pass  # Don't fail the prediction if logging fails

    try:
        db.commit()
    except Exception:
        db.rollback()

    return results
