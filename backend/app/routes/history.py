"""
history.py — GET /history endpoint.

Returns recent predictions from the logging database.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.database import get_db, Prediction

router = APIRouter()


@router.get("/history")
async def get_history(
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Return recent predictions, most recent first.

    Query params:
        limit: max number of predictions to return (default 50, max 500)
    """
    preds = (
        db.query(Prediction)
        .order_by(desc(Prediction.timestamp))
        .limit(limit)
        .all()
    )

    return [
        {
            "id": p.id,
            "drug_a": p.drug_a,
            "drug_b": p.drug_b,
            "smiles_a": p.smiles_a,
            "smiles_b": p.smiles_b,
            "model_name": p.model_name,
            "predicted_class": p.predicted_class,
            "predicted_label": p.predicted_label,
            "confidence": p.confidence,
            "timestamp": p.timestamp.isoformat() if p.timestamp else None,
        }
        for p in preds
    ]
