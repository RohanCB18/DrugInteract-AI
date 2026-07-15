"""
compare.py — GET /compare endpoint.

Returns the saved evaluation metrics (from evaluate.py) for the
comparison dashboard.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.config import settings

router = APIRouter()


@router.get("/compare")
async def get_comparison():
    """
    Return the evaluation metrics for all trained model variants.

    This data is produced by running:
        python -m backend.app.training.evaluate
    """
    results_path = settings.CHECKPOINT_DIR / "evaluation_results.json"

    if not results_path.exists():
        # Return empty results instead of failing — dashboard can show "no data"
        return {
            "models": [],
            "num_classes": settings.NUM_CLASSES,
            "test_size": 0,
            "message": "No evaluation results found. Run the evaluation script first.",
        }

    with open(results_path) as f:
        data = json.load(f)

    return data


@router.get("/labels")
async def get_labels():
    """Return the interaction label map (class index → human-readable label)."""
    path = settings.DATA_DIR / "processed" / "label_map.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {str(i): f"Class {i}" for i in range(settings.NUM_CLASSES)}


@router.get("/drugs")
async def get_drug_list():
    """Return the list of known drugs (for frontend autocomplete)."""
    path = settings.DATA_DIR / "processed" / "drug_list.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []
