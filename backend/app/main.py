"""
main.py — FastAPI application entry point.

Starts the backend server with:
  - CORS middleware (for React frontend)
  - Model loading on startup
  - Database table creation on startup
  - Routes: /predict, /compare, /labels, /drugs, /history
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.database import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # ── Startup ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Drug Interaction Prediction — API Server")
    print("=" * 60)

    # Create DB tables
    print("\n  [STARTUP] Creating database tables...")
    create_tables()

    # Load models (lazy import to avoid loading at module level)
    print("  [STARTUP] Loading model checkpoints...")
    from backend.app.models.model_registry import get_registry
    registry = get_registry()

    print(f"  [STARTUP] Models loaded: {registry.get_loaded_models()}")
    print(f"  [STARTUP] Device: {settings.DEVICE}")
    print(f"  [STARTUP] Database: {settings.DATABASE_URL}")
    print("=" * 60 + "\n")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    print("\n  [SHUTDOWN] Server stopping...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Drug Interaction Prediction API",
    description=(
        "Compare GAT (graph-based) and ChemBERTa (text-based) models "
        "for drug-drug interaction prediction, with self-supervised "
        "pretraining variants."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
from backend.app.routes.predict import router as predict_router
from backend.app.routes.compare import router as compare_router
from backend.app.routes.history import router as history_router

app.include_router(predict_router, prefix="/api", tags=["Prediction"])
app.include_router(compare_router, prefix="/api", tags=["Comparison"])
app.include_router(history_router, prefix="/api", tags=["History"])


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    from backend.app.models.model_registry import get_registry
    registry = get_registry()
    return {
        "status": "ok",
        "models_loaded": registry.get_loaded_models(),
        "device": settings.DEVICE,
    }
