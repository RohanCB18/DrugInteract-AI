import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # -- Paths --------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data_cache"
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"

    # -- Database -----------------------------------------------------------
    DATABASE_URL: str = f"sqlite:///{Path(__file__).resolve().parent.parent / 'predictions.db'}"

    # -- Model hyper-params ------------------------------------------------
    NUM_CLASSES: int = 5
    GAT_HIDDEN: int = 128
    GAT_HEADS: int = 8
    GAT_LAYERS: int = 3
    GAT_DROPOUT: float = 0.2
    CHEMBERTA_MODEL_NAME: str = "seyonec/ChemBERTa-zinc-base-v1"
    MAX_SEQ_LEN: int = 128

    # -- Training ----------------------------------------------------------
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    CHEMBERTA_LR: float = 2e-5
    GAT_EPOCHS: int = 25
    CHEMBERTA_EPOCHS: int = 8
    PATIENCE: int = 5

    # -- Device ------------------------------------------------------------
    DEVICE: str = "cuda" if os.environ.get("FORCE_CPU") is None and __import__("torch").cuda.is_available() else "cpu"

    # -- API ----------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
