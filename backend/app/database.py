"""
database.py — SQLAlchemy setup and ORM models for prediction logging.

Supports both SQLite (local dev) and PostgreSQL (AWS RDS) via DATABASE_URL.
"""

from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

from .config import settings

# ── Engine & Session ──────────────────────────────────────────────────────────
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    # SQLite needs check_same_thread=False for FastAPI async
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class Prediction(Base):
    """Log of every prediction made through the API."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    drug_a = Column(String(255), nullable=False, index=True)
    drug_b = Column(String(255), nullable=False, index=True)
    smiles_a = Column(Text, nullable=False)
    smiles_b = Column(Text, nullable=False)
    model_name = Column(String(50), nullable=False)
    predicted_class = Column(Integer, nullable=False)
    predicted_label = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_tables():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
