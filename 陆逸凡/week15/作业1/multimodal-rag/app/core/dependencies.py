"""Dependency injection for FastAPI."""
from functools import lru_cache
from typing import Generator

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.database import engine, SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Get database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@lru_cache()
def get_settings():
    """Get cached settings instance."""
    return settings