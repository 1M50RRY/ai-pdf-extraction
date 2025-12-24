"""
Database configuration and session management.

This module sets up SQLAlchemy engine and session factory for PostgreSQL.
"""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Get database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/extraction_db"
)

# Create SQLAlchemy engine
# - pool_pre_ping: Verify connections are alive before using them
# - pool_size: Number of connections to keep in pool
# - max_overflow: Number of connections to allow beyond pool_size
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=os.getenv("SQL_DEBUG", "false").lower() == "true",
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for declarative models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.
    
    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    
    Note: In production, use Alembic migrations instead.
    """
    # Import models to ensure they are registered with Base
    from . import models_db  # noqa: F401
    
    Base.metadata.create_all(bind=engine)

