"""
Database Session and Engine Management for ANPRX Edge ANPR Platform.
"""
from __future__ import annotations

import logging
import time
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from db.config import ensure_mysql_database_exists, get_database_url
from db.models import Base

logger = logging.getLogger("anprx.db")

DATABASE_URL = get_database_url()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_connectivity(max_retries: int = 5, retry_delay: float = 1.0) -> bool:
    """
    Verifies that MySQL is reachable and credentials are valid.
    """
    ensure_mysql_database_exists()
    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                res = conn.execute(text("SELECT 1")).scalar()
                if res == 1:
                    print(f"[DB] Connected to MySQL database successfully on attempt {attempt}")
                    return True
        except Exception as exc:
            print(f"[DB] Connection attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    return False


def init_db() -> None:
    """
    Initializes all 24 database tables safely without destroying existing records.
    """
    try:
        ensure_mysql_database_exists()
        Base.metadata.create_all(bind=engine)
        print("[DB] All 24 ANPRX database tables verified and initialized successfully.")
    except Exception as exc:
        print(f"[DB ERROR] Table initialization failed: {exc}")
        raise
