"""
Configuration module for ANPRX Edge ANPR and Trip Management database connection.
"""
from __future__ import annotations

import os
import urllib.parse
from pathlib import Path

# Load .env file from backend directory if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "anprx_app")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "AnprxAppSecurePass2026!")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "anprx")


def get_database_url() -> str:
    """
    Returns the SQLAlchemy formatted database URL.
    Prefers DATABASE_URL from environment if explicitly set; otherwise constructs
    properly URL-encoded mysql+pymysql URL.
    """
    custom_url = os.environ.get("DATABASE_URL")
    if custom_url and custom_url.startswith("mysql"):
        return custom_url

    encoded_user = urllib.parse.quote_plus(MYSQL_USER)
    encoded_pass = urllib.parse.quote_plus(MYSQL_PASSWORD)
    encoded_db = urllib.parse.quote_plus(MYSQL_DATABASE)

    return f"mysql+pymysql://{encoded_user}:{encoded_pass}@{MYSQL_HOST}:{MYSQL_PORT}/{encoded_db}?charset=utf8mb4"


def ensure_mysql_database_exists() -> None:
    """
    Checks if the MySQL target database exists and creates it safely if missing.
    """
    try:
        import pymysql
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            connect_timeout=3
        )
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        conn.close()
    except Exception as e:
        print(f"[DB Notice] Database pre-check: {e}")
