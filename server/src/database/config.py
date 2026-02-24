import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Resolve .env relative to THIS file so the path works regardless of CWD.
_ENV_PATH = Path(__file__).parent.parent / "agents" / ".env"
load_dotenv(_ENV_PATH)

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trip_management")

SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Guard: skip DB entirely if password is not configured
DB_CONFIGURED = bool(DB_PASSWORD)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"connect_timeout": 3},   # fail fast (3 s)
    pool_pre_ping=True,                     # verify connections before use
    pool_size=2,
    pool_timeout=5,
    pool_recycle=300,
) if DB_CONFIGURED else None

SessionLocal = (
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
    if engine
    else None
)

Base = declarative_base()

def get_db():
    if SessionLocal is None:
        raise RuntimeError("Database is not configured (DB_PASSWORD not set)")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
