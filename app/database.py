from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager


DATABASE_URL = "sqlite:///./stock_analysis.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    Base.metadata.create_all(bind=engine)
    ensure_optional_columns()  # <— adăugat


def ensure_optional_columns():
    with engine.connect() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(stock_predictions)").fetchall()
        cols = {r[1] for r in rows}  # nume coloane

        if "ci_low_pct" not in cols:
            conn.exec_driver_sql("ALTER TABLE stock_predictions ADD COLUMN ci_low_pct REAL")
        if "ci_high_pct" not in cols:
            conn.exec_driver_sql("ALTER TABLE stock_predictions ADD COLUMN ci_high_pct REAL")


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
