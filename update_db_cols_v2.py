# app/migrate_2025_08_add_intervals.py
from sqlalchemy import text
from app.database import engine

REQUIRED_COLS = {
    # text / justificări
    "trade_outcome": "TEXT",
    "prob_justification": "TEXT",
    "reco_reason": "TEXT",
    "short_justification": "TEXT",
    # scoruri/valori numerice
    "actual_price": "REAL",
    "final_percent": "REAL",
    "final_probability": "REAL",
    # intervale / cuantile
    "ci_low_pct": "REAL",
    "ci_high_pct": "REAL",
    "p20": "REAL",
    "p80": "REAL",
    "price_low": "REAL",
    "price_high": "REAL",
}

TABLE = "stock_predictions"


def existing_columns():
    with engine.connect() as conn:
        cols = conn.execute(text(f"PRAGMA table_info('{TABLE}')")).fetchall()
        return {row[1] for row in cols}  # row[1] = name


def add_missing_columns():
    cols_now = existing_columns()
    missing = [c for c in REQUIRED_COLS if c not in cols_now]
    if not missing:
        print("[MIGRATE] Nimic de făcut. Toate coloanele există deja.")
        return

    print(f"[MIGRATE] Lipsesc coloanele: {missing}")
    with engine.begin() as conn:
        for col in missing:
            coltype = REQUIRED_COLS[col]
            sql = f"ALTER TABLE {TABLE} ADD COLUMN {col} {coltype}"
            conn.execute(text(sql))
            print(f"[MIGRATE] Adăugat: {col} ({coltype})")

    print("[MIGRATE] Gata. Tabela a fost actualizată.")


if __name__ == "__main__":
    add_missing_columns()
