# app/calibrate_probs.py
import os
import sqlite3
import datetime as dt
import yfinance as yf
from app.calib import ProbCalibrator

DB = "stock_analysis.db"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
CALIB_PATH = os.path.join(MODELS_DIR, "prob_calibrator.joblib")

HORIZON = 7


def realized_dir(symbol: str, d0: str) -> int | None:
    """
    returnează 1 dacă close(d0+7) >= close(d0), altfel 0; None dacă lipsesc date
    """
    t = yf.Ticker(symbol)
    # luăm ceva buffer ca să acoperim weekend-uri
    df = t.history(
        start=(dt.datetime.fromisoformat(d0) - dt.timedelta(days=3)).date(),
        end=(dt.datetime.fromisoformat(d0) + dt.timedelta(days=HORIZON + 7)).date(),
    )
    if df.empty:
        return None
    # găsește prima zi de tranzacționare >= d0
    idx = df.index.searchsorted(dt.datetime.fromisoformat(d0))
    if idx >= len(df):
        return None
    price0 = float(df["Close"].iloc[idx])
    idx2 = idx + HORIZON
    if idx2 >= len(df):
        return None
    price1 = float(df["Close"].iloc[idx2])
    return 1 if price1 >= price0 else 0


def load_samples():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    # folosim final_probability dacă există, altfel probability (LLM)
    cur.execute(
        """
      SELECT symbol, published_at,
             COALESCE(final_probability, probability) AS p_pred
      FROM stock_predictions
      WHERE p_pred IS NOT NULL
      ORDER BY published_at ASC
    """
    )
    rows = cur.fetchall()
    conn.close()
    XS = []
    YS = []
    for sym, ts, p in rows:
        try:
            d0 = ts.replace(" ", "T").split(".")[0]  # iso-ish
            y = realized_dir(sym, d0)
            if y is None:
                continue
            XS.append(float(p))
            YS.append(int(y))
        except Exception:
            continue
    return XS, YS


if __name__ == "__main__":
    xs, ys = load_samples()
    if len(xs) < 50:
        print(f"Not enough samples for calibration: {len(xs)}")
    else:
        cal = ProbCalibrator()
        cal.fit(xs, ys)
        cal.save(CALIB_PATH)
        print(f"[OK] Saved calibrator to {CALIB_PATH} with {len(xs)} samples")
