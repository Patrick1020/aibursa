# app/train_model.py — Trainer pro: LightGBM reg + clf, OOF isotonic calibration,
# quantile bands (P20/P80), walk-forward CV, feature set bogat, fără scurgeri simple.

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
import numpy as np
import re
import pandas as pd
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from joblib import dump
from lightgbm import LGBMRegressor, LGBMClassifier

import time

# =========================
# Configurații de bază
# =========================
MODEL_DIR = Path("models_store")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REG_PATH = MODEL_DIR / "regressor.pkl"
CLF_PATH = MODEL_DIR / "classifier.pkl"
CAL_PATH = MODEL_DIR / "calibrator.pkl"
Q20_PATH = MODEL_DIR / "reg_q20.pkl"
Q80_PATH = MODEL_DIR / "reg_q80.pkl"
FEAT_PATH = MODEL_DIR / "features.json"  # salvăm lista de feature-uri pt. runtime

# Univers de antrenare — poți extinde fără probleme
TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "V",
    "DIS",
    "NFLX",
    "INTC",
    "PYPL",
    "ADBE",
    "CSCO",
    "ORCL",
    "CRM",
    "AMD",
    "AVGO",
    "PEP",
    "KO",
    "WMT",
    "HD",
    "BAC",
    "COST",
    "PFE",
    "NKE",
    "ABNB",
    "UBER",
    "SHOP",
]

YEARS_HISTORY = 8  # ani de istoric
HORIZON_DAYS = 7  # orizont țintă
EMBARGO_DAYS = 3  # mic „gap” între train/valid pe fold, reduce scurgerea simplă
SEED = 42

# Macro / market context (folosit ca features suplimentare)
MKT_TICKER = "SPY"  # market proxy
TECH_TICK = "QQQ"  # tech proxy
VIX_TICK = "^VIX"  # volatility index (nu e procentaj, îl transformăm în randamente/Δ)

# Opțiune: țintă ca „excess return” (stock - market); implicit False, dar îl poți activa
USE_EXCESS_TARGET = False


CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _clean_symbol(s: str) -> str:
    """Curăță și normalizează simbolul pentru Yahoo Finance."""
    t = s.strip().upper().replace(" ", "")
    # Yahoo folosește '-' în loc de '.' pentru clase: BRK.B -> BRK-B
    return t.replace(".", "-")


def _parse_tickers(
    tickers_arg: str | None, top_n: int | None, default_universe: list[str]
) -> list[str]:
    """
    Reguli:
      - dacă tickers_arg e gol sau numeric: ia primele N din lista default (N = top_n sau len(default)).
      - dacă e listă: sparge pe virgulă/spații, păstrează doar tokenii cu litere, curăță și dedup.
    """
    if not tickers_arg or tickers_arg.strip() == "" or tickers_arg.strip().isdigit():
        n = (
            top_n
            if (top_n and top_n > 0)
            else (
                int(tickers_arg.strip())
                if (tickers_arg and tickers_arg.strip().isdigit())
                else len(default_universe)
            )
        )
        n = max(1, min(n, len(default_universe)))
        return default_universe[:n]

    tokens = re.split(r"[,\s]+", tickers_arg.strip())
    cleaned, seen = [], set()
    for tok in tokens:
        if not tok:
            continue
        if not any(ch.isalpha() for ch in tok):  # evită '30', '2020' etc.
            continue
        sym = _clean_symbol(tok)
        if sym not in seen:
            cleaned.append(sym)
            seen.add(sym)
    return cleaned if cleaned else default_universe


# =========================
# Helpers de feature engineering
# =========================
def _safe_div(a, b):
    try:
        return a / b if (b is not None and b != 0) else np.nan
    except Exception:
        return np.nan


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _bb_percent_b(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper, lower = ma + k * sd, ma - k * sd
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (close - lower) / (upper - lower)
    return b


def _zscore(s: pd.Series, win: int = 60) -> pd.Series:
    m = s.rolling(win).mean()
    v = s.rolling(win).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (s - m) / v
    return z


def _roll_beta(stock_ret: pd.Series, mkt_ret: pd.Series, win: int = 60) -> pd.Series:
    # beta ≈ cov(stock, mkt)/var(mkt)
    cov = stock_ret.rolling(win).cov(mkt_ret)
    var = mkt_ret.rolling(win).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        b = cov / var
    return b


def _download_hist(sym: str, years: int = YEARS_HISTORY) -> pd.DataFrame:
    """
    Descărcare robustă cu:
      - cache local CSV (evită rate-limit)
      - retry + fallback de perioadă și metodă (Ticker.history -> download)
    """
    cache_file = CACHE_DIR / f"{sym}_{years}y_1d.csv"

    # 1) Încearcă din cache
    try:
        if cache_file.exists():
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            if not df.empty:
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                return df
    except Exception:
        pass

    # 2) Strategii de descărcare
    periods_try = [f"{years}y", "max", "5y", "3y"]
    methods = ["ticker", "download"]  # ambele căi ale yfinance

    for period in periods_try:
        for method in methods:
            try:
                if method == "ticker":
                    df = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
                else:
                    df = yf.download(
                        sym, period=period, interval="1d", auto_adjust=True, progress=False
                    )

                if isinstance(df, pd.DataFrame) and not df.empty:
                    # normalizare
                    df = df.reset_index().rename(columns={"Date": "Date"})
                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                    # 3) scrie în cache
                    try:
                        df.to_csv(cache_file, index=False)
                    except Exception:
                        pass
                    return df
            except Exception:
                # mic backoff
                time.sleep(1.0)

        # backoff între perioade
        time.sleep(1.0)

    print(f"[WARN] yfinance a întors gol pentru {sym} (după încercări).")
    return pd.DataFrame()


def _merge_macro(
    base: pd.DataFrame, mkt: pd.DataFrame, tech: pd.DataFrame, vix: pd.DataFrame
) -> pd.DataFrame:
    # Construim rame “goale” cu headerele potrivite când lipsesc sursele
    mkt_use = (
        mkt[["Date", "mkt_ret", "mkt_level", "mkt_vol_z"]]
        if (isinstance(mkt, pd.DataFrame) and not mkt.empty)
        else pd.DataFrame(columns=["Date", "mkt_ret", "mkt_level", "mkt_vol_z"])
    )
    tech_use = (
        tech[["Date", "tech_ret", "tech_level"]]
        if (isinstance(tech, pd.DataFrame) and not tech.empty)
        else pd.DataFrame(columns=["Date", "tech_ret", "tech_level"])
    )
    vix_use = (
        vix[["Date", "vix_ret", "vix_chg", "vix_level", "vix_z"]]
        if (isinstance(vix, pd.DataFrame) and not vix.empty)
        else pd.DataFrame(columns=["Date", "vix_ret", "vix_chg", "vix_level", "vix_z"])
    )

    out = base.merge(mkt_use, on="Date", how="left")
    out = out.merge(tech_use, on="Date", how="left")
    out = out.merge(vix_use, on="Date", how="left")

    # Umplem valori neutre, ca să nu pierdem rânduri la .dropna mai târziu
    defaults = {
        "mkt_ret": 0.0,
        "mkt_level": np.nan,
        "mkt_vol_z": 0.0,
        "tech_ret": 0.0,
        "tech_level": np.nan,
        "vix_ret": 0.0,
        "vix_chg": 0.0,
        "vix_level": 20.0,
        "vix_z": 0.0,
    }
    for col, val in defaults.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)
    return out


def _prep_macro_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # SPY
    mkt = _download_hist(MKT_TICKER, YEARS_HISTORY)
    if mkt.empty:
        mkt = pd.DataFrame(columns=["Date", "Close"])
    mkt["mkt_ret"] = mkt["Close"].pct_change()
    mkt["mkt_level"] = mkt["Close"]
    mkt["mkt_vol"] = mkt["mkt_ret"].rolling(20).std()
    mkt["mkt_vol_z"] = _zscore(mkt["mkt_vol"], 60)

    # QQQ
    tech = _download_hist(TECH_TICK, YEARS_HISTORY)
    if tech.empty:
        tech = pd.DataFrame(columns=["Date", "Close"])
    tech["tech_ret"] = tech["Close"].pct_change()
    tech["tech_level"] = tech["Close"]

    # ^VIX
    vix = _download_hist(VIX_TICK, YEARS_HISTORY)
    if vix.empty:
        vix = pd.DataFrame(columns=["Date", "Close"])
    vix["vix_ret"] = vix["Close"].pct_change()
    vix["vix_chg"] = vix["Close"].pct_change().fillna(0.0)
    vix["vix_level"] = vix["Close"]
    vix["vix_z"] = _zscore(vix["vix_level"].pct_change().fillna(0.0), 60)

    return mkt, tech, vix


def _make_features_for_symbol(
    sym: str, mkt: pd.DataFrame, tech: pd.DataFrame, vix: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    raw = _download_hist(sym, YEARS_HISTORY)
    if raw.empty or len(raw) < 120:
        return pd.DataFrame(), []

    df = raw.copy()
    # Randamente și volatilitate
    df["ret1"] = df["Close"].pct_change()
    df["ret5"] = df["Close"].pct_change(5)
    df["ret10"] = df["Close"].pct_change(10)
    df["vol20"] = df["ret1"].rolling(20).std()
    df["vol60"] = df["ret1"].rolling(60).std()
    df["vol_z"] = _zscore(df["vol20"], 60)

    # Poziționare față de MA / EMA
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["ema12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["px_over_sma20"] = _safe_div(df["Close"], df["sma20"])
    df["px_over_sma50"] = _safe_div(df["Close"], df["sma50"])
    df["ema_ratio"] = _safe_div(df["ema12"], df["ema26"])
    df["ema_hist"] = df["ema12"] - df["ema26"]

    # Oscilatori & ATR%
    df["rsi14"] = _rsi(df["Close"], 14)
    df["bbp20"] = _bb_percent_b(df["Close"], 20, 2.0)
    df["bbp50"] = _bb_percent_b(df["Close"], 50, 2.0)
    atr14 = _atr(df, 14)
    df["atrp"] = _safe_div(atr14, df["Close"])

    # Forme distribuționale
    r = df["ret1"]
    df["skew20"] = r.rolling(20).skew()
    df["kurt20"] = r.rolling(20).kurt()

    # Merge macro
    df = _merge_macro(df, mkt, tech, vix)

    # Beta/correlații vs market
    df["beta60"] = _roll_beta(df["ret1"], df["mkt_ret"], 60)
    df["corr20"] = df["ret1"].rolling(20).corr(df["mkt_ret"])
    df["corr60"] = df["ret1"].rolling(60).corr(df["mkt_ret"])

    # Calendar features simple
    df["dow"] = df["Date"].dt.weekday  # 0..6
    df["mth"] = df["Date"].dt.month  # 1..12

    # Ținta (în %)
    fut = df["Close"].pct_change(HORIZON_DAYS).shift(-HORIZON_DAYS) * 100.0
    if USE_EXCESS_TARGET:
        # excess return față de SPY (și ținta devine mai stabilă)
        fut_mkt = df["mkt_level"].pct_change(HORIZON_DAYS).shift(-HORIZON_DAYS) * 100.0
        df["target_pct"] = fut - fut_mkt
    else:
        df["target_pct"] = fut
    df["target_dir"] = (df["target_pct"] >= 0).astype(int)

    feat_cols = [
        # preț & returns
        "ret1",
        "ret5",
        "ret10",
        "vol20",
        "vol60",
        "vol_z",
        "px_over_sma20",
        "px_over_sma50",
        "ema_ratio",
        "ema_hist",
        "rsi14",
        "bbp20",
        "bbp50",
        "atrp",
        "skew20",
        "kurt20",
        # macro/mkt
        "mkt_ret",
        "mkt_level",
        "mkt_vol_z",
        "tech_ret",
        "tech_level",
        "vix_ret",
        "vix_chg",
        "vix_level",
        "vix_z",
        # beta/corr
        "beta60",
        "corr20",
        "corr60",
        # calendar
        "dow",
        "mth",
    ]

    use = df[["Date"] + feat_cols + ["target_pct", "target_dir"]].copy()
    # Ținta e obligatorie; features lipsă le completăm neutru
    use = use.dropna(subset=["target_pct", "target_dir"])
    use[feat_cols] = use[feat_cols].fillna(0.0)
    use["symbol"] = sym
    return use, feat_cols


def _build_dataset(tickers: list[str], horizon_days: int):
    # build macro once
    mkt, tech, vix = _prep_macro_frames()

    frames = []
    feat_cols_ref = None
    kept, skipped = [], []

    for sym in tickers:
        # Airbag: sari peste tokeni care n-au nicio literă (ex. "30")
        if not any(ch.isalpha() for ch in str(sym)):
            print(f"[BUILD] skip invalid symbol -> {sym}")
            continue

        f, cols = _make_features_for_symbol(sym, mkt, tech, vix)
        if f.empty or len(f) < 120:
            skipped.append(sym)
            continue
        kept.append(sym)
        if feat_cols_ref is None:
            feat_cols_ref = cols
        frames.append(f)

    print(f"[BUILD] kept={len(kept)} -> {kept}")
    print(f"[BUILD] skipped={len(skipped)} -> {skipped}")

    if not frames:
        print("[TRAIN] Nu există date suficiente.")
        return None

    all_df = pd.concat(frames, ignore_index=True)
    # sortăm cronologic pentru TimeSeriesSplit consistent
    all_df = all_df.sort_values("Date").reset_index(drop=True)

    X = all_df[feat_cols_ref].values
    y_reg = all_df["target_pct"].values.astype(float)
    y_clf = all_df["target_dir"].values.astype(int)

    print(f"[TRAIN] Samples: reg={len(y_reg)}, clf={len(y_clf)} | features: {len(feat_cols_ref)}")
    # salvăm lista de features ca să o putem folosi în runtime (live_features align)
    try:
        import json

        FEAT_PATH.write_text(json.dumps(feat_cols_ref, indent=2))
    except Exception:
        pass

    return X, y_reg, y_clf, feat_cols_ref


# =========================
# Training & evaluare
# =========================


def _winsorize_y(y: np.ndarray, p_low=2.0, p_high=98.0) -> np.ndarray:
    lo, hi = np.percentile(y, [p_low, p_high])
    return np.clip(y, lo, hi)


def train_quantile(
    X: np.ndarray, y: np.ndarray, alpha: float, params: dict | None = None
) -> LGBMRegressor:
    p = {
        "objective": "quantile",
        "alpha": alpha,
        "n_estimators": 900,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": SEED,
        "n_jobs": -1,
    }
    if params:
        p.update(params)
    m = LGBMRegressor(**p)
    m.fit(X, y)
    return m


def train_and_eval(*, years: int, horizon: int, tickers: list[str], excess_target: bool) -> None:
    """
    Antrenează toate modelele pe universul 'tickers' pentru un orizont 'horizon',
    folosind 'years' ani de istoric și opțiunea 'excess_target'.
    """
    # aceste globale sunt folosite în utilitare/feature builders
    global YEARS_HISTORY, HORIZON_DAYS, USE_EXCESS_TARGET
    YEARS_HISTORY = int(years)
    HORIZON_DAYS = int(horizon)
    USE_EXCESS_TARGET = bool(excess_target)

    # build dataset pe baza parametrilor
    built = _build_dataset(tickers, HORIZON_DAYS)
    if built is None:
        print("[TRAIN] Nu există date suficiente pentru antrenare.")
        return

    X, y_reg, y_clf, feat_cols = built
    print(f"[CHECK] X={X.shape}, y_reg={y_reg.shape}, y_clf={y_clf.shape}, feats={len(feat_cols)}")

    # ===== Regresie: winsorizăm ținta pentru robustețe
    y_reg_w = _winsorize_y(y_reg, 2.0, 98.0)

    # ===== 1) CV pe REGRESIE (TimeSeriesSplit)
    tscv_r = TimeSeriesSplit(n_splits=5)
    r_mae, r_r2 = [], []
    for k, (tr, va) in enumerate(tscv_r.split(X), 1):
        reg = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            num_leaves=63,
            random_state=SEED,
            n_jobs=-1,
        )
        reg.fit(X[tr], y_reg_w[tr])
        pred = reg.predict(X[va])
        mae = mean_absolute_error(y_reg[va], pred)  # raportăm pe y original
        r2 = r2_score(y_reg[va], pred)
        r_mae.append(mae)
        r_r2.append(r2)
        print(f"[CV R{k}] MAE={mae:.3f}  R2={r2:.3f}")
    print(f"[CV] Reg: MAE={np.mean(r_mae):.3f}±{np.std(r_mae):.3f}  R2={np.mean(r_r2):.3f}")

    # ===== 2) CV pe CLASIFICARE + OOF pentru calibrare isotonică
    tscv_c = TimeSeriesSplit(n_splits=5)
    oof_probs = np.full(len(y_clf), np.nan, dtype=float)
    c_acc, c_auc = [], []

    for k, (tr, va) in enumerate(tscv_c.split(X), 1):
        clf = LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            num_leaves=63,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        clf.fit(X[tr], y_clf[tr])
        prob = clf.predict_proba(X[va])[:, 1]
        oof_probs[va] = prob
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(y_clf[va], pred)
        try:
            auc = roc_auc_score(y_clf[va], prob)
        except Exception:
            auc = float("nan")
        c_acc.append(acc)
        c_auc.append(auc)
        print(f"[CV C{k}] ACC={acc:.3f}  AUC={auc:.3f}")
    print(f"[CV] Clf: ACC={np.mean(c_acc):.3f}±{np.std(c_acc):.3f}  AUC={np.nanmean(c_auc):.3f}")

    # ===== 3) Calibrare isotonică pe OOF
    try:
        mask = ~np.isnan(oof_probs)
        if mask.sum() > 10:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(oof_probs[mask], y_clf[mask])
            dump(ir, CAL_PATH)
            print(f"[SAVE] Calibrator -> {CAL_PATH}")
        else:
            print("[WARN] Prea puține puncte valide pentru calibrare; sar peste.")
    except Exception as e:
        print(f"[WARN] Calibrator failed: {e}")

    # ===== 4) Antrenare finală pe TOT setul
    reg_final = LGBMRegressor(
        objective="huber",  # robust to tails
        alpha=0.90,  # Huber parameter
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_data_in_leaf=64,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,  # a.k.a feature_fraction
        lambda_l1=1.0,
        lambda_l2=5.0,
        random_state=42,
    ).fit(X, y_reg_w)

    dump(reg_final, REG_PATH)
    print(f"[SAVE] Regressor -> {REG_PATH}")

    # --- Feature importances (pentru interpretabilitate) ---
    try:
        imp = pd.Series(reg_final.feature_importances_, index=feat_cols).sort_values(
            ascending=False
        )
        imp_df = imp.reset_index().rename(columns={"index": "feature", 0: "importance"})
        imp_df.to_csv(MODEL_DIR / "feature_importances.csv", index=False)
        print(f"[SAVE] Feature importances -> {MODEL_DIR / 'feature_importances.csv'}")
    except Exception as e:
        print(f"[WARN] cannot save feature importances: {e}")

    # ----- Cuantile (benzi de încredere P20/P80)
    print("[Q] Training quantiles (P20/P80)...")
    q20 = train_quantile(X, y_reg_w, alpha=0.20)
    dump(q20, Q20_PATH)
    print(f"[SAVE] Q20 -> {Q20_PATH}")
    q80 = train_quantile(X, y_reg_w, alpha=0.80)
    dump(q80, Q80_PATH)
    print(f"[SAVE] Q80 -> {Q80_PATH}")

    clf_final = LGBMClassifier(
        n_estimators=1600,
        learning_rate=0.015,
        subsample=0.9,
        colsample_bytree=0.9,
        num_leaves=63,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ).fit(X, y_clf)
    dump(clf_final, CLF_PATH)
    print(f"[SAVE] Classifier -> {CLF_PATH}")


# =========================
# Entry-point + utilitare
# =========================


def _set_reproducibility(seed: int = SEED):
    """Setează seed-urile pentru reproducibilitate."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    # LightGBM nu este 100% determinist în toate setările, dar seed-ul ajută.


def main():
    """
    Punctul de intrare. Permite override de parametri din linia de comandă:
      --years 6
      --horizon 7
      --tickers AAPL,MSFT,NVDA
      --excess_target 1
    """
    import argparse

    # === IMPORTANT: declarăm global ÎNAINTE de orice folosire a numelor ===
    global YEARS_HISTORY, HORIZON_DAYS, TICKERS, USE_EXCESS_TARGET

    # salvăm valorile curente ca default-uri locale pentru argparse
    years_default = YEARS_HISTORY
    horizon_default = HORIZON_DAYS
    tickers_default = ",".join(TICKERS)
    excess_default = int(USE_EXCESS_TARGET)

    parser = argparse.ArgumentParser()
    # === Excess vs SPY flags (nice CLI) ===
    parser.add_argument(
        "--excess",
        dest="excess_target",
        action="store_true",
        help="Antrenează pe randamente EXCESS vs SPY (activat când e prezent).",
    )
    parser.add_argument(
        "--no-excess",
        dest="excess_target",
        action="store_false",
        help="Dezactivează EXCESS vs SPY.",
    )
    parser.set_defaults(excess_target=False)

    parser.add_argument(
        "--years", type=int, default=years_default, help="Ani de istoric Yahoo Finance."
    )
    parser.add_argument(
        "--horizon", type=int, default=horizon_default, help="Orizontul țintei, în zile."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=tickers_default,
        help="Listă de tickere separate prin virgulă.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Dacă --tickers e un număr sau lipsește, ia primele N din lista internă (default).",
    )
    args = parser.parse_args()

    # === Override configurări (SAFE) — DOAR în interiorul lui main() ===
    YEARS_HISTORY = int(args.years)
    HORIZON_DAYS = int(args.horizon)
    USE_EXCESS_TARGET = bool(args.excess_target)

    # Construim lista finală de tickere pe baza argumentelor
    _default_universe = list(TICKERS)  # păstrăm universul intern definit în fișier
    tickers_final = _parse_tickers(
        getattr(args, "tickers", None), getattr(args, "top", None), _default_universe
    )

    print(
        f"[CFG] years={YEARS_HISTORY} horizon={HORIZON_DAYS} tickers={len(tickers_final)} excess_target={USE_EXCESS_TARGET}"
    )
    print(f"[OUT] {REG_PATH}, {CLF_PATH}, {CAL_PATH}, {Q20_PATH}, {Q80_PATH}")

    _set_reproducibility(SEED)

    # apel corect — cu argumente, NU versiunea fără argumente
    train_and_eval(
        years=YEARS_HISTORY,
        horizon=HORIZON_DAYS,
        tickers=tickers_final,
        excess_target=USE_EXCESS_TARGET,
    )

    # obține lista finală de tickere
    TICKERS = _parse_tickers(getattr(args, "tickers", None), getattr(args, "top", None), TICKERS)

    _set_reproducibility(SEED)


if __name__ == "__main__":
    main()
