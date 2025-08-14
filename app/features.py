# app/features.py
import math
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from functools import lru_cache


def _hist(symbol: str, period="6mo", interval="1d") -> pd.DataFrame:
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval=interval)
    return df if not df.empty else pd.DataFrame()


def daily_vol_pct(symbol: str, lookback=20) -> float:
    """Annualized daily volatility in %, approximated from last N daily returns."""
    df = _hist(symbol, period="3mo", interval="1d")
    if df.empty or len(df) < lookback + 2:
        return 0.0
    r = df["Close"].pct_change().dropna()
    sigma = r.tail(lookback).std() * 100.0  # daily %
    return float(sigma)


def atr_pct(symbol: str, lookback=14) -> float:
    df = _hist(symbol, period="3mo", interval="1d")
    if df.empty or len(df) < lookback + 2:
        return 0.0
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = tr.rolling(lookback).mean().iloc[-1]
    return float((atr / close.iloc[-1]) * 100.0) if close.iloc[-1] else 0.0


def days_to_next_earnings(symbol: str) -> int | None:
    """Uses yfinance calendar when available."""
    try:
        t = yf.Ticker(symbol)
        t.get_income_stmt(freq="quarterly")  # forces metadata load
        # yfinance exposes earnings dates through 't.calendar' or 't.get_earnings_dates'
        dates = t.get_earnings_dates(limit=8)
        future = dates[dates.index >= pd.Timestamp.utcnow().tz_localize("UTC")]
        if len(future) == 0:
            return None
        next_date = future.index.min().to_pydatetime()
        days = (next_date - dt.datetime.now(dt.timezone.utc)).days
        return int(days)
    except Exception:
        return None


# după linia 055 (înlocuiește market_regime existent)
def market_regime() -> dict:
    """
    Expune macro snapshot (din data_sources.get_macro_snapshot) într-un format
    gata de folosit de model/feature engineering.
    """
    try:
        from .data_sources import get_macro_snapshot

        snap = get_macro_snapshot()
        return {
            "reg_vix_last": snap.get("vix_last"),
            "reg_vix_z20": snap.get("vix_z20"),
            "reg_vix_term": snap.get("vix_term"),
            "reg_dxy_ret5": snap.get("dxy_ret5"),
            "reg_dxy_z20": snap.get("dxy_z20"),
            "reg_tnx_last": snap.get("tnx_last"),
            "reg_tnx_chg5": snap.get("tnx_chg5"),
            "reg_hyg_lqd_ratio": snap.get("hyg_lqd_ratio"),
            "reg_hyg_lqd_z20": snap.get("hyg_lqd_z20"),
            "reg_spy_vol20": snap.get("spy_vol20"),
            "reg_spy_ret5": snap.get("spy_ret5"),
            "reg_label": snap.get("regime_label"),
        }
    except Exception as e:
        print(f"[regime] features fail: {e}")
        return {
            "reg_vix_last": None,
            "reg_vix_z20": None,
            "reg_vix_term": None,
            "reg_dxy_ret5": None,
            "reg_dxy_z20": None,
            "reg_tnx_last": None,
            "reg_tnx_chg5": None,
            "reg_hyg_lqd_ratio": None,
            "reg_hyg_lqd_z20": None,
            "reg_spy_vol20": None,
            "reg_spy_ret5": None,
            "reg_label": 0,
        }


# ====== ML features builder ======


def _safe_pct(a, b):
    try:
        return (a - b) / b * 100.0 if (a is not None and b not in (None, 0)) else None
    except ZeroDivisionError:
        return None


def _get_history_until(symbol: str, end_dt: dt.datetime, lookback_days: int = 420):
    end_dt = end_dt if end_dt.tzinfo else end_dt.replace(tzinfo=dt.timezone.utc)
    start_dt = end_dt - dt.timedelta(days=lookback_days)
    df = yf.Ticker(symbol).history(start=start_dt.date(), end=end_dt.date())
    return df.dropna() if df is not None else None


def _rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr(df: pd.DataFrame, period: int = 14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _bb_percent_b(close: pd.Series, period: int = 20, k: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + k * sd
    lower = ma - k * sd
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (close - lower) / (upper - lower)
    return b


def _skew_kurt(series: pd.Series, win: int = 20):
    s = series.pct_change().rolling(win)
    return s.skew(), s.kurt()


def _vol_zscore(vol: pd.Series, win: int = 20):
    m = vol.rolling(win).mean()
    sd = vol.rolling(win).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (vol - m) / sd
    return z


@lru_cache(maxsize=1)
def _macro_hist():
    # SPY (benchmark), ^VIX (volatilitate), ^TNX (10y)
    spy = yf.Ticker("SPY").history(period="10y")
    vix = yf.Ticker("^VIX").history(period="10y")
    tnx = yf.Ticker("^TNX").history(period="10y")
    return spy, vix, tnx


def _regime_at(date_utc: dt.datetime):
    spy, vix, tnx = _macro_hist()
    # aliniază pe data curentă (prima bară >= date_utc)
    ts = pd.Timestamp(date_utc)

    def _val(df, col="Close"):
        row = df[df.index >= ts].head(1)
        if row.empty:
            row = df.tail(1)
        return float(row[col].iloc[0]) if not row.empty else np.nan

    # SPY ret20 (proxy momentum piață)
    spy_close = spy["Close"]
    spy_row = spy_close[spy_close.index >= ts].head(1)
    if spy_row.empty:
        spy_row = spy_close.tail(1)
    idx = spy_close.index.get_indexer([spy_row.index[0]])[0]
    if idx >= 20:
        spy_ret20 = float((spy_close.iloc[idx] / spy_close.iloc[idx - 20] - 1) * 100.0)
    else:
        spy_ret20 = 0.0

    vix_now = _val(vix, "Close")
    # Δ5 zile VIX
    vix_close = vix["Close"]
    vix_row = vix_close[vix_close.index >= ts].head(1)
    if vix_row.empty:
        vix_row = vix_close.tail(1)
    vidx = vix_close.index.get_indexer([vix_row.index[0]])[0]
    vix_d5 = float(vix_close.iloc[vidx] - vix_close.iloc[max(0, vidx - 5)]) if vidx >= 1 else 0.0

    # TNX nivel (aprox. yield 10y *10)
    _val(tnx, "Close")

    # 3 valori numerice
    return [
        spy_ret20 if np.isfinite(spy_ret20) else 0.0,
        vix_now if np.isfinite(vix_now) else 0.0,
        vix_d5 if np.isfinite(vix_d5) else 0.0,
        # (opțional) tnx_now, dar îl lăsăm pe 0 pentru compatibilitate; adaugă-l dacă vrei +1 feature
    ][:3]


def build_features_for_date(symbol: str, asof: dt.datetime):
    """
    Vector de features (1, N), folosind DOAR date <= asof (fără scurgeri).
    Ordinea este sincronă cu train_model.py.
    """
    try:
        df = _get_history_until(symbol, asof)
        if df is None or df.empty or len(df) < 80:
            return None

        df = df.copy()
        close = df["Close"].astype(float)
        df["High"].astype(float)
        df["Low"].astype(float)
        vol = df["Volume"].astype(float)
        c = float(close.iloc[-1])

        # randamente
        ret1 = _safe_pct(close.iloc[-1], close.iloc[-2]) if len(close) >= 2 else 0.0
        ret5 = _safe_pct(close.iloc[-1], close.iloc[-6]) if len(close) >= 6 else 0.0
        ret20 = _safe_pct(close.iloc[-1], close.iloc[-21]) if len(close) >= 21 else 0.0

        # vol
        daily_ret = close.pct_change()
        vol20 = float(daily_ret.rolling(20).std().iloc[-1] * 100.0) if len(close) >= 21 else 0.0
        vol60 = float(daily_ret.rolling(60).std().iloc[-1] * 100.0) if len(close) >= 61 else 0.0
        vol_ratio = (vol20 / vol60) if (vol60 and np.isfinite(vol60) and vol60 != 0) else 1.0

        sma5 = float(close.rolling(5).mean().iloc[-1]) if len(close) >= 5 else c
        sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else c
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else c

        d5 = (c - sma5) / c * 100.0
        d20 = (c - sma20) / c * 100.0
        d50 = (c - sma50) / c * 100.0

        rsi14 = float(_rsi(close, 14).iloc[-1]) if len(close) >= 15 else 50.0
        atr14 = _atr(df, 14).iloc[-1]
        atrp = float((atr14 / c) * 100.0) if np.isfinite(atr14) else 0.0

        # Bollinger %b, volum z-score, skew/kurt pe 20
        bb = float(_bb_percent_b(close, 20, 2.0).iloc[-1]) if len(close) >= 25 else 0.5
        vz = float(_vol_zscore(vol, 20).iloc[-1]) if len(vol) >= 25 else 0.0
        skew20 = float(_skew_kurt(close, 20)[0].iloc[-1]) if len(close) >= 40 else 0.0
        kurt20 = float(_skew_kurt(close, 20)[1].iloc[-1]) if len(close) >= 40 else 0.0

        # distanțe 52w
        win = min(len(close), 252)
        hh = float(close.iloc[-win:].max())
        ll = float(close.iloc[-win:].min())
        dist_high = _safe_pct(c, hh) if np.isfinite(hh) and hh else 0.0
        dist_low = _safe_pct(c, ll) if np.isfinite(ll) and ll else 0.0

        # Regim piață (3 valori numerice din _regime_at)
        reg1, reg2, reg3 = _regime_at(asof)

        # --- NOI: beta & corr vs SPY (60d), slope20, gap1, sezonalitate
        # SPY pentru fereastra comună
        spy, _, _ = _macro_hist()  # spy, vix, tnx
        spy_close = spy["Close"].astype(float)
        # aliniază pe calendar
        join = pd.DataFrame({"stock": close}).join(spy_close.rename("spy"), how="inner")
        if len(join) >= 65:
            sret = join["stock"].pct_change().tail(60)
            rspy = join["spy"].pct_change().tail(60)
            cov = (
                np.cov(sret.dropna(), rspy.dropna())[0, 1]
                if sret.dropna().size and rspy.dropna().size
                else 0.0
            )
            var_spy = np.var(rspy.dropna()) if rspy.dropna().size else 1e-9
            beta60 = float(cov / var_spy) if var_spy != 0 else 0.0
            corr60 = (
                float(np.corrcoef(sret.dropna(), rspy.dropna())[0, 1])
                if sret.dropna().size and rspy.dropna().size
                else 0.0
            )
        else:
            beta60, corr60 = 1.0, 0.0

        # slope pe 20 zile (regresie liniară pe index normalizat)
        slope20 = 0.0
        if len(close) >= 20:
            y = close.tail(20).values
            x = np.arange(len(y))
            m, b = np.polyfit(x, y, 1)
            slope20 = float((m / y[-1]) * 100.0)  # % din preț pe pas

        # gap de deschidere față de închiderea precedentă
        gap1 = 0.0
        if "Open" in df.columns and len(df) >= 2:
            prev_close = float(df["Close"].iloc[-2])
            today_open = float(df["Open"].iloc[-1])
            gap1 = _safe_pct(today_open, prev_close) or 0.0

        # sezonalitate
        ts = df.index[-1].to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        wd = ts.weekday()  # 0..6
        mo = ts.month  # 1..12
        wd_s, wd_c = math.sin(2 * math.pi * wd / 7.0), math.cos(2 * math.pi * wd / 7.0)
        mo_s, mo_c = math.sin(2 * math.pi * mo / 12.0), math.cos(2 * math.pi * mo / 12.0)

        feats = [
            c,
            ret1 or 0.0,
            ret5 or 0.0,
            ret20 or 0.0,
            vol20 or 0.0,
            vol60 or 0.0,
            d5,
            d20,
            d50,
            rsi14 if np.isfinite(rsi14) else 50.0,
            atrp if np.isfinite(atrp) else 0.0,
            bb if np.isfinite(bb) else 0.5,
            vz if np.isfinite(vz) else 0.0,
            skew20 if np.isfinite(skew20) else 0.0,
            kurt20 if np.isfinite(kurt20) else 0.0,
            dist_high or 0.0,
            dist_low or 0.0,
            reg1,
            reg2,
            reg3,
            # new
            beta60,
            corr60,
            vol_ratio,
            slope20,
            gap1,
            wd_s,
            wd_c,
            mo_s,
            mo_c,
        ]
        return np.asarray(feats, dtype=float).reshape(1, -1)
    except Exception:
        return None


def build_features_for_symbol(symbol: str):
    return build_features_for_date(symbol, dt.datetime.now(dt.timezone.utc))


def _safe_last(series, default=np.nan):
    try:
        return float(series.iloc[-1])
    except Exception:
        return default


def _pct_change(a, b):
    # return % change from b to a
    try:
        return (float(a) / float(b) - 1.0) * 100.0
    except Exception:
        return np.nan


def _zscore(x, win=60):
    # zscore of last value vs rolling mean/std
    if len(x) < win:
        return np.nan
    roll = x.rolling(win)
    mu = roll.mean().iloc[-1]
    sd = roll.std(ddof=0).iloc[-1]
    if sd is None or sd == 0 or np.isnan(sd):
        return np.nan
    return (x.iloc[-1] - mu) / sd


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _bbp(close, window=20, num_sd=2.0):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper = ma + num_sd * sd
    lower = ma - num_sd * sd
    rng = upper - lower
    if len(close) < window or rng.iloc[-1] == 0 or pd.isna(rng.iloc[-1]):
        return np.nan
    return (close.iloc[-1] - lower.iloc[-1]) / rng.iloc[-1]


def _atr_percent(df, period=14):
    # df must have High, Low, Close
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_close = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(
        axis=1
    )
    atr = tr.rolling(period).mean()
    last_close = _safe_last(c)
    last_atr = _safe_last(atr)
    if last_close and not math.isclose(last_close, 0.0):
        return (last_atr / last_close) * 100.0
    return np.nan


def _beta_and_corr(asset_rets, mkt_rets, win):
    # returns beta_win, corr_win
    if len(asset_rets) < win or len(mkt_rets) < win:
        return (np.nan, np.nan)
    xr = asset_rets.iloc[-win:]
    yr = mkt_rets.iloc[-win:]
    cov = np.cov(xr, yr, ddof=0)[0, 1]
    var = np.var(yr, ddof=0)
    beta = cov / var if var and not np.isnan(var) else np.nan
    corr = np.corrcoef(xr, yr)[0, 1]
    return (beta, corr)


def _history(symbol, period_days=270):
    # 270 zile ca să acoperim rolări de 60/120 fără probleme
    t = yf.Ticker(symbol)
    # folosim auto_adjust pentru a evita dividend/split noise
    df = t.history(period=f"{period_days}d", auto_adjust=True)
    # asigură coloane standard (yfinance poate întoarce doar 'Close' dacă e crypto)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    return df.dropna(subset=["Close"])


def make_live_features(symbol: str) -> dict:
    """
    Returnează un dict cu cheile folosite la antrenare:
      ret1, ret5, ret10, vol20, vol60, vol_z,
      px_over_sma20, px_over_sma50, ema_ratio, ema_hist, rsi14,
      bbp20, bbp50, atrp, skew20, kurt20,
      mkt_ret, mkt_level, mkt_vol_z,
      tech_ret, tech_level,
      vix_ret, vix_chg, vix_level, vix_z,
      beta60, corr20, corr60,
      dow, mth
    """
    # 1) Date pentru activ, piață (SPY), tehnologie (QQQ), risc (VIX)
    df = _history(symbol)
    df_mkt = _history("SPY")
    df_tech = _history("QQQ")
    df_vix = _history("^VIX")

    if len(df) < 70 or len(df_mkt) < 70 or len(df_tech) < 70 or len(df_vix) < 70:
        # insuficiente date pentru rolări
        return {}

    close = df["Close"].copy()
    df["High"].copy()
    df["Low"].copy()

    rets = close.pct_change() * 100.0  # % points / zi
    mkt_close = df_mkt["Close"].copy()
    mkt_rets = mkt_close.pct_change() * 100.0

    tech_close = df_tech["Close"].copy()
    tech_rets = tech_close.pct_change() * 100.0

    vix_close = df_vix["Close"].copy()
    vix_rets = vix_close.pct_change() * 100.0

    # 2) Returns scurte
    ret1 = _safe_last(rets)
    ret5 = _pct_change(close.iloc[-1], close.iloc[-6]) if len(close) >= 6 else np.nan
    ret10 = _pct_change(close.iloc[-1], close.iloc[-11]) if len(close) >= 11 else np.nan

    # 3) Volatilități
    vol20 = _safe_last(rets.rolling(20).std(ddof=0))
    vol60 = _safe_last(rets.rolling(60).std(ddof=0))
    # z-score simplu al vol20 față de istoric 60
    vol_z = (
        (vol20 - vol60) / (vol60 + 1e-9) if not np.isnan(vol20) and not np.isnan(vol60) else np.nan
    )

    # 4) Medii/SMA/EMA
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    px_over_sma20 = (
        (close.iloc[-1] / sma20.iloc[-1] - 1.0) if not pd.isna(sma20.iloc[-1]) else np.nan
    )
    px_over_sma50 = (
        (close.iloc[-1] / sma50.iloc[-1] - 1.0) if not pd.isna(sma50.iloc[-1]) else np.nan
    )

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    ema_ratio = (
        (ema12.iloc[-1] / ema26.iloc[-1] - 1.0)
        if not pd.isna(ema12.iloc[-1]) and not pd.isna(ema26.iloc[-1])
        else np.nan
    )
    ema_hist = (
        (macd.iloc[-1] - signal.iloc[-1])
        if not pd.isna(macd.iloc[-1]) and not pd.isna(signal.iloc[-1])
        else np.nan
    )

    # 5) RSI & Bollinger
    rsi14 = _safe_last(_rsi(close, 14))
    bbp20 = _bbp(close, 20, 2.0)
    bbp50 = _bbp(close, 50, 2.0)

    # 6) ATR în %
    atrp = _atr_percent(df[["High", "Low", "Close"]], 14)

    # 7) Skew/Kurt pe 20 de zile
    skew20 = _safe_last(rets.rolling(20).apply(lambda x: pd.Series(x).skew(), raw=False))
    kurt20 = _safe_last(rets.rolling(20).apply(lambda x: pd.Series(x).kurt(), raw=False))

    # 8) Piață (SPY)
    mkt_ret = _safe_last(mkt_rets)
    mkt_level = (
        (mkt_close.iloc[-1] / mkt_close.rolling(200).mean().iloc[-1] - 1.0)
        if len(mkt_close) >= 200
        else np.nan
    )
    mkt_vol20 = _safe_last(mkt_rets.rolling(20).std(ddof=0))
    mkt_vol60 = _safe_last(mkt_rets.rolling(60).std(ddof=0))
    mkt_vol_z = (
        (mkt_vol20 - mkt_vol60) / (mkt_vol60 + 1e-9)
        if not np.isnan(mkt_vol20) and not np.isnan(mkt_vol60)
        else np.nan
    )

    # 9) Tech (QQQ)
    tech_ret = _safe_last(tech_rets)
    tech_level = (
        (tech_close.iloc[-1] / tech_close.rolling(200).mean().iloc[-1] - 1.0)
        if len(tech_close) >= 200
        else np.nan
    )

    # 10) VIX
    vix_ret = _safe_last(vix_rets)
    vix_chg = _pct_change(vix_close.iloc[-1], vix_close.iloc[-2]) if len(vix_close) >= 2 else np.nan
    vix_level = (
        (vix_close.iloc[-1] / vix_close.rolling(200).mean().iloc[-1] - 1.0)
        if len(vix_close) >= 200
        else np.nan
    )
    vix_z = _zscore(vix_close, 60)

    # 11) Beta/Corr față de SPY
    beta60, corr60 = _beta_and_corr(rets, mkt_rets, 60)
    _, corr20 = _beta_and_corr(rets, mkt_rets, 20)

    # 12) Datetime features (de pe ultimul index)
    last_ts = df.index[-1]
    # yfinance poate avea timezone; folosim .tz_localize(None) ca să nu stricăm formatele
    try:
        dt = pd.to_datetime(last_ts).tz_localize(None)
    except Exception:
        dt = pd.to_datetime(last_ts)
    dow = int(dt.weekday())
    mth = int(dt.month)

    # Dict final – chei ALINIATE cu train_model
    feats = {
        "ret1": ret1,
        "ret5": ret5,
        "ret10": ret10,
        "vol20": vol20,
        "vol60": vol60,
        "vol_z": vol_z,
        "px_over_sma20": px_over_sma20,
        "px_over_sma50": px_over_sma50,
        "ema_ratio": ema_ratio,
        "ema_hist": ema_hist,
        "rsi14": rsi14,
        "bbp20": bbp20,
        "bbp50": bbp50,
        "atrp": atrp,
        "skew20": skew20,
        "kurt20": kurt20,
        "mkt_ret": mkt_ret,
        "mkt_level": mkt_level,
        "mkt_vol_z": mkt_vol_z,
        "tech_ret": tech_ret,
        "tech_level": tech_level,
        "vix_ret": vix_ret,
        "vix_chg": vix_chg,
        "vix_level": vix_level,
        "vix_z": vix_z,
        "beta60": beta60,
        "corr20": corr20,
        "corr60": corr60,
        "dow": float(dow),
        "mth": float(mth),
    }

    # ---- macro extra din market_regime() extins ----
    rgx = market_regime()
    feats.update(
        {
            "reg_vix_last": rgx.get("reg_vix_last"),
            "reg_vix_z20": rgx.get("reg_vix_z20"),
            "reg_vix_term": rgx.get("reg_vix_term"),
            "reg_dxy_ret5": rgx.get("reg_dxy_ret5"),
            "reg_dxy_z20": rgx.get("reg_dxy_z20"),
            "reg_tnx_last": rgx.get("reg_tnx_last"),
            "reg_tnx_chg5": rgx.get("reg_tnx_chg5"),
            "reg_hyg_lqd_ratio": rgx.get("reg_hyg_lqd_ratio"),
            "reg_hyg_lqd_z20": rgx.get("reg_hyg_lqd_z20"),
            "reg_spy_vol20": rgx.get("reg_spy_vol20"),
            "reg_spy_ret5": rgx.get("reg_spy_ret5"),
            "reg_label": rgx.get("reg_label"),
        }
    )

    # Înlocuiește NaN/inf cu 0.0 ca să nu pice pipeline-ul
    for k, v in feats.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            feats[k] = 0.0
    return feats
