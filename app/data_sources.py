# app/data_sources.py

import os
import requests
import yfinance as yf
from dotenv import load_dotenv

# după linia 6
from .cache import get_cached_response, set_cached_response
import pandas as pd
import numpy as np


load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")


# după linia 12
# Yahoo Finance tickers pentru macro-proxy ce funcționează stabil:
#  - VIX: ^VIX
#  - VIX 3M: ^VIX3M
#  - Dolar Index (futures): DX=F
#  - 10Y yield (×10): ^TNX
#  - Credit spread proxy: HYG/LQD
#  - Benchmark broad market: SPY
_YF_MACRO_TICKERS = ["^VIX", "^VIX3M", "DX=F", "^TNX", "HYG", "LQD", "SPY"]
_YF_CACHE_TTL_MIN = 30  # minute


def fetch_current_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # 1) fast_info
        price = None
        fi = getattr(ticker, "fast_info", {}) or {}
        price = fi.get("lastPrice") or fi.get("last_price") or fi.get("regularMarketPrice")

        # 2) info (uneori există aici)
        if price is None:
            info = ticker.info or {}
            price = info.get("regularMarketPrice") or info.get("currentPrice")

        # 3) history fallback (5 zile ca să prindă ultima închidere validă)
        if price is None:
            hist = ticker.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].dropna().iloc[-1])

        return float(price) if price is not None else None
    except Exception as e:
        print(f"[Yahoo] Error fetching price for {symbol}: {e}")
        return None


def fetch_finnhub_news(symbol, max_items=5):
    try:
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-01-01&to=2025-12-31&token={FINNHUB_API_KEY}"
        r = requests.get(url)
        if r.status_code == 200:
            news = r.json()
            # Sortează cele mai recente și taie la max_items
            news_sorted = sorted(news, key=lambda x: x.get("datetime", 0), reverse=True)
            out = []
            for n in news_sorted[:max_items]:
                out.append(
                    {
                        "title": n.get("headline"),
                        "description": n.get("summary"),
                        "published_at": n.get("datetime"),
                        "url": n.get("url", "#"),
                        "sentiment": n.get("sentiment", None),  # dacă există
                    }
                )
            return out
        else:
            print(f"[Finnhub] Error news for {symbol}: {r.status_code}")
            return []
    except Exception as e:
        print(f"[Finnhub] Exception news {symbol}: {e}")
        return []


def fetch_finnhub_sentiment(symbol):
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={FINNHUB_API_KEY}"
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            return data  # are "companyNewsScore", "sectorAverageBullishPercent" etc.
        else:
            print(f"[Finnhub] Error sentiment for {symbol}: {r.status_code}")
            return None
    except Exception as e:
        print(f"[Finnhub] Exception sentiment {symbol}: {e}")
        return None


# === Macro snapshot (VIX, DXY, ^TNX, HYG/LQD, SPY vol/ret) ===
# Cache simplu (10 minute) ca să nu abuzăm de Yahoo
_MACRO_CACHE = {"t": 0, "val": None}

# === Macro snapshot pentru features.market_regime() ===


def _z(s, win=20):
    s = pd.Series(s).astype(float)
    if len(s) < win:
        return np.nan
    m = s.rolling(win).mean()
    v = s.rolling(win).std(ddof=0)
    return ((s - m) / (v + 1e-12)).iloc[-1]


def _ret(s, k=5):
    s = pd.Series(s).astype(float)
    if len(s) <= k:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-1 - k] - 1.0) * 100.0)


def _last(df, col="Close"):
    try:
        return float(df[col].iloc[-1])
    except Exception:
        return np.nan


# după linia 71
def _yf_cached_download(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Wrapper peste yfinance.download cu cache în APICache (DB).
    Cache key: yf:{ticker}:{period}:{interval}
    TTL: _YF_CACHE_TTL_MIN minute
    """
    cache_key = f"yf:{ticker}:{period}:{interval}"
    cached = get_cached_response("yfinance", cache_key)
    if cached:
        try:
            df = pd.DataFrame.from_records(cached)
            if not df.empty:
                df.index = pd.to_datetime(df["Date"])
                df.drop(
                    columns=[c for c in ["Date"] if c in df.columns],
                    inplace=True,
                    errors="ignore",
                )
                return df
        except Exception:
            pass

    import yfinance as yf

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.reset_index()
        # Persistăm doar coloanele utile + index
        set_cached_response("yfinance", cache_key, df.to_dict(orient="records"))
        df.index = pd.to_datetime(df["Date"])
        df.drop(columns=["Date"], inplace=True, errors="ignore")
        return df
    return pd.DataFrame()


def get_macro_snapshot() -> dict:
    """
    Returnează un pachet compact de macro-features:
      - vix_last, vix_z20
      - vix_term (VIX3M - VIX)
      - dxy_ret5, dxy_z20
      - tnx_last (10y yield %), tnx_chg5
      - hyg_lqd_ratio, hyg_lqd_z20
      - spy_vol20 (realized vol), spy_ret5
      - regime_label in { -1: risk_off, 0: neutral, +1: risk_on }
    Fără web extern; doar Yahoo + cache.
    """
    out = {
        "vix_last": None,
        "vix_z20": None,
        "vix_term": None,
        "dxy_ret5": None,
        "dxy_z20": None,
        "tnx_last": None,
        "tnx_chg5": None,
        "hyg_lqd_ratio": None,
        "hyg_lqd_z20": None,
        "spy_vol20": None,
        "spy_ret5": None,
        "regime_label": None,
    }
    try:
        # === VIX & VIX3M ===
        vix = _yf_cached_download("^VIX", period="12mo", interval="1d")
        vix3 = _yf_cached_download("^VIX3M", period="12mo", interval="1d")
        if not vix.empty:
            vix["ret"] = vix["Close"].pct_change()
            if len(vix) >= 21:
                out["vix_last"] = float(vix["Close"].iloc[-1])
                out["vix_z20"] = float(
                    (vix["Close"].iloc[-1] - vix["Close"].rolling(20).mean().iloc[-1])
                    / (vix["Close"].rolling(20).std(ddof=1).iloc[-1] + 1e-9)
                )
        if (not vix.empty) and (not vix3.empty):
            out["vix_term"] = float(vix3["Close"].iloc[-1] - vix["Close"].iloc[-1])

        # === DXY (DX=F) ===
        dxy = _yf_cached_download("DX=F", period="12mo", interval="1d")
        if not dxy.empty:
            dxy["ret"] = dxy["Close"].pct_change()
            if len(dxy) >= 6:
                out["dxy_ret5"] = float(
                    (dxy["Close"].iloc[-1] / dxy["Close"].iloc[-6] - 1.0) * 100.0
                )
            if len(dxy) >= 21:
                out["dxy_z20"] = float(
                    (dxy["Close"].iloc[-1] - dxy["Close"].rolling(20).mean().iloc[-1])
                    / (dxy["Close"].rolling(20).std(ddof=1).iloc[-1] + 1e-9)
                )

        # === 10y yield (TNX/10) ===
        tnx = _yf_cached_download("^TNX", period="6mo", interval="1d")
        if not tnx.empty:
            tnx_last = float(tnx["Close"].iloc[-1]) / 10.0  # ^TNX este ×10
            out["tnx_last"] = tnx_last
            if len(tnx) >= 6:
                out["tnx_chg5"] = float((tnx["Close"].iloc[-1] - tnx["Close"].iloc[-6]) / 10.0)

        # === Credit spread proxy (HYG/LQD) ===
        hyg = _yf_cached_download("HYG", period="12mo", interval="1d")
        lqd = _yf_cached_download("LQD", period="12mo", interval="1d")
        if (not hyg.empty) and (not lqd.empty):
            # aliniere pe index
            j = (
                hyg[["Close"]]
                .rename(columns={"Close": "HYG"})
                .join(lqd[["Close"]].rename(columns={"Close": "LQD"}), how="inner")
            )
            ratio = j["HYG"] / j["LQD"]
            out["hyg_lqd_ratio"] = float(ratio.iloc[-1])
            if len(ratio) >= 21:
                out["hyg_lqd_z20"] = float(
                    (ratio.iloc[-1] - ratio.rolling(20).mean().iloc[-1])
                    / (ratio.rolling(20).std(ddof=1).iloc[-1] + 1e-9)
                )

        # === SPY realized vol + ret ===
        spy = _yf_cached_download("SPY", period="6mo", interval="1d")
        if not spy.empty:
            r = spy["Close"].pct_change()
            if len(r) >= 21:
                out["spy_vol20"] = float(
                    np.sqrt(252) * r.rolling(20).std(ddof=1).iloc[-1] * 100.0
                )  # anualizat %
            if len(spy) >= 6:
                out["spy_ret5"] = float(
                    (spy["Close"].iloc[-1] / spy["Close"].iloc[-6] - 1.0) * 100.0
                )

        # === Regime euristic ===
        vix_term = out["vix_term"]
        vix_z = out["vix_z20"]
        hy_z = out["hyg_lqd_z20"]
        dxy_z = out["dxy_z20"]
        regime = 0
        # reguli simple: term structure în contango + credit ok + dolar nu e super bid -> risk_on
        if (
            (vix_term is not None and vix_term > 0.8)
            and (vix_z is not None and vix_z < 0.5)
            and (hy_z is not None and hy_z > -0.3)
            and (dxy_z is not None and dxy_z < 0.8)
        ):
            regime = +1
        # invers: backwardation + vix ridicat + hy slăbit + dolar tare -> risk_off
        if (
            (vix_term is not None and vix_term < -0.5)
            or (vix_z is not None and vix_z > 1.0)
            or (hy_z is not None and hy_z < -1.0)
            or (dxy_z is not None and dxy_z > 1.2)
        ):
            regime = -1
        out["regime_label"] = int(regime)

    except Exception as e:
        print(f"[macro] snapshot fail: {e}")

    return out
