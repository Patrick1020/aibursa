# app/backtest_v2.py
from __future__ import annotations

import os
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from datetime import datetime, timedelta, timezone, date

import numpy as np
import pandas as pd
import yfinance as yf

from .database import SessionLocal  # type: ignore
from .models import StockPrediction  # type: ignore
from .logger import logger  # type: ignore


# ========= Config & paths =========
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
REPORT_HTML = os.path.join(STATIC_DIR, "backtest_report.html")
REPORT_CSV = os.path.join(STATIC_DIR, "backtest_pro.csv")
IMG_RELIAB = os.path.join(STATIC_DIR, "bt_reliability.png")
IMG_DIST = os.path.join(STATIC_DIR, "bt_dist.png")
IMG_EQUITY = os.path.join(STATIC_DIR, "bt_equity.png")
TUNING_CSV = os.path.join(STATIC_DIR, "backtest_tuning.csv")

# Silențiem FutureWarning-urile cunoscute (ex: cast de Series la float).
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ========= Helpers =========
def _to_float(x) -> Optional[float]:
    """
    Întoarce float din scalari pandas/numpy sau din Series cu un element.
    """
    try:
        if isinstance(x, pd.Series):
            if x.empty:
                return None
            x = x.iloc[-1]
        if hasattr(x, "item"):  # pandas/numpy scalar
            x = x.item()
        return float(x)
    except Exception:
        return None


def _safe_float(x, default=None):
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _last_yf_close_date(symbol_for_probe: str = "SPY") -> date:
    """
    Întoarce cea mai recentă dată de închidere (date, tz-agnostic) disponibilă pe Yahoo.
    Fallback: azi (UTC).
    """
    try:
        h = yf.Ticker(symbol_for_probe).history(period="10d", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return datetime.now(timezone.utc).date()
        return h.index[-1].date()
    except Exception:
        return datetime.now(timezone.utc).date()


# ========= Yahoo price cache =========
@dataclass
class _HistoryWindow:
    start: date
    end: date
    df: (
        pd.DataFrame
    )  # Index tipic DatetimeIndex (fără tz), coloane: Open/High/Low/Close/Adj Close/Volume


class YahooPriceCache:
    """
    Cache minimal pe proces pentru a evita descarcări repetate ale acelorași ferestre.
    Cheia: (symbol)
    Valoarea: listă de ferestre ordonate crescător după start.
    """

    def __init__(self, max_windows_per_symbol: int = 8) -> None:
        self._store: Dict[str, List[_HistoryWindow]] = {}
        self._max = max_windows_per_symbol

    def get_close_on_or_before(
        self, symbol: str, target: date, back_days: int = 30
    ) -> Optional[float]:
        """
        Întoarce Close din ziua `target`; dacă e nelucrătoare/holiday, ia ultimul Close <= target.
        """
        # Încearcă o fereastră existentă care acoperă target-back_days .. target
        start = target - timedelta(days=back_days)
        end = target

        df = self._get_or_fetch(symbol, start, end)
        if df is None or df.empty:
            return None

        # Asigurăm index fără tz, comparabil cu pd.Timestamp(date)
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)

        sel = df.loc[: pd.Timestamp(target)]
        if sel.empty:
            return None
        return _to_float(sel["Close"].iloc[-1])

    def window_close_between(
        self, symbol: str, start_d: date, end_d: date
    ) -> Optional[pd.DataFrame]:
        """
        Returnează istoricul (1d) dintre start_d și end_d (capete incluse), din cache sau Yahoo.
        """
        return self._get_or_fetch(symbol, start_d, end_d)

    # ----- intern -----
    def _get_or_fetch(self, symbol: str, start_d: date, end_d: date) -> Optional[pd.DataFrame]:
        win = self._find_covering_window(symbol, start_d, end_d)
        if win:
            return win.df

        # Nu avem o fereastră care să acopere, descarcă o mică marjă.
        fetch_start = start_d - timedelta(days=2)  # safety
        fetch_end = end_d + timedelta(days=2)

        # Yahoo 'end' este exclusiv la download, cerem +1 zi
        start_dt = datetime.combine(fetch_start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(
            fetch_end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
        )

        # Clamp la ultima zi existentă în Yahoo
        last_close = _last_yf_close_date()
        if fetch_end > last_close:
            end_dt = datetime.combine(
                last_close + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
            )

        hist = None
        for attempt in range(2):
            try:
                hist = yf.download(
                    symbol,
                    start=start_dt,
                    end=end_dt,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if hist is not None and not hist.empty:
                    break
            except Exception as e:
                if attempt == 0:
                    continue
                logger.error(f"[YF] download failed for {symbol}: {e}")

        if hist is None or hist.empty:
            return None

        # Index fără tz
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_convert(None)

        self._insert_window(symbol, _HistoryWindow(start=start_d, end=end_d, df=hist))
        return hist

    def _find_covering_window(
        self, symbol: str, start_d: date, end_d: date
    ) -> Optional[_HistoryWindow]:
        wins = self._store.get(symbol, [])
        for w in wins:
            if w.start <= start_d and w.end >= end_d:
                return w
        return None

    def _insert_window(self, symbol: str, window: _HistoryWindow) -> None:
        wins = self._store.setdefault(symbol, [])
        wins.append(window)
        wins.sort(key=lambda w: w.start)
        if len(wins) > self._max:
            # păstrăm cele mai recente ferestre
            self._store[symbol] = wins[-self._max :]


# Instanță globală simplă (în proces) – OK pentru script/worker unici
PRICE_CACHE = YahooPriceCache()


class Backtester:
    """
    Backtester configurabil care:
      - colectează predicții din DB,
      - calculează realizatul pe orizont,
      - scrie CSV cu (pred, prob, bandă, real),
      - produce grafice și raport HTML,
      - poate rula grid-tuning pentru praguri de strategie.
    """

    def __init__(
        self,
        horizon_days: int = 7,
        thr_abs_pct: float = 1.0,
        thr_prob: float = 0.60,
        limit: int = 2000,
        symbols: Optional[List[str]] = None,
    ) -> None:
        self.horizon_days = int(horizon_days)
        self.thr_abs_pct = float(thr_abs_pct)
        self.thr_prob = float(thr_prob)
        self.limit = int(limit)
        self.symbols = symbols

        # rezultat (opțional) din auto-tuning pentru a-l folosi în raport
        self.best_tuned: Optional[dict] = None

        _ensure_dir(REPORT_HTML)
        _ensure_dir(REPORT_CSV)
        _ensure_dir(IMG_RELIAB)
        _ensure_dir(IMG_DIST)
        _ensure_dir(IMG_EQUITY)
        _ensure_dir(TUNING_CSV)

    # ----- Data access -----
    def collect_predictions(self) -> List[StockPrediction]:
        """
        Ultimele N predicții (toate simbolurile sau subset), cronologic ascendent.
        """
        db = SessionLocal()
        try:
            q = db.query(StockPrediction).order_by(StockPrediction.published_at.asc())
            if self.symbols:
                q = q.filter(StockPrediction.symbol.in_(self.symbols))
            rows: List[StockPrediction] = q.limit(self.limit).all()
            return rows
        finally:
            db.close()

    # ----- Market data -----
    def _close_on_or_before(
        self, symbol: str, target: date, back_days: int = 30
    ) -> Optional[float]:
        return PRICE_CACHE.get_close_on_or_before(symbol, target, back_days=back_days)

    def _close_around(self, symbol: str, dt_utc: datetime) -> Optional[float]:
        """
        Close în ziua 'dt_utc' (UTC) sau cel imediat următor disponibil în fereastra cerută,
        clamped la ultima zi disponibilă în Yahoo.
        """
        dt_utc = _to_utc(dt_utc)
        assert dt_utc is not None
        target = dt_utc.date()

        last_close = _last_yf_close_date()
        end_d = min(target, last_close)
        start_d = end_d - timedelta(days=7)

        df = PRICE_CACHE.window_close_between(symbol, start_d, end_d)
        if df is None or df.empty:
            return None

        # alege exact ziua dacă există, altfel primul >= target
        for idx, row in df.iterrows():
            if idx.date() >= target and "Close" in row:
                val = _to_float(row["Close"])
                if val is not None and math.isfinite(val):
                    return val

        # fallback: ultimul din fereastră
        if "Close" in df.columns:
            return _to_float(df["Close"].iloc[-1])
        return None

    def _realized_change_pct(self, symbol: str, t0_utc: datetime) -> Optional[float]:
        """
        % realizat între close la/după t0 și close la/după t0 + horizon_days.
        """
        try:
            p0 = self._close_around(symbol, t0_utc)
            p1 = self._close_around(symbol, t0_utc + timedelta(days=self.horizon_days))
            if p0 and p1:
                return (p1 - p0) / p0 * 100.0
            return None
        except Exception:
            return None

    # ----- Core evaluation -----
    def evaluate(self, rows: List[StockPrediction]) -> Dict[str, Optional[float]]:
        """
        Rulează evaluarea + scriere CSV + grafice + raport HTML.
        Returnează metrice sumarizate.
        """
        # 1) filtrăm predicțiile prea recente (nu avem încă p1)
        last_close_d = _last_yf_close_date()
        usable: List[StockPrediction] = []
        too_new = 0
        for r in rows:
            pub = getattr(r, "published_at", None)
            if pub is None:
                continue
            pub_d = pub.date() if hasattr(pub, "date") else pub
            if pub_d and (pub_d + timedelta(days=self.horizon_days) <= last_close_d):
                usable.append(r)
            else:
                too_new += 1

        if too_new:
            logger.info(
                f"[BT] Skipped {too_new} predictions (too recent for horizon={self.horizon_days}d)."
            )

        if not usable:
            raise RuntimeError(
                "Nu există predicții suficiente/vechi pentru backtest la acest orizont."
            )

        # 2) parcurgem rândurile și construim CSV-ul (și seturi pt metrice)
        recs: List[Dict] = []
        y_true_up: List[int] = []
        p_up_list: List[float] = []
        err_abs: List[float] = []
        cover_hits: List[int] = []
        ret_strat: List[float] = []

        for r in usable:
            sym = r.symbol or "?"
            t0 = _to_utc(r.published_at)
            if t0 is None:
                continue

            pct_pred = _safe_float(
                getattr(r, "final_percent", None), _safe_float(getattr(r, "estimated_change", None))
            )
            p_final = _safe_float(
                getattr(r, "final_probability", None), _safe_float(getattr(r, "probability", None))
            )
            ci_low = _safe_float(
                getattr(r, "ci_low_pct", None), _safe_float(getattr(r, "p20", None))
            )
            ci_high = _safe_float(
                getattr(r, "ci_high_pct", None), _safe_float(getattr(r, "p80", None))
            )

            real_pct = self._realized_change_pct(sym, t0)
            recs.append(
                {
                    "symbol": sym,
                    "date": t0.strftime("%Y-%m-%d"),
                    "pct_pred": pct_pred,
                    "p_final": p_final,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "real_pct": real_pct,
                }
            )

            if (real_pct is not None) and (pct_pred is not None):
                err_abs.append(abs(pct_pred - real_pct))

            if real_pct is not None:
                y_true_up.append(1 if real_pct >= 0 else 0)
                if (p_final is not None) and (pct_pred is not None):
                    p = float(p_final) / 100.0
                    p = max(0.0, min(1.0, p))
                    if pct_pred < 0:
                        p = 1.0 - p
                    p_up_list.append(p)

            if (real_pct is not None) and (ci_low is not None) and (ci_high is not None):
                cover_hits.append(1 if (ci_low <= real_pct <= ci_high) else 0)

            if (real_pct is not None) and (pct_pred is not None) and (p_final is not None):
                if (abs(pct_pred) >= self.thr_abs_pct) and (p_final >= self.thr_prob * 100.0):
                    sign = 1.0 if pct_pred >= 0 else -1.0
                    ret_strat.append(sign * real_pct)

        # 3) metrice
        mae = float(np.mean(err_abs)) if err_abs else None
        hit_rate = None
        if recs:
            mask = [(r["real_pct"] is not None and r["pct_pred"] is not None) for r in recs]
            if any(mask):
                real_s = np.array([r["real_pct"] for r, m in zip(recs, mask) if m], dtype=float)
                pred_s = np.array([r["pct_pred"] for r, m in zip(recs, mask) if m], dtype=float)
                ok = np.sign(real_s) == np.sign(pred_s)
                hit_rate = float(np.mean(ok))

        brier = self._brier_score(y_true_up, p_up_list) if (y_true_up and p_up_list) else None
        coverage = float(np.mean(cover_hits)) if cover_hits else None
        avg_ret, n_tr = (None, 0)
        if ret_strat:
            avg_ret = float(np.mean(ret_strat))
            n_tr = int(len(ret_strat))

        # 3.5) auto-tuning praguri pe recs -> optimizare grilă
        # obiectiv: Sharpe (implict); poți folosi "mean" sau "total"
        try:
            self.best_tuned = self._grid_tune_thresholds(
                recs,
                abs_grid=[0.5, 1.0, 1.5, 2.0, 3.0],  # extinde cum vrei
                prob_grid=[0.55, 0.60, 0.65, 0.70],  # extinde cum vrei
                objective="sharpe",
            )
        except Exception as e:
            logger.warning(f"[BT] grid tuning failed: {e}")
            self.best_tuned = None

        # 4) scriem CSV
        _ensure_dir(REPORT_CSV)
        with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "symbol",
                    "date",
                    "pct_pred",
                    "p_final",
                    "ci_low",
                    "ci_high",
                    "real_pct",
                ],
            )
            w.writeheader()
            for row in recs:
                w.writerow(row)
        logger.info(f"[BT] CSV -> {REPORT_CSV}")

        # 5) grafice
        self._save_plots(y_true_up, p_up_list, err_abs, ret_strat)

        # 6) HTML raport (+ tuning grid)
        self._write_html_report(
            mae, hit_rate, brier, coverage, avg_ret, n_tr, tuned=self.best_tuned
        )

        self._append_tuning_summary()  # best effort

        return {
            "mae": mae,
            "hit_rate": hit_rate,
            "brier": brier,
            "coverage": coverage,
            "avg_return": avg_ret,
            "trades": n_tr,
        }

    # ----- Metrics helpers -----
    @staticmethod
    def _brier_score(y_true: List[int], p_up: List[float]) -> Optional[float]:
        if not y_true or not p_up or len(y_true) != len(p_up):
            return None
        err = [(float(t) - float(p)) ** 2 for t, p in zip(y_true, p_up)]
        return float(np.mean(err))

    @staticmethod
    def _calibration_bins(
        y_true: List[int], p_up: List[float], bins: int = 10
    ) -> List[Tuple[float, float, int]]:
        edges = np.linspace(0.0, 1.0, bins + 1)
        out: List[Tuple[float, float, int]] = []
        y_true_a = np.asarray(y_true, dtype=float)
        p_up_a = np.asarray(p_up, dtype=float)
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (
                (p_up_a >= lo) & (p_up_a < hi) if b < bins - 1 else (p_up_a >= lo) & (p_up_a <= hi)
            )
            sel = np.where(mask)[0]
            if sel.size == 0:
                out.append(((lo + hi) / 2.0, np.nan, 0))
            else:
                out.append(((lo + hi) / 2.0, float(np.mean(y_true_a[sel])), int(sel.size)))
        return out

    # ----- Plots & HTML -----
    def _save_plots(
        self,
        y_true_up: List[int],
        p_up_list: List[float],
        err_abs: List[float],
        ret_strat: List[float],
    ) -> None:
        try:
            import matplotlib.pyplot as plt

            if y_true_up and p_up_list:
                bins = self._calibration_bins(y_true_up, p_up_list, bins=10)
                x = [b[0] for b in bins if b[2] > 0]
                y = [b[1] for b in bins if b[2] > 0]
                plt.figure(figsize=(5, 4))
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.plot(x, y, marker="o")
                plt.xlabel("Predicted prob up")
                plt.ylabel("Empirical up rate")
                plt.title("Reliability")
                plt.tight_layout()
                plt.savefig(IMG_RELIAB, dpi=120)
                plt.close()

            if err_abs:
                plt.figure(figsize=(5, 4))
                plt.hist(err_abs, bins=30)
                plt.xlabel("|pred - real| (%)")
                plt.ylabel("Count")
                plt.title("Absolute Error Distribution")
                plt.tight_layout()
                plt.savefig(IMG_DIST, dpi=120)
                plt.close()

            if ret_strat:
                eq = np.cumsum(ret_strat)
                plt.figure(figsize=(6, 3.5))
                plt.plot(eq)
                plt.title("Equity Curve (toy strategy)")
                plt.xlabel("Trades")
                plt.ylabel("Cum. return (pct)")
                plt.tight_layout()
                plt.savefig(IMG_EQUITY, dpi=120)
                plt.close()
        except Exception as e:
            logger.warning(f"[BT] plotting failed: {e}")

    def _write_html_report(
        self,
        mae: Optional[float],
        hit_rate: Optional[float],
        brier: Optional[float],
        coverage: Optional[float],
        avg_ret: Optional[float],
        n_trades: int,
        tuned: Optional[dict] = None,
    ) -> None:

        def _fmt(v, pct=False):
            if v is None:
                return "-"
            return f"{v*100.0:.2f}%" if pct else f"{v:.2f}"

        html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Backtest Pro</title>
<style>
body {{ background:#0f1629; color:#f0f2fa; font-family:Segoe UI,Arial,sans-serif; }}
.wrap {{ max-width: 1100px; margin: 24px auto; }}
.card {{ background:#171e2c; border-radius:12px; padding:18px 20px; margin-bottom:16px; }}
h1,h2 {{ color:#6cd4ff; }}
.grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; }}
.kpi {{ background:#10172a; border:1px solid #22304b; border-radius:10px; padding:12px; }}
.kpi b {{ color:#9fe8ff; display:block; margin-bottom:6px; }}
img {{ max-width:100%; height:auto; border-radius:10px; border:1px solid #22304b; }}
a.btn {{ display:inline-block; background:#23305b; color:#6cd4ff; padding:9px 14px; border-radius:8px; text-decoration:none; margin-right:8px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Backtest Pro (horizon {self.horizon_days}d)</h1>

  <div class="card grid">
    <div class="kpi"><b>MAE (pct)</b><span>{_fmt(mae)}</span></div>
    <div class="kpi"><b>Hit rate</b><span>{_fmt(hit_rate, pct=True)}</span></div>
    <div class="kpi"><b>Brier</b><span>{_fmt(brier)}</span></div>
    <div class="kpi"><b>CI coverage</b><span>{_fmt(coverage, pct=True)}</span></div>
  </div>

  <div class="card">
    <a class="btn" href="/static/{os.path.basename(REPORT_CSV)}" target="_blank">Download CSV</a>
    <a class="btn" href="/" >Back to Dashboard</a>
  </div>

  <div class="card">
    <h2>Reliability</h2>
    <img src="/static/{os.path.basename(IMG_RELIAB)}" alt="reliability"/>
  </div>

  <div class="card">
    <h2>Error Distribution</h2>
    <img src="/static/{os.path.basename(IMG_DIST)}" alt="dist"/>
  </div>

  <div class="card">
    <h2>Equity Curve (toy)</h2>
    <img src="/static/{os.path.basename(IMG_EQUITY)}" alt="equity"/>
  </div>
</div>
</body>
</html>
"""
        with open(REPORT_HTML, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"[BT] Report -> {REPORT_HTML}")

        if tuned:
            try:
                t_abs = tuned.get("thr_abs_pct", None)
                t_prob = tuned.get("thr_prob", None)
                t_trades = tuned.get("n_trades", 0)
                t_avg = tuned.get("avg_ret", None)
                t_total = tuned.get("total_ret", None)
                t_hit = tuned.get("hit_rate", None)
                t_sharpe = tuned.get("sharpe", None)

                def _fmt_opt(v, pct=False):
                    if v is None:
                        return "-"
                    return f"{v*100.0:.2f}%" if pct else f"{v:.2f}"

                html_rec = f"""
  <div class="card">
    <h2>Recommended thresholds</h2>
    <p>
      <b>thr_abs_pct</b> = {t_abs},
      <b>thr_prob</b> = {t_prob}
      <span style="opacity:.8">(prob în [0..1])</span>
    </p>
    <p>
      Trades = {int(t_trades)} · Avg return = {_fmt_opt(t_avg)}% · Total return = {_fmt_opt(t_total)}% ·
      Hit rate = {_fmt_opt(t_hit, pct=True)} · Sharpe = {_fmt_opt(t_sharpe)}
    </p>
  </div>
"""
                html = html.replace(
                    '<div class="card">\n    <h2>Reliability</h2>',
                    html_rec + '\n  <div class="card">\n    <h2>Reliability</h2>',
                )
            except Exception as e:
                logger.warning(f"[BT] render tuned thresholds failed: {e}")

    # ----- Tuning -----
    def _append_tuning_summary(self) -> None:
        """
        Rulează optimize_thresholds() pe CSV-ul curent, apoi inserează un card de rezumat în HTML.
        Best-effort; dacă nu există CSV sau are prea puține date, doar iese.
        """
        try:
            if not os.path.exists(REPORT_CSV):
                return
            df = pd.read_csv(REPORT_CSV)
            if df.empty:
                return
        except Exception:
            return

        grid_abs = (0.5, 1.0, 1.5, 2.0)
        grid_prob = (0.55, 0.60, 0.65)

        try:
            tuning = []
            for ta in grid_abs:
                for tp in grid_prob:
                    m = self._metrics_from_df(df, thr_abs_pct=ta, thr_prob=tp)
                    tuning.append({"thr_abs_pct": ta, "thr_prob": tp, **m})
            tdf = pd.DataFrame(tuning)
            tdf.to_csv(TUNING_CSV, index=False)

            # selectăm best: maximăm hit_rate, apoi trades, apoi avg_return
            best = (
                tdf.dropna(subset=["hit_rate"])
                .sort_values(["hit_rate", "trades", "avg_return"], ascending=[False, False, False])
                .head(1)
            )
            best_row = best.to_dict(orient="records")[0] if not best.empty else None
        except Exception as e:
            logger.warning(f"[BT] tuning failed: {e}")
            best_row = None

        # injectăm în HTML
        try:
            with open(REPORT_HTML, "r", encoding="utf-8") as f:
                html = f.read()
            add = "<div class='card'><h2>Tuning praguri</h2>"
            add += f"<p><a class='btn' href='/static/{os.path.basename(TUNING_CSV)}' target='_blank'>Download tuning CSV</a></p>"
            if best_row:
                avg_ret = best_row.get("avg_return", None)
                avg_str = "-" if avg_ret is None else f"{avg_ret:.2f}%"
                add += (
                    "<p><b>Best:</b> "
                    f"thr_abs_pct={best_row['thr_abs_pct']}, "
                    f"thr_prob={best_row['thr_prob']}, "
                    f"hit_rate={best_row['hit_rate']:.2%}, "
                    f"MAE={best_row['mae']:.2f}, "
                    f"trades={int(best_row['trades'])}, "
                    f"avg_return={avg_str}</p>"
                )
            add += "</div>"
            html = html.replace("</div>\n</body>", f"{add}\n</div>\n</body>")
            with open(REPORT_HTML, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:
            logger.warning(f"[BT] inject tuning HTML failed: {e}")

    @staticmethod
    def _metrics_from_df(
        df: pd.DataFrame, thr_abs_pct: float, thr_prob: float
    ) -> Dict[str, Optional[float]]:
        d = df.copy()
        # MAE
        m = (~d["real_pct"].isna()) & (~d["pct_pred"].isna())
        if not m.any():
            return {
                "mae": None,
                "hit_rate": None,
                "brier": None,
                "coverage": None,
                "avg_return": None,
                "trades": 0,
            }

        dd = d[m]
        mae = float((dd["pct_pred"] - dd["real_pct"]).abs().mean())

        # Hit rate
        hit_rate = float((np.sign(dd["pct_pred"]) == np.sign(dd["real_pct"])).mean())

        # Brier
        dprob = dd.dropna(subset=["p_final"])
        if not dprob.empty:
            p = np.clip(dprob["p_final"].astype(float) / 100.0, 0.0, 1.0)
            p_up = p.where(dd["pct_pred"] >= 0, 1.0 - p)
            y_up = (dprob["real_pct"] >= 0).astype(int).to_numpy()
            brier = float(np.mean((p_up.to_numpy() - y_up) ** 2))
        else:
            brier = None

        # Coverage
        dci = dd.dropna(subset=["ci_low", "ci_high"])
        coverage = (
            float(((dci["real_pct"] >= dci["ci_low"]) & (dci["real_pct"] <= dci["ci_high"])).mean())
            if not dci.empty
            else None
        )

        # Strategie
        sel = dd.dropna(subset=["p_final"]).copy()
        cond = (sel["pct_pred"].abs() >= thr_abs_pct) & (sel["p_final"] >= thr_prob * 100.0)
        trades = sel[cond]
        if trades.empty:
            avg_return = None
            n_tr = 0
        else:
            signs = np.where(trades["pct_pred"] >= 0, 1.0, -1.0)
            avg_return = float((signs * trades["real_pct"]).mean())
            n_tr = int(len(trades))

        return {
            "mae": mae,
            "hit_rate": hit_rate,
            "brier": brier,
            "coverage": coverage,
            "avg_return": avg_return,
            "trades": n_tr,
        }

    def _grid_tune_thresholds(
        self,
        recs: list[dict],
        abs_grid: list[float] | None = None,
        prob_grid: list[float] | None = None,
        objective: str = "sharpe",
    ) -> dict | None:
        """
        Grid-search pe (thr_abs_pct, thr_prob) folosind 'recs' (rândurile CSV deja pregătite în evaluate()).
        - thr_abs_pct  în procente pe |pct_pred|
        - thr_prob     în [0..1] (ex: 0.60 => 60%)
        - objective:   'sharpe' (default) sau 'mean' / 'total'

        Returnează cel mai bun candidate dict:
        {
          'thr_abs_pct': float,
          'thr_prob': float,
          'n_trades': int,
          'avg_ret': float,
          'total_ret': float,
          'hit_rate': float | None,
          'sharpe': float,
          'score': float
        }
        """
        import numpy as np

        if not recs:
            return None

        if abs_grid is None:
            abs_grid = [0.5, 1.0, 1.5, 2.0, 3.0]  # extensibil
        if prob_grid is None:
            prob_grid = [0.55, 0.60, 0.65, 0.70, 0.75]

        def _score(trades: list[float], hits: list[int]) -> tuple[float, float, float, float]:
            if not trades:
                return 0.0, 0.0, 0.0, 0.0
            arr = np.asarray(trades, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            sharpe = (mean / std) if std > 1e-8 else 0.0
            total = float(np.sum(arr))
            hit = float(np.mean(hits)) if hits else 0.0
            if objective == "mean":
                use = mean
            elif objective == "total":
                use = total
            else:
                use = sharpe
            return use, mean, total, hit

        best = None
        # explorare grilă
        for thr_abs in abs_grid:
            for thr_prob in prob_grid:
                trades = []
                hits = []
                for r in recs:
                    pct_pred = _safe_float(r.get("pct_pred"))
                    p_final = _safe_float(r.get("p_final"))
                    real_pct = _safe_float(r.get("real_pct"))
                    if (pct_pred is None) or (p_final is None) or (real_pct is None):
                        continue
                    # p_final în CSV e în % (0..100), thr_prob e în [0..1]
                    if (abs(pct_pred) >= thr_abs) and (p_final >= thr_prob * 100.0):
                        sign = 1.0 if pct_pred >= 0 else -1.0
                        ret = sign * real_pct
                        trades.append(ret)
                        hits.append(1 if ret >= 0 else 0)

                score, mean, total, hit = _score(trades, hits)
                cand = {
                    "thr_abs_pct": thr_abs,
                    "thr_prob": thr_prob,
                    "n_trades": len(trades),
                    "avg_ret": mean,
                    "total_ret": total,
                    "hit_rate": hit,
                    "sharpe": (
                        (mean / (np.std(trades, ddof=1) if len(trades) > 1 else 1.0))
                        if trades
                        else 0.0
                    ),
                    "score": score,
                }
                if (best is None) or (cand["score"] > best["score"]):
                    best = cand
        return best


# ========= Public API =========


def generate_report(
    horizon_days: int = 7,
    limit: int = 2000,
    symbols: Optional[List[str]] = None,
    thr_abs_pct: float = 1.0,
    thr_prob: float = 0.60,
) -> str:
    """
    Punctul public folosit de route-ul din FastAPI.
    Creează raportul, returnează calea către HTML.
    """
    bt = Backtester(
        horizon_days=horizon_days,
        thr_abs_pct=thr_abs_pct,
        thr_prob=thr_prob,
        limit=limit,
        symbols=symbols,
    )
    rows = bt.collect_predictions()
    _ = bt.evaluate(rows)
    return REPORT_HTML


# ========= CLI =========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Pro (enterprise)")
    parser.add_argument(
        "--horizon", type=int, default=7, help="Horizon in calendar days (default: 7)"
    )
    parser.add_argument(
        "--limit", type=int, default=2000, help="Max number of predictions to evaluate"
    )
    parser.add_argument(
        "--symbols", type=str, default="", help="Comma-separated symbols to filter (optional)"
    )
    parser.add_argument(
        "--thr-abs", type=float, default=1.0, help="Abs pct threshold for toy strategy entry"
    )
    parser.add_argument(
        "--thr-prob", type=float, default=0.60, help="Probability threshold (0..1) for toy strategy"
    )

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    out = generate_report(
        horizon_days=args.horizon,
        limit=args.limit,
        symbols=symbols,
        thr_abs_pct=args.thr_abs,
        thr_prob=args.thr_prob,
    )
    print("Raport generat la:", out)
