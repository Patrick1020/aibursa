import os
import re
import time
import json
import requests
import openai
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from dateutil.parser import parse as parse_date

# --- Bloc A: OpenAI v1 client & erori ---
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    BadRequestError,
    Timeout,
)


from .models import StockPrediction, HistoricalImpact
from .database import SessionLocal, get_db
from .logger import logger
from .cache import get_cached_response, set_cached_response
from .news_ranking import rank_news, summarize_news
from .data_sources import fetch_finnhub_news, fetch_current_price
from .features import daily_vol_pct, atr_pct, days_to_next_earnings, market_regime

# opțional: dacă folosești fallback în _live_features
from .features import build_features_for_symbol  # dacă ai funcția în features.py

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# Cheia e luată din variabila de mediu OPENAI_API_KEY
_openai_client = OpenAI()

# === Where trained models live (same names as trainer) ===
MODEL_DIR = Path("models_store")

# Handlers globale (sunt None dacă lipsesc fișierele)
_REG = _CLF = _CAL = _Q20 = _Q80 = None
_FEAT_NAMES = None

_FEATS_LIST = None
try:
    _FEATS_LIST = json.load(open(Path("models_store") / "features.json", "r"))
except Exception:
    _FEATS_LIST = None


def _load_models_once():
    """Încarcă o singură dată modelele și lista de feature-uri, dacă există."""
    global _REG, _CLF, _CAL, _Q20, _Q80, _FEAT_NAMES

    try:
        _REG = load(MODEL_DIR / "regressor.pkl")
    except Exception:
        _REG = None
    try:
        _CLF = load(MODEL_DIR / "classifier.pkl")
    except Exception:
        _CLF = None
    try:
        _CAL = load(MODEL_DIR / "calibrator.pkl")
    except Exception:
        _CAL = None
    try:
        _Q20 = load(MODEL_DIR / "reg_q20.pkl")
    except Exception:
        _Q20 = None
    try:
        _Q80 = load(MODEL_DIR / "reg_q80.pkl")
    except Exception:
        _Q80 = None

    # lista de feature-uri (ordinea trebuie să fie identică la runtime)
    try:
        import json

        _FEAT_NAMES = json.loads((MODEL_DIR / "features.json").read_text())
    except Exception:
        _FEAT_NAMES = None


# cheamă loader-ul la import
_load_models_once()


logger.info(
    f"[MODELS] reg={_REG is not None} clf={_CLF is not None} cal={_CAL is not None} q20={_Q20 is not None} q80={_Q80 is not None}"
)


# --- Helper: aplică calibratorul indiferent de API (transform / predict_proba) ---
def _apply_calibrator(p_pct):
    """
    Primește o probabilitate fie în [0..100], fie în [0..1] și o calibrează
    cu modelul salvat în _CAL (Isotonic/Platt/altul). Returnează % în [0..100]
    sau valoarea inițială dacă nu se poate calibra.
    """
    if p_pct is None or _CAL is None:
        return p_pct
    try:
        import numpy as np

        x = float(p_pct)
        # normalizare la [0..1] pentru input către calibrator
        x_arr = np.asarray([x / 100.0 if x > 1.0 else x], dtype=float)  # shape (1,)

        if hasattr(_CAL, "predict"):
            y = _CAL.predict(x_arr)  # IsotonicRegression are predict
        elif hasattr(_CAL, "transform"):
            y = _CAL.transform(x_arr)  # unele clase folosesc transform
        elif hasattr(_CAL, "predict_proba"):
            y = _CAL.predict_proba(x_arr)  # rar, dar acoperim
            y = y[:, 1]
        else:
            return p_pct

        p = float(y[0])
        # adu la % dacă e în [0..1]
        if 0.0 <= p <= 1.0:
            p *= 100.0

        # limitează la [0..100]
        if p < 0.0:
            p = 0.0
        if p > 100.0:
            p = 100.0
        return p
    except Exception as e:
        logger.warning(f"[CAL] failed: {e}")
        return p_pct


def _apply_policy_rules(symbol, pct_final, p_final, rr, earn_days, sigma_daily):
    """
    Reguli finale de bun-simț:
      - dacă prob < 55% => recomandare HOLD
      - dacă |pct| > 5 * sigma_daily * sqrt(7) și NU avem earnings < 10 zile => taie la cap și reduce prob
      - dacă reward_to_risk < 1.0 => nu recomandăm BUY (preferăm HOLD / SELL după semn)
    """
    rec_override = None

    if p_final is not None and p_final < 55.0:
        rec_override = "Hold"

    if pct_final is not None and sigma_daily:
        cap = 5.0 * sigma_daily * np.sqrt(7.0)
        if abs(pct_final) > cap and (earn_days is None or earn_days > 10):
            pct_final = float(np.sign(pct_final) * cap)
            if p_final is not None:
                p_final = float(max(0.0, min(100.0, p_final * 0.8)))

    if rr is not None and rr < 1.0 and rec_override != "Hold":
        rec_override = "Sell" if (pct_final is not None and pct_final < 0) else "Hold"

    return pct_final, p_final, rec_override


def _live_features(symbol: str):
    """
    Construiește vectorul X (1, n_features) pt. modelele statistice.
    Încearcă make_live_features; dacă nu e disponibil, folosește build_features_for_symbol(...).iloc[-1].
    """
    feat_map = None

    # 1) Path preferat: make_live_features -> dict {feat_name: value}
    try:
        from .features import make_live_features

        feat_map = make_live_features(symbol)
    except Exception:
        feat_map = None

    # 2) Fallback: build_features_for_symbol -> ia ultimul rând ca dict
    if not feat_map:
        try:
            df = build_features_for_symbol(symbol)
            if df is not None and len(df) > 0:
                drop_cols = {"target_reg", "target_clf", "date", "symbol"}
                cols = [c for c in df.columns if c not in drop_cols]
                feat_map = df[cols].iloc[-1].to_dict()
        except Exception:
            feat_map = None

    if not feat_map or not _FEAT_NAMES:
        return None

    row = [float(feat_map.get(name, 0.0)) for name in _FEAT_NAMES]
    x = np.array(row, dtype=float).reshape(1, -1)
    if not np.isfinite(x).all():
        return None
    return x


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
DB_PATH = "stock_analysis.db"


# după ce ai dict-ul feats și lista feat_names (din models_store/features.json)
def _to_model_frame(feats: dict, feat_names: list | None):
    """Construiește un DataFrame 1xN cu ordinea corectă a coloanelor."""
    cols = feat_names or sorted(feats.keys())
    row = {c: feats.get(c, 0.0) for c in cols}
    return pd.DataFrame([row])


def normalize_date(val):
    if isinstance(val, int):
        # Presupunem că e timestamp (secunde) și îl convertim la string ISO
        return datetime.fromtimestamp(val, timezone.utc).isoformat()
    elif isinstance(val, str):
        try:
            return parse_date(val).isoformat()
        except Exception:
            return val
    return ""


def _as_aware_utc(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fetch_all_news_for_symbol(symbol, page_size=10, finnhub_items=5):
    newsapi_news = fetch_news_for_symbol(symbol, page_size=page_size)
    finnhub_news = fetch_finnhub_news(symbol, max_items=finnhub_items)
    all_news = (newsapi_news or []) + (finnhub_news or [])
    # Sortează după dată (dacă ai published_at ca string sau timestamp)
    all_news = sorted(
        all_news, key=lambda n: normalize_date(n.get("published_at", "")), reverse=True
    )
    return all_news


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ===== Caching + news fetching =====
def fetch_news_for_symbol(symbol, page_size=10):
    cache_key = f"news:{symbol}:{page_size}"
    cached = get_cached_response("newsapi", cache_key)
    if cached:
        return cached
    url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Normalizare - returnăm o listă de dicturi cu aceleași chei
        articles = []
        for a in data.get("articles", []):
            articles.append(
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "published_at": a.get("publishedAt", "") or a.get("published_at", ""),
                    "url": a.get("url", "#"),
                }
            )
        logger.debug(f"[CONVERT] For {symbol}: data['percent'] = {data.get('percent')}")
        set_cached_response("newsapi", cache_key, articles)
        return articles
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}", exc_info=True)
        return []


# ===== Date financiare fundamentale =====
def fetch_financial_data(symbol):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()

            return {
                "Name": data.get("Name", ""),
                "MarketCap": data.get("MarketCapitalization", ""),
                "PERatio": data.get("PERatio", ""),
                "EPS": data.get("EPS", ""),
                "DividendYield": data.get("DividendYield", ""),
                "Sector": data.get("Sector", ""),
                "Industry": data.get("Industry", ""),
                "RevenueTTM": data.get("RevenueTTM", ""),
                "ProfitMargin": data.get("ProfitMargin", ""),
            }

    except Exception as e:
        logger.error(f"Error fetching AlphaVantage financials for {symbol}: {e}", exc_info=True)
    return {}


# ===== Indicatori tehnici (dummy/demo, extinde cu date reale dacă vrei) =====
def fetch_technicals(symbol):
    # Poți folosi AlphaVantage sau alt API real aici!
    # Pentru demo:
    return {
        "RSI": "55.1",
        "SMA_20": "in uptrend",
        "SMA_50": "flat",
        "Volume": "increasing",
    }


# ===== Context macroeconomic demo (poți integra API știri macro/Evenimente) =====
def fetch_macro(symbol):
    # Exemplu static; extinde după nevoie
    return [
        "US interest rates unchanged this week.",
        "Recent tensions in trade between US and China.",
    ]


# ===== Context istoric: știri + impact stocate în DB =====
def fetch_historical_impacts(symbol, limit=5):
    with get_db() as db:
        history = (
            db.query(HistoricalImpact)
            .filter(HistoricalImpact.symbol == symbol)
            .order_by(HistoricalImpact.news_date.desc())
            .limit(limit)
            .all()
        )
        # Transformăm rezultatul în listă de dict (pentru prompt/afisare)
        results = []
        for h in history:
            results.append(
                {
                    "date": h.news_date.strftime("%Y-%m-%d") if h.news_date else "",
                    "news": h.news_title,
                    "impact": (
                        f"{h.impact_percentage:+.2f}%" if h.impact_percentage is not None else "N/A"
                    ),
                }
            )
        return results


# ===== Prompt AI avansat =====
def build_advanced_prompt(
    symbol, news_text, financials, technicals, macro_events, historical_impacts
):
    prompt = f"""
You are an advanced AI financial analyst with access to global news, market data, and financial reports.

Analyze the following for the stock {symbol}:

1. Recent News:
{news_text}

2. Recent and similar historical news impacts on this stock or sector:
{historical_impacts}

3. Company financials:
{financials}

4. Technical indicators:
{technicals}

5. Relevant macroeconomic context (trade wars, crises, interest rates, etc):
{macro_events}

Based on all these, provide a thorough analysis and then answer clearly:
- What is the expected percentage change for {symbol} in the next 7 days?
- What is the probability (in %) that your prediction will be correct? (justify)
- Is this trade likely to result in a win, loss, or breakeven (considering dividends too)?
- What is the reward-to-risk ratio?
- Should an investor buy, hold, or sell now, and why? (with optimal entry/exit suggestion)
- Justify each conclusion using multiple sources of reasoning (news, fundamentals, technicals, sentiment, macro).
Respond with clear structure: Expected Change, Probability, Trade Outcome, Reward/Risk, Recommendation, Reasoning.
"""
    return prompt


def _to_float_safe(val):
    """Conversie robustă la float: acceptă ',' ca separator zecimal și minus unicode."""
    if val is None:
        return None
    s = str(val).strip().replace("\u2212", "-")  # minus unicode -> normal
    s = s.replace(",", ".")
    # respinge șiruri invalide
    if s in {"", "."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def analyze_with_openai(
    symbol,
    news_list,
    financials,
    technicals,
    macro_events,
    historical_impacts,
    actual_price=None,
    extra_context=None,
):
    """
    AI prediction cu context complet pentru stock, extensibil și robust.
    JSON-first parsing + fallback regex. Fără inserări de 0.0 când parsarea eșuează.
    """
    # ======= 1) Filtrare știri relevante =======
    top_news = rank_news(news_list, top_n=5)
    summarized_news = summarize_news(top_news, max_sentences=2)

    # ======= 2) Construiește context scurt și relevant =======
    news_text = ""
    for news in summarized_news:
        news_text += f"Title: {news['title']}\nSummary: {news['summary']}\n\n"

    historical_str = (
        "\n".join(
            [
                f"{h['date']}: {h['news']} | Impact: {h['impact']}"
                for h in (historical_impacts or [])[:5]
            ]
        )
        if historical_impacts
        else ""
    )

    financials_str = (
        "\n".join([f"{k}: {v}" for k, v in (financials or {}).items()][:8]) if financials else ""
    )
    technicals_str = (
        "\n".join([f"{k}: {v}" for k, v in (technicals or {}).items()][:5]) if technicals else ""
    )
    macro_str = "\n".join((macro_events or [])[:5]) if macro_events else ""
    price_str = f"Current Price: {actual_price}" if actual_price else ""
    extra_str = ""
    if extra_context:
        for key, value in extra_context.items():
            extra_str += f"{key}: {value}\n"

    vol_daily = daily_vol_pct(symbol, lookback=20) or 0.0  # %
    atr_p = atr_pct(symbol, lookback=14) or 0.0  # %
    earn_days = days_to_next_earnings(symbol)  # int|None
    reg = market_regime()
    reg_str = ", ".join(
        [f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in reg.items()]
    )

    # ======= 3) PROMPT (JSON-only) =======
    prompt = f"""
You are a professional stock market analyst AI. You never write disclaimers. Use ONLY the structured data below, do not make up facts.
If there is not enough data, do your best to estimate, but mark probability as 'Low' and explicitly explain the lack of context in the justification.

Stock: {symbol}
{price_str}

Recent News (top, summarized):
{news_text}

Historical Similar Impacts:
{historical_str}

Company Financials:
{financials_str}

Technicals:
{technicals_str}

Macro/Regime:
{macro_str}
Market Regime: {reg_str}
Volatility(20d σ, daily %): {vol_daily:.2f}
ATR(14, % of price): {atr_p:.2f}
Days to next earnings: {earn_days if earn_days is not None else "unknown"}

Rules:
- 7d expected change should be consistent with volatility. If |expected_change_pct| > 5 * daily σ, lower probability unless earnings/catalyst < 10 days.
- If probability < 55, recommendation must be HOLD (not Buy/Sell).
- If reward_to_risk < 1.0, don't recommend BUY.
- Always be concise.

Respond ONLY with compact JSON, no text outside JSON, matching exactly:

{{
 "symbol": "{symbol}",
 "horizon_days": 7,
 "expected_change_pct": <number>,      // e.g., 3.5 means +3.5%
 "probability": <number>,              // 0..100
 "trade_outcome": "Win|Loss|Breakeven",
 "reward_to_risk": <number>,
 "recommendation": "Buy|Hold|Sell",
 "short_justification": "<<=140 chars>",
 "prob_justification": "<<=140 chars>"
}}
"""

    # pentru fallback regex
    MINUS = "\u2212"  # minus unicode
    NUM = rf"([+\- {MINUS}]?\d+(?:[.,]\d+)?)"

    max_retries = 4
    percent = None
    probability = None
    reward_risk = None
    trade_outcome = ""
    recommendation = ""
    short_justif = ""
    prob_justif = ""
    reco_reason = ""

    for attempt in range(max_retries):
        try:
            resp = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,  # ai deja OPENAI_MODEL sus
                messages=[
                    {
                        "role": "system",
                        "content": "You are a no-nonsense, real-time stock analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
                timeout=60,  # v1 folosește 'timeout'; 'request_timeout' e din vechiul API
            )
            answer = resp.choices[0].message.content.strip()

            # ======= Retry dacă primesc disclaimere =======
            disclaimer_fragments = [
                "as an ai",
                "i don't have access",
                "i do not have access",
                "i cannot access",
                "real-time access",
                "no prediction possible",
            ]
            if any(x in answer.lower() for x in disclaimer_fragments):
                logger.warning(
                    f"Received disclaimer/no-prediction from AI, retrying ({attempt+1}/{max_retries})..."
                )
                continue

                # ======= 4) JSON-first (robust) =======

            def _strip_code_fences(s: str) -> str:
                s = s.strip()
                # ```json ... ```  sau ``` ... ```
                if s.startswith("```"):
                    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
                    s = re.sub(r"\s*```$", "", s)
                return s.strip()

            def _extract_json_candidate(s: str) -> str | None:
                # ia substringul de la primul '{' la ultimul '}' (acoperă text înainte/după)
                i, j = s.find("{"), s.rfind("}")
                if i != -1 and j != -1 and j > i:
                    return s[i : j + 1]
                return None

            def _load_json_robust(s: str) -> dict | None:
                # 1) încearcă direct (după strip fences)
                try:
                    return json.loads(s)
                except Exception:
                    pass
                # 2) extrage cel mai mare {...}
                cand = _extract_json_candidate(s)
                if not cand:
                    return None
                # 2a) încearcă direct
                try:
                    return json.loads(cand)
                except Exception:
                    pass
                # 2b) sanitize: scoate comentarii //, convertește ' la "
                fixed = re.sub(r"(?m)//.*?$", "", cand)  # remove // comments
                fixed = fixed.replace("'", '"')  # single -> double quotes
                fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
                try:
                    return json.loads(fixed)
                except Exception:
                    return None

            data = _load_json_robust(_strip_code_fences(answer))
            if data is not None:
                percent = _to_float_safe(data.get("expected_change_pct"))
                probability = _to_float_safe(data.get("probability"))
                trade_outcome = (data.get("trade_outcome") or "").strip()
                reward_risk = _to_float_safe(data.get("reward_to_risk"))
                recommendation = (data.get("recommendation") or "").strip()
                short_justif = (data.get("short_justification") or "").strip()
                prob_justif = (data.get("prob_justification") or "").strip()
                reco_reason = (data.get("reco_reason") or short_justif or "").strip()

                # --- Recovery dacă lipsesc cheile standard dar info există sub alt nume ---
                if percent is None:
                    logger.warning(
                        f"[{symbol}] JSON missing 'expected_change_pct' – trying alt keys."
                    )
                    for k in (
                        "expected_change",
                        "expected_change_percent",
                        "expected_pct",
                        "expectedChangePct",
                        "exp_change_pct",
                    ):
                        if k in data and data.get(k) not in (None, ""):
                            percent = _to_float_safe(data.get(k))
                            if percent is not None:
                                break
                    # ultimă încercare: caută orice cheie "expected..." cu un număr
                    if percent is None:
                        m = re.search(
                            r'"expected[^"]*"\s*:\s*([+\-\u2212]?\d+(?:[.,]\d+)?)',
                            answer,
                            re.I,
                        )
                        if m:
                            percent = _to_float_safe(m.group(1))

                if probability is None:
                    logger.warning(f"[{symbol}] JSON missing 'probability' – trying alt keys.")
                    for k in (
                        "prob",
                        "prob_pct",
                        "probability_pct",
                        "probabilityPercent",
                    ):
                        if k in data and data.get(k) not in (None, ""):
                            probability = _to_float_safe(data.get(k))
                            if probability is not None:
                                break
                    if probability is None:
                        # dacă în JSON există "probability": "58%" (string cu %)
                        m = re.search(
                            r'"prob[^"]*"\s*:\s*"?\s*([+\-\u2212]?\d+(?:[.,]\d+)?)\s*%',
                            answer,
                            re.I,
                        )
                        if m:
                            probability = _to_float_safe(m.group(1))

                if reward_risk is None:
                    for k in ("rr", "reward_risk", "rewardToRisk", "rewardToRiskRatio"):
                        if k in data and data.get(k) not in (None, ""):
                            reward_risk = _to_float_safe(data.get(k))
                            if reward_risk is not None:
                                break

                # Normalizează badge-urile dacă vin ciudat
                if trade_outcome:
                    trade_outcome = trade_outcome.replace("Break-even", "Breakeven").strip().title()
                if recommendation:
                    recommendation = recommendation.strip().title()

            else:
                # ======= 5) Fallback regex (format vechi non-JSON) =======
                def extract(key, pattern, cast=None):
                    m = re.search(pattern, answer, re.I | re.S)
                    if not m:
                        return None
                    val = m.group(1).strip()
                    return cast(val) if cast else val

                # Expected Change (%)
                percent = extract(
                    "Expected Change",
                    rf"Expected\s*Change[^\n]*?:\s*{NUM}\s*%?",
                    _to_float_safe,
                )
                if percent is None:
                    m = re.search(rf"Expected[^\n]*?{NUM}\s*%", answer, re.I | re.S)
                    if m:
                        percent = _to_float_safe(m.group(1))

                # Probability (%)
                m = re.search(
                    rf"Probability[^\n]*?:\s*{NUM}\s*%?(?:\s*[-–—]\s*(.*))?",
                    answer,
                    re.I | re.S,
                )
                if m:
                    probability = _to_float_safe(m.group(1))
                    prob_justif = (m.group(2) or "").strip()
                else:
                    line = re.search(r"^.*Probability.*$", answer, re.I | re.M)
                    if line:
                        mm = re.search(rf"{NUM}\s*%", line.group(0))
                        if mm:
                            probability = _to_float_safe(mm.group(1))
                        prob_justif = ""
                    else:
                        probability, prob_justif = None, ""

                # Reward/Risk
                reward_risk = extract(
                    "Reward-to-Risk Ratio",
                    r"Reward[\s\-]?to[\s\-]?Risk\s*Ratio\s*:\s*([0-9]+(?:[.,][0-9]+)?)(?:\s*[xX])?",
                    _to_float_safe,
                )

                # Win/Loss/Breakeven
                trade_outcome = extract(
                    "Win/Loss/Breakeven", r"Win/Loss/Breakeven\s*:\s*([A-Za-z\-]+)"
                )
                if trade_outcome:
                    trade_outcome = trade_outcome.replace("Break-even", "Breakeven").strip().title()

                # Recommendation + motiv scurt
                reco = extract("Buy/Hold/Sell and why", r"Buy/Hold/Sell and why\s*:\s*(.+)")
                if reco and "-" in reco:
                    rec_badge, rec_reason = reco.split("-", 1)
                else:
                    _rec_badge, _rec_reason = (reco.strip() if reco else ""), ""

                    # --- Normalizări finale (clamp) înainte de return ---
            if probability is not None:
                probability = float(max(0.0, min(100.0, probability)))
            if percent is not None:
                percent = float(max(-30.0, min(30.0, percent)))  # protecție outliers
            if reward_risk is not None and reward_risk < 0:
                reward_risk = None

            # badge-uri standard
            if trade_outcome:
                trade_outcome = trade_outcome.replace("Break-even", "Breakeven").strip().title()
            if recommendation:
                recommendation = recommendation.strip().title()

            logger.info(f"AI answer for {symbol}: {answer[:240]}...")
            logger.info(
                f"[PARSE] {symbol} -> pct={percent} prob={probability} rr={reward_risk} reco={recommendation}"
            )

            return {
                "ai_full_text": answer,
                "percent": percent,
                "probability": probability,
                "prob_justification": prob_justif,
                "reward_to_risk": reward_risk,
                "trade_outcome": trade_outcome,
                "recommendation": recommendation,
                "reco_reason": reco_reason,
                "short_justification": short_justif,
            }

        except RateLimitError:
            wait_time = 2**attempt
            logger.warning(f"OpenAI rate limit hit, retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue
        except Timeout:
            wait_time = 2**attempt
            logger.warning(f"OpenAI timeout, retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue
        except (APIConnectionError, BadRequestError, APIError) as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            time.sleep(1.0)
            continue
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}", exc_info=True)
            time.sleep(1.0)
            continue

    # ======= 6) dacă toate încercările au eșuat =======
    return {
        "ai_full_text": "Failed to get prediction from AI.",
        "percent": None,
        "probability": None,
        "prob_justification": "",
        "reward_to_risk": None,
        "trade_outcome": None,
        "recommendation": "",
        "reco_reason": "",
        "short_justification": "",
    }


# ===== Batch process complex =====
def batch_process(symbols, batch_size=20, force_refresh=False):
    """
    Procesează simboluri în batch:
      - dacă există predicție recentă (<24h) și nu e force_refresh -> întoarce din DB
      - altfel:
          * strânge context (news/fin/tech/macro/history)
          * cere LLM
          * face blend cu ML (dacă există REG/CLF)
          * aplică calibrator (dacă există)
          * calculează benzi (P20/P80) + price_low/high (dacă există modele de cuantile)
          * salvează totul în DB
    Returnează: dict {symbol -> payload pentru UI}
    """
    import numpy as np

    # modele încărcate global (dacă nu există, vor fi None)
    _REG = globals().get("_REG", None)
    _CLF = globals().get("_CLF", None)
    _CAL = globals().get("_CAL", None)
    _Q20 = globals().get("_Q20", None)
    _Q80 = globals().get("_Q80", None)

    results = {}
    db = SessionLocal()
    try:
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]

            for symbol in batch:
                # --------- 1) preț live (util în ambele ramuri) ----------
                live_price = None
                try:
                    live_price = fetch_current_price(symbol)
                except Exception:
                    live_price = None

                # --------- 2) recency check ----------
                recent_prediction = (
                    db.query(StockPrediction)
                    .filter(StockPrediction.symbol == symbol)
                    .order_by(StockPrediction.published_at.desc())
                    .first()
                )

                now_utc = datetime.now(timezone.utc)
                last_ts = recent_prediction.published_at if recent_prediction else None
                if last_ts and (last_ts.tzinfo is None):
                    last_ts = last_ts.replace(tzinfo=timezone.utc)

                if (
                    recent_prediction
                    and last_ts
                    and last_ts > (now_utc - timedelta(hours=24))
                    and not force_refresh
                ):
                    # --------- CACHED ----------
                    out = {
                        "percent": (
                            getattr(recent_prediction, "final_percent", None)
                            if getattr(recent_prediction, "final_percent", None) is not None
                            else recent_prediction.estimated_change
                        ),
                        "final_percent": getattr(recent_prediction, "final_percent", None),
                        "explanation": getattr(recent_prediction, "analysis", "") or "",
                        "cached": True,
                        "probability": (
                            getattr(recent_prediction, "final_probability", None)
                            if getattr(recent_prediction, "final_probability", None) is not None
                            else getattr(recent_prediction, "probability", None)
                        ),
                        "final_probability": getattr(recent_prediction, "final_probability", None),
                        "reward_to_risk": getattr(recent_prediction, "reward_to_risk", None),
                        "trade_outcome": getattr(recent_prediction, "trade_outcome", None),
                        "recommendation": getattr(recent_prediction, "recommendation", None),
                        "reco_reason": getattr(recent_prediction, "reco_reason", None),
                        "short_justification": getattr(
                            recent_prediction, "short_justification", None
                        ),
                        "ai_full_text": getattr(recent_prediction, "ai_full_text", "") or "",
                        "actual_price": (
                            live_price
                            if live_price is not None
                            else getattr(recent_prediction, "actual_price", None)
                        ),
                        "date_pred": (last_ts.strftime("%Y-%m-%d %H:%M") if last_ts else None),
                        # intervale/cuantile dacă există în DB
                        "ci_low_pct": getattr(recent_prediction, "ci_low_pct", None),
                        "ci_high_pct": getattr(recent_prediction, "ci_high_pct", None),
                        "p20": getattr(recent_prediction, "p20", None),
                        "p80": getattr(recent_prediction, "p80", None),
                        "price_low": getattr(recent_prediction, "price_low", None),
                        "price_high": getattr(recent_prediction, "price_high", None),
                    }
                    results[symbol] = out
                    logger.info(
                        f"[CACHED] {symbol} price={out['actual_price']} pct={out['percent']} outcome={out['trade_outcome']} date={out['date_pred']}"
                    )
                    continue

                # --------- FRESH ----------
                # (a) strânge context
                try:
                    all_news = fetch_all_news_for_symbol(symbol, page_size=10, finnhub_items=5)
                except Exception:
                    all_news = []
                try:
                    top_news = rank_news(all_news, top_n=5)
                    summarized_news = summarize_news(top_news, max_sentences=2)
                except Exception:
                    summarized_news = []

                try:
                    financials = fetch_financial_data(symbol)
                except Exception:
                    financials = {}
                try:
                    technicals = fetch_technicals(symbol)
                except Exception:
                    technicals = {}
                try:
                    macro = fetch_macro(symbol)
                except Exception:
                    macro = []
                try:
                    history = fetch_historical_impacts(symbol)
                except Exception:
                    history = []

                actual_price = live_price
                extra_context = {}

                # (b) cere LLM
                try:
                    ai_data = analyze_with_openai(
                        symbol=symbol,
                        news_list=summarized_news,
                        financials=financials,
                        technicals=technicals,
                        macro_events=macro,
                        historical_impacts=history,
                        actual_price=actual_price,
                        extra_context=extra_context,
                    )
                except Exception as e:
                    logger.error(f"[{symbol}] analyze_with_openai failed: {e}", exc_info=True)
                    # fallback minimal (nu inserăm rând prost în DB)
                    results[symbol] = {
                        "percent": None,
                        "final_percent": None,
                        "explanation": "",
                        "cached": False,
                        "probability": None,
                        "final_probability": None,
                        "reward_to_risk": None,
                        "trade_outcome": None,
                        "recommendation": "",
                        "reco_reason": "",
                        "short_justification": "",
                        "ai_full_text": "",
                        "actual_price": actual_price,
                        "date_pred": now_utc.strftime("%Y-%m-%d %H:%M"),
                        "ci_low_pct": None,
                        "ci_high_pct": None,
                        "p20": None,
                        "p80": None,
                        "price_low": None,
                        "price_high": None,
                    }
                    continue

                if not isinstance(ai_data, dict):
                    # protecție în cazul unui răspuns ne-JSON
                    ai_txt = str(ai_data)
                    results[symbol] = {
                        "percent": None,
                        "final_percent": None,
                        "explanation": ai_txt,
                        "cached": False,
                        "probability": None,
                        "final_probability": None,
                        "reward_to_risk": None,
                        "trade_outcome": None,
                        "recommendation": "",
                        "reco_reason": "",
                        "short_justification": "",
                        "ai_full_text": ai_txt,
                        "actual_price": actual_price,
                        "date_pred": now_utc.strftime("%Y-%m-%d %H:%M"),
                        "ci_low_pct": None,
                        "ci_high_pct": None,
                        "p20": None,
                        "p80": None,
                        "price_low": None,
                        "price_high": None,
                    }
                    continue

                pct_llm = ai_data.get("percent")
                p_llm = ai_data.get("probability")

                # (c) features pentru ML (acceptă și dict, și ndarray) – folosim DataFrame cu nume de coloane pentru a elimina warning-urile
                x_model = None
                try:
                    feats_raw = _live_features(symbol)  # poate fi dict SAU vector
                    import pandas as pd
                    import numpy as np

                    if isinstance(feats_raw, dict):
                        # avem deja dict -> îl aliniem la ordinea canonică
                        x_model = _to_model_frame(feats_raw, _FEATS_LIST)  # DataFrame (1, n_feats)
                    else:
                        arr = np.asarray(feats_raw, dtype=float)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                        # aliniem la numele de coloane cu care s-a antrenat modelul
                        x_model = pd.DataFrame(arr, columns=_FEATS_LIST)
                except Exception as e:
                    logger.warning(f"[{symbol}] live features error: {e}")
                    x_model = None

                # (d) blend cu ML, plus cap pe volatilitate
                pct_final = pct_llm
                p_final = p_llm

                try:
                    # Probabilitate din CLF (în %)
                    if (_CLF is not None) and (x_model is not None):
                        try:
                            p_stat = float(_CLF.predict_proba(x_model)[0, 1] * 100.0)
                            if p_final is None:
                                p_final = p_stat
                            else:
                                p_final = 0.5 * float(p_final) + 0.5 * p_stat
                        except Exception:
                            pass

                    # Pct din REG
                    if (_REG is not None) and (x_model is not None):
                        try:
                            pct_stat = float(_REG.predict(x_model)[0])
                            if pct_final is None:
                                pct_final = pct_stat
                            else:
                                pct_final = 0.5 * float(pct_final) + 0.5 * pct_stat
                        except Exception:
                            pass

                        # cap volatilitate 3σ * sqrt(7) (heuristic)
                        try:
                            from .features import (
                                daily_vol_pct,
                            )  # import local, dacă există

                            sigma = daily_vol_pct(symbol, lookback=20) or 0.0
                            if sigma:
                                cap = 3.0 * sigma * np.sqrt(7)
                                pct_final = float(np.clip(pct_final, -cap, cap))
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"[{symbol}] Ensemble error: {e}")

                # (f) benzi de cuantile (procente) + interval de preț
                p20 = p80 = price_low = price_high = None
                try:
                    if (x_model is not None) and (_Q20 is not None) and (_Q80 is not None):
                        p20 = float(_Q20.predict(x_model)[0])  # procente
                        p80 = float(_Q80.predict(x_model)[0])
                        if actual_price is not None:
                            price_low = float(actual_price * (1.0 + p20 / 100.0))
                            price_high = float(actual_price * (1.0 + p80 / 100.0))
                except Exception as e:
                    logger.warning(f"[{symbol}] Quantile error: {e}")

                # (g) CI din vol (fallback dacă nu avem cuantile)
                ci_low_pct = ci_high_pct = None
                if (p20 is not None) and (p80 is not None):
                    ci_low_pct, ci_high_pct = p20, p80
                    if pct_final is not None:
                        pct_final = float(np.minimum(np.maximum(pct_final, p20), p80))
                else:
                    try:
                        from .features import daily_vol_pct

                        sigma_d = daily_vol_pct(symbol, lookback=20) or 0.0
                        if (sigma_d is not None) and (pct_final is not None):
                            band = float(1.64 * sigma_d * np.sqrt(7))  # ~90%
                            ci_low_pct = float(max(pct_final - band, -30.0))
                            ci_high_pct = float(min(pct_final + band, 30.0))
                    except Exception:
                        pass
                # --- Calibrare probabilitate (după blend LLM+ML, înainte de reguli/DB)
                p_before = p_final
                p_final = _apply_calibrator(p_final)

                before_str = "None" if p_before is None else f"{p_before:.2f}"
                after_str = "None" if p_final is None else f"{p_final:.2f}"
                logger.info(f"[CAL] {symbol} before={before_str}% after={after_str}%")

                # ---- Policy rules (final sanity) ----
                try:
                    from .features import daily_vol_pct, days_to_next_earnings

                    sigma_d = daily_vol_pct(symbol, lookback=20) or 0.0
                    earn_days = days_to_next_earnings(symbol)
                except Exception:
                    sigma_d, earn_days = 0.0, None

                pct_final, p_final, rec_override = _apply_policy_rules(
                    symbol,
                    pct_final,
                    p_final,
                    ai_data.get("reward_to_risk"),
                    earn_days,
                    sigma_d,
                )
                if rec_override:
                    ai_data["recommendation"] = rec_override

                # Guard final: fără procente -> nu inserăm în DB o linie proastă
                if (pct_final is None) and (pct_llm is None):
                    logger.warning(f"[{symbol}] Missing percent from AI/ML. Skipping DB insert.")
                    results[symbol] = {
                        "percent": None,
                        "final_percent": None,
                        "explanation": ai_data.get("ai_full_text") or "",
                        "cached": False,
                        "probability": float(p_final) if p_final is not None else None,
                        "final_probability": (float(p_final) if p_final is not None else None),
                        "reward_to_risk": ai_data.get("reward_to_risk"),
                        "trade_outcome": ai_data.get("trade_outcome"),
                        "recommendation": ai_data.get("recommendation"),
                        "reco_reason": ai_data.get("reco_reason"),
                        "short_justification": ai_data.get("short_justification"),
                        "ai_full_text": ai_data.get("ai_full_text") or "",
                        "actual_price": actual_price,
                        "date_pred": now_utc.strftime("%Y-%m-%d %H:%M"),
                        "ci_low_pct": ci_low_pct,
                        "ci_high_pct": ci_high_pct,
                        "p20": p20,
                        "p80": p80,
                        "price_low": price_low,
                        "price_high": price_high,
                    }
                    continue

                # --------- write to DB ----------
                try:
                    new_pred = StockPrediction(
                        title=f"AI Prediction for {symbol}",
                        description="Based on all data.",
                        url="#",
                        analysis=ai_data.get("ai_full_text") or "",
                        symbol=symbol,
                        estimated_change=(
                            float(pct_llm) if pct_llm is not None else float(pct_final)
                        ),
                        published_at=now_utc,
                        probability=(
                            float(ai_data.get("probability"))
                            if ai_data.get("probability") is not None
                            else None
                        ),
                        prob_justification=ai_data.get("prob_justification"),
                        reward_to_risk=ai_data.get("reward_to_risk"),
                        trade_outcome=ai_data.get("trade_outcome"),
                        recommendation=ai_data.get("recommendation"),
                        reco_reason=ai_data.get("reco_reason"),
                        short_justification=ai_data.get("short_justification"),
                        ai_full_text=ai_data.get("ai_full_text") or "",
                        actual_price=actual_price,
                        final_percent=(float(pct_final) if pct_final is not None else None),
                        final_probability=(float(p_final) if p_final is not None else None),
                        ci_low_pct=ci_low_pct,
                        ci_high_pct=ci_high_pct,
                        p20=p20,
                        p80=p80,
                        price_low=price_low,
                        price_high=price_high,
                    )
                    db.add(new_pred)
                    db.commit()
                except Exception as e:
                    logger.error(f"[{symbol}] DB insert error: {e}", exc_info=True)

                # --------- rezultat pentru UI ----------
                out = {
                    "percent": (float(pct_final) if pct_final is not None else float(pct_llm)),
                    "final_percent": (
                        float(pct_final) if pct_final is not None else float(pct_llm)
                    ),
                    "explanation": ai_data.get("ai_full_text") or "",
                    "cached": False,
                    "probability": (
                        float(p_final)
                        if p_final is not None
                        else (float(p_llm) if p_llm is not None else None)
                    ),
                    "final_probability": (
                        float(p_final)
                        if p_final is not None
                        else (float(p_llm) if p_llm is not None else None)
                    ),
                    "reward_to_risk": ai_data.get("reward_to_risk"),
                    "trade_outcome": ai_data.get("trade_outcome"),
                    "recommendation": ai_data.get("recommendation"),
                    "reco_reason": ai_data.get("reco_reason"),
                    "short_justification": ai_data.get("short_justification"),
                    "ai_full_text": ai_data.get("ai_full_text") or "",
                    "actual_price": actual_price,
                    "date_pred": now_utc.strftime("%Y-%m-%d %H:%M"),
                    "ci_low_pct": ci_low_pct,
                    "ci_high_pct": ci_high_pct,
                    "p20": p20,
                    "p80": p80,
                    "price_low": price_low,
                    "price_high": price_high,
                }
                results[symbol] = out
                logger.info(
                    f"[FRESH] {symbol} price={out['actual_price']} pct={out['percent']} outcome={out['trade_outcome']} date={out['date_pred']}"
                )

            # ---- API rate-limit buffer între batch-uri ----
            # la finalul fiecărui batch, dacă mai urmează unul, fă pauză anti-rate-limit
            if i + batch_size < len(symbols):
                logger.info(f"Batch {i // batch_size + 1}: sleeping 65s (rate limit Finnhub)...")
                time.sleep(65)

    finally:
        db.close()

    return results
