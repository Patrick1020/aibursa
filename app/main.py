# -*- coding: utf-8 -*-
"""
AIBursa — FastAPI app (enterprise-grade main.py)
- Lifespan async (init/shutdown)
- Static + Templates
- Request-ID middleware + global error handlers
- CORS + GZip + basic rate-limit (soft)
- Health endpoints (/health, /healthz)
- UI routes: "/", "/dashboard", "/stock/{symbol}", "/backtest-dashboard", "/backtest/pro"
- API routes: /api/refresh/{symbol}, /api/refresh_all
"""
from __future__ import annotations

import os
import time
import uuid
import subprocess
import json
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from collections import defaultdict


from fastapi import Depends, Header
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from types import SimpleNamespace

# --- app modules ---
from .database import init_db, get_db, SessionLocal
from .models import StockPrediction
from .logger import logger

# backtest: generează raport HTML din predicțiile din DB
try:
    from .backtest_v2 import generate_report
except ImportError:
    # fallback în cazul rulării fără pachet relativ
    pass


from .stock_analysis import (
    batch_process,
    fetch_news_for_symbol,
    fetch_financial_data,
    fetch_historical_impacts,
    rank_news,
    summarize_news,
)


NGROK_BASE = os.getenv("NGROK_BASE", "")  # ex: https://1234-abc.ngrok-free.app

# -------- Paths / Settings --------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# ---------- Lifespan (startup/shutdown) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("[BOOT] init_db()...")
    init_db()

    from fastapi import Body

    @app.post(
        "/ops/train",
        operation_id="ops_train",
        summary="Train ML models",
        tags=["ops"],
        dependencies=[Depends(require_ops_key)],
    )
    async def ops_train(
        years: int = Body(8),
        horizon: int = Body(7),
        tickers: int = Body(15),
        excess_target: bool = Body(True),
    ):
        """
        Rulează `python -m app.train_model` cu parametrii dați.
        Returnează stdout/stderr ca text.
        """
        cmd = [
            os.getenv("PYTHON_EXECUTABLE", "python"),
            "-m",
            "app.train_model",
            "--years",
            str(years),
            "--horizon",
            str(horizon),
            "--tickers",
            str(tickers),
            "--excess_target",
            "true" if excess_target else "false",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "ok": (p.returncode == 0),
            "cmd": " ".join(cmd),
            "stdout": p.stdout[-4000:],  # tail scurt
            "stderr": p.stderr[-4000:],
        }

    @app.post(
        "/ops/backtest",
        operation_id="ops_backtest",
        summary="Run Backtest Pro and produce HTML report",
        tags=["ops"],
        dependencies=[Depends(require_ops_key)],
    )
    async def ops_backtest(horizon_days: int = Body(7)):
        """
        Apelează backtest_v2.generate_report(horizon_days) și returnează calea fișierului HTML.
        """
        try:
            from .backtest_v2 import generate_report

            path = generate_report(horizon_days=horizon_days)
            return {"ok": True, "report_html": path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    @app.post(
        "/ops/refresh_all",
        operation_id="ops_refresh_all",
        summary="Refresh AI predictions for the default symbol universe",
        tags=["ops"],
        dependencies=[Depends(require_ops_key)],
    )
    async def ops_refresh_all():
        """
        Rulează batch_process pe universul implicit (sau preia din request).
        """
        try:
            from .stock_analysis import batch_process

            universe = [
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
            out = batch_process(universe, force_refresh=True)
            return {"ok": True, "count": len(out)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Refresh failed: {e}")

    # optional: background scheduler (best-effort; won't fail app if missing)
    scheduler = None
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        scheduler = AsyncIOScheduler()
        # EXAMPLE job (disabled by default). Uncomment to enable a nightly refresh.
        # scheduler.add_job(lambda: logger.info("[JOB] nightly tick"), "cron", hour=3, minute=0)
        scheduler.start()
        logger.info("[BOOT] scheduler started.")
    except Exception as e:
        logger.warning(f"[BOOT] scheduler not started: {e}")

    app.state.scheduler = scheduler
    yield

    # Shutdown
    try:
        if app.state.scheduler:
            app.state.scheduler.shutdown(wait=False)
            logger.info("[SHUTDOWN] scheduler stopped.")
    except Exception:
        pass
    logger.info("[SHUTDOWN] complete.")


app = FastAPI(lifespan=lifespan)
app.title = "AIBursa API"
app.version = "1.1.0"
app.description = "Stock predictions (LLM + ML ensemble) with FastAPI."

# --- Q&A cu Asistentul (Assistants API) ---
from fastapi import Body
from pydantic import BaseModel
from openai import OpenAI


class AskPayload(BaseModel):
    question: str


@app.post("/assistant/ask")
async def assistant_ask(payload: AskPayload):
    api_key = os.environ.get("OPENAI_API_KEY")
    asst_id = os.environ.get("ASSISTANT_ID")
    if not api_key or not asst_id:
        return JSONResponse({"error": "OPENAI_API_KEY/ASSISTANT_ID lipsă"}, status_code=500)

    client = OpenAI(api_key=api_key)

    thread = client.beta.threads.create(messages=[{"role": "user", "content": payload.question}])
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=asst_id)
    # poll simplu
    import time as _t

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status in ("completed", "failed", "cancelled", "expired", "requires_action"):
            break
        _t.sleep(0.8)

    if run.status != "completed":
        return JSONResponse(
            {"error": f"run status {run.status}", "details": getattr(run, "last_error", None)},
            status_code=500,
        )

    msgs = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    answer = ""
    for m in msgs.data:
        if m.role == "assistant":
            for c in m.content:
                if getattr(c, "type", None) == "text":
                    answer += c.text.value + "\n"
    return {"answer": answer.strip()}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # poți restrânge ulterior la URL-ul ngrok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/public/openapi", include_in_schema=False, tags=["ops"])
def public_openapi():
    path = os.path.join(os.path.dirname(__file__), "..", "openapi.public.json")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="openapi.public.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.get("/actions/openapi.json", include_in_schema=False)
def actions_openapi():
    with open("openapi_actions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # dacă ai setat NGROK_BASE în .env, îl injectăm aici ca să nu uiți să modifici fișierul manual
    if NGROK_BASE:
        data["servers"] = [{"url": NGROK_BASE}]
    return JSONResponse(data)


# ---------- Templates / Static ----------
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
OPS_API_KEY = os.getenv("OPS_API_KEY", "").strip()


def require_ops_key(x_api_key: str = Header(default="")):
    if not OPS_API_KEY or x_api_key != OPS_API_KEY:
        # nu divulgăm motivul exact
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------- Middlewares ----------
# GZip
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS (adjust in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request-ID for log correlation
import contextvars

_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        _request_id_ctx.set(rid)
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


app.add_middleware(RequestIDMiddleware)


def get_request_id() -> str | None:
    try:
        return _request_id_ctx.get()
    except LookupError:
        return None


# Basic soft rate-limit for /api/*
_RATE_BUCKET = defaultdict(list)  # ip -> [timestamps]
_RATE_MAX = 60
_RATE_WINDOW = 60


@app.middleware("http")
async def simple_rate_limiter(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = _RATE_BUCKET[ip]
        # evict old
        while bucket and (now - bucket[0] > _RATE_WINDOW):
            bucket.pop(0)
        if len(bucket) >= _RATE_MAX:
            rid = getattr(request.state, "request_id", None)
            return JSONResponse({"error": "Too Many Requests", "request_id": rid}, status_code=429)
        bucket.append(now)
    return await call_next(request)


# Security headers (simple, non-breaking)
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    # Keep CSP basic since we render dynamic HTML with inline styles in templates
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


# ---------- Global Exception Handlers ----------
from fastapi.exceptions import RequestValidationError


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", None)
    logger.warning(f"[HTTP {exc.status_code}] {request.url} rid={rid} detail={exc.detail}")
    return JSONResponse({"error": exc.detail, "request_id": rid}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exc_handler(request: Request, exc: RequestValidationError):
    rid = getattr(request.state, "request_id", None)
    logger.warning(f"[VALIDATION] {request.url} rid={rid} errors={exc.errors()}")
    return JSONResponse(
        {"error": "Validation error", "details": exc.errors(), "request_id": rid},
        status_code=422,
    )


@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    logger.error(f"[UNHANDLED] {request.url} rid={rid}", exc_info=True)
    return JSONResponse({"error": "Internal Server Error", "request_id": rid}, status_code=500)


# ---------- Helpers ----------
def _safe_float(*vals) -> float | None:
    for v in vals:
        try:
            if v is None:
                continue
            f = float(v)
            if math.isfinite(f):
                return f
        except Exception:
            continue
    return None


def _ns_from_dict(d: dict, keys: list[str]) -> SimpleNamespace:
    return SimpleNamespace(**{k: d.get(k) for k in keys})


# ---------- Health ----------
@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict:
    # DB ping
    db_ok = False
    try:
        with get_db() as db:
            db.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False

    sch_ok = bool(getattr(app.state, "scheduler", None)) and bool(app.state.scheduler.running)

    return {
        "status": "ok" if db_ok and sch_ok else "degraded",
        "time": datetime.now(timezone.utc).isoformat(),
        "components": {"db": "up" if db_ok else "down", "scheduler": "up" if sch_ok else "down"},
        "version": getattr(app, "version", "unknown"),
    }


# ---------- UI: Homepage ----------
@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def homepage(
    request: Request,
    q: str = Query("", description="Filter by symbol substring"),
    mode: str = Query("all", description="'all' or 'batch'"),
    batch_page: int = Query(1),
    batch_size: int = Query(30),
):
    # Symbols universe (adjust as needed or load from config)
    symbols = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
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
    ]
    if q:
        symbols = [s for s in symbols if q.upper() in s]

    total = len(symbols)

    # Execute predictions
    if mode == "batch":
        start_idx = (batch_page - 1) * batch_size
        end_idx = start_idx + batch_size
        page_symbols = symbols[start_idx:end_idx]
        predictions = batch_process(page_symbols, batch_size=batch_size)
        shown_symbols = page_symbols
    else:
        predictions = batch_process(symbols, batch_size=batch_size)
        shown_symbols = symbols

    # Normalize shape for template
    fields = [
        "percent",
        "final_percent",
        "probability",
        "final_probability",
        "reward_to_risk",
        "recommendation",
        "trade_outcome",
        "explanation",
        "ai_full_text",
        "reco_reason",
        "short_justification",
        "prob_justification",
        "actual_price",
        "date_pred",
        "ci_low_pct",
        "ci_high_pct",
        "p20",
        "p80",
        "price_low",
        "price_high",
        "cached",
    ]

    converted_results: list[SimpleNamespace] = []
    for sym in shown_symbols:
        d = predictions.get(sym, {}) or {}
        ns = _ns_from_dict(d, fields)
        # attach symbol at top-level
        ns.symbol = sym
        converted_results.append(ns)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": converted_results,
            "query": q,
            "count": len(converted_results),
            "total": total,
            "mode": mode,
            "batch_page": batch_page,
            "batch_size": batch_size,
        },
    )


# ---------- UI: Dashboard ----------
@app.get("/dashboard", response_class=HTMLResponse, tags=["ui"])
async def dashboard(request: Request, symbol: str = Query("", description="Filter by symbol")):
    db = SessionLocal()
    try:
        query = db.query(StockPrediction)
        if symbol:
            query = query.filter(StockPrediction.symbol.ilike(f"%{symbol}%"))
        results = query.order_by(StockPrediction.published_at.desc()).all()

        dashboard_items = []
        for item in results:
            dashboard_items.append(
                {
                    "symbol": item.symbol,
                    "estimated_change": item.estimated_change,
                    "analysis": item.analysis,
                    "published_at": item.published_at,
                    "reward_to_risk": getattr(item, "reward_to_risk", None),
                }
            )
    finally:
        db.close()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "predictions": dashboard_items,
            "filter": symbol,
            "total": len(results),
            "now": datetime.now,
        },
    )


# ---------- UI: Stock Details ----------
@app.get("/stock/{symbol}", response_class=HTMLResponse, tags=["ui"])
async def stock_details(request: Request, symbol: str):
    db = SessionLocal()
    try:
        predictions = (
            db.query(StockPrediction)
            .filter(StockPrediction.symbol == symbol)
            .order_by(StockPrediction.published_at.desc())
            .limit(10)
            .all()
        )
    finally:
        db.close()

    if not predictions:
        raise HTTPException(status_code=404, detail="No data found for this stock.")

    news = fetch_news_for_symbol(symbol, page_size=8)
    financials = fetch_financial_data(symbol)
    historical_impacts = fetch_historical_impacts(symbol)
    all_news = fetch_news_for_symbol(symbol, page_size=30)
    top_news = rank_news(all_news, top_n=5)
    summarized = summarize_news(top_news, max_sentences=2)

    return templates.TemplateResponse(
        "stock_detail.html",
        {
            "request": request,
            "symbol": symbol,
            "predictions": predictions,
            "news": news,
            "financials": financials,
            "historical_impacts": historical_impacts,
            "top_news": summarized,
        },
    )


# ---------- UI: Backtest (simple legacy dashboard) ----------
@app.get("/backtest-dashboard", response_class=HTMLResponse, tags=["ui"])
async def backtest_dashboard(request: Request):
    csv_path = os.path.join(BASE_DIR, "backtest_results.csv")
    if not os.path.exists(csv_path):
        return HTMLResponse(
            "<h2>Nu există încă rezultate de backtest. Rulează scriptul de backtest întâi!</h2>"
        )
    import pandas as pd

    df = pd.read_csv(csv_path)
    total = len(df)
    correct = (df["direction_correct"] == True).sum()
    avg_error = df["error"].mean()
    content = f"""
    <h1>Rezultate Backtest AI</h1>
    <p><b>Total predicții:</b> {total}</p>
    <p><b>Procent direcție corectă:</b> {100*correct/total:.2f}%</p>
    <p><b>Eroare medie (pct):</b> {avg_error:.2f}%</p>
    <table border='1' cellpadding='6' style='margin-top:20px;'>
      <tr>
        <th>Symbol</th><th>Date</th><th>AI Pred (%)</th><th>Real (%)</th><th>Error</th>
        <th>Corect?</th><th>Prob AI</th><th>Recomandare</th>
      </tr>
    """
    for _, row in df.head(200).iterrows():
        content += f"""
        <tr>
            <td>{row.get('symbol','')}</td>
            <td>{row.get('date','')}</td>
            <td>{row.get('ai_pct','')}</td>
            <td>{row.get('real_pct','')}</td>
            <td>{row.get('error','')}</td>
            <td style="color:{'lime' if row.get('direction_correct') else 'red'};">
                {"✔" if row.get('direction_correct') else "✗"}
            </td>
            <td>{row.get('ai_probability','')}</td>
            <td>{row.get('ai_recommendation','')}</td>
        </tr>
        """
    content += "</table>"
    return HTMLResponse(content)


# ---------- UI: Backtest Pro (calls backtest_v2.generate_report) ----------
@app.get("/backtest/pro", response_class=HTMLResponse, tags=["ui"])
async def backtest_pro(request: Request, horizon_days: int = Query(7, ge=3, le=21)):
    """
    Generate and serve the advanced HTML report from backtest_v2.generate_report().
    """
    try:
        from .backtest_v2 import generate_report
    except Exception as e:
        logger.error(f"[BT] backtest_v2 import failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Backtest module not available.")

    path = generate_report(horizon_days=horizon_days)
    # Serve the produced HTML file
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(html)
    except Exception as e:
        logger.error(f"[BT] failed to open report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Backtest report not found.")


# ---------- Alias compat: /backtest-pro -> redirect la /backtest/pro ----------
@app.get("/backtest-pro", include_in_schema=False)
async def backtest_pro_alias(
    request: Request,
    horizon: int = Query(7, ge=3, le=21),
    limit: int = Query(2000, ge=10, le=50000),
    symbols: str = Query("", description="Lista separată prin virgulă: AAPL,MSFT"),
    thr_abs: float = Query(1.0, ge=0.0, description="Prag absolut pct pt toy strategy"),
    thr_prob: float = Query(
        0.60, ge=0.0, le=1.0, description="Probabilitate (0..1) pt toy strategy"
    ),
):
    """
    Alias pentru vechiul buton /backtest-pro: redirecționează spre ruta canonică /backtest/pro.
    Dacă generate_report suportă parametri extra, îi folosim mai jos (în ruta canonică).
    Pentru alias alegem doar să redirectăm ca să unificăm logica într-un singur handler.
    """
    # Construim URL-ul canonic cu parametrii. Ruta canonică folosește 'horizon_days'.
    # Chiar dacă handler-ul canonic actual nu consumă limit/symbols/thr_abs/thr_prob,
    # e ok să-i trecem în query pentru viitor.
    query = (
        f"horizon_days={horizon}"
        f"&limit={limit}"
        f"&symbols={symbols}"
        f"&thr_abs={thr_abs}"
        f"&thr_prob={thr_prob}"
    )
    return RedirectResponse(url=f"/backtest/pro?{query}", status_code=307)


# ---------- API: force refresh a single symbol ----------
@app.post("/api/refresh/{symbol}", tags=["api"])
async def api_force_refresh(symbol: str):
    try:
        results = batch_process([symbol], force_refresh=True)
    except Exception as e:
        logger.error(f"[API] refresh {symbol} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Refresh failed.")

    if symbol in results:
        d = results[symbol]
        return {
            "symbol": symbol,
            "percent": d.get("percent"),
            "final_percent": d.get("final_percent"),
            "probability": d.get("final_probability", d.get("probability")),
            "explanation": d.get("explanation"),
            "date_pred": d.get("date_pred"),
        }
    raise HTTPException(status_code=500, detail="No prediction returned")


# ---------- API: force refresh all ----------
@app.post("/api/refresh_all", tags=["api"])
async def refresh_all(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    symbols = data.get("symbols") or [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
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
    ]
    try:
        results = batch_process(symbols, force_refresh=True)
    except Exception as e:
        logger.error(f"[API] refresh_all failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Refresh all failed.")

    return {"status": "ok", "count": len(results), "symbols": list(results.keys())}


# --- Assistant Q&A on codebase ---
from fastapi import HTTPException
import os


@app.post("/ops/ask_assistant", tags=["ops"])
async def ask_assistant(prompt: str = Body(..., embed=True)):
    """
    Întreabă Assistant-ul 'StockApp Engineer' folosind File Search pe codul repo-ului.
    Returnează răspunsul text.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        assistant_id = os.getenv("ASSISTANT_ID")
        if not assistant_id:
            raise RuntimeError("ASSISTANT_ID lipsă (.env)")

        # 1) creăm un thread
        thread = client.beta.threads.create()

        # 2) adăugăm user message
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)

        # 3) rulăm assistant-ul pe thread
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant_id
        )

        # 4) extragem ultimul mesaj
        msgs = client.beta.threads.messages.list(thread_id=thread.id)
        # ultimul mesaj al asistentului
        txts = []
        for m in msgs.data:
            if m.role == "assistant":
                for c in m.content:
                    if getattr(c, "type", None) == "text":
                        txts.append(c.text.value)
        answer = "\n\n".join(reversed(txts)) if txts else "(fără conținut)"
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assistant error: {e}")
