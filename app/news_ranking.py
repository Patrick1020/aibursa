# app/news_ranking.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

# lazy singletons
_FINBERT = {"tok": None, "model": None}


def _load_finbert():
    if _FINBERT["tok"] is None:
        _FINBERT["tok"] = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _FINBERT["model"] = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        ).eval()
    return _FINBERT["tok"], _FINBERT["model"]


def compute_sentiment(text: str) -> float:
    """
    Returns a finance-aware sentiment score in [-1, +1]
    (positive prob - negative prob) using FinBERT.
    """
    try:
        if not text:
            return 0.0
        tok, model = _load_finbert()
        with torch.no_grad():
            inputs = tok(text[:450], return_tensors="pt", truncation=True)
            logits = model(**inputs).logits[0].detach().numpy()
            exps = np.exp(logits - logits.max())
            probs = exps / exps.sum()
            # FinBERT label order: [negative, neutral, positive]
            score = float(probs[2] - probs[0])
            return score  # ~[-1,+1]
    except Exception:
        return 0.0


def rank_news(news_list, top_n=5, keywords=None):
    """
    Primește o listă de știri (dict cu title, description, published_at),
    returnează top_n știri relevante, sortate după scor.
    """
    if keywords is None:
        keywords = [
            "profit",
            "earnings",
            "loss",
            "guidance",
            "forecast",
            "dividend",
            "record",
            "buyback",
            "crash",
            "invest",
            "merge",
            "lawsuit",
            "upgrade",
            "downgrade",
            "acquisition",
            "split",
            "scandal",
        ]
    scored = []
    for news in news_list:
        title = (news.get("title") or "").lower()
        descr = (news.get("description") or "").lower()
        score = 0
        # 1. Keyword bonus (puncte în plus pentru fiecare keyword relevant)
        for kw in keywords:
            if kw in title or kw in descr:
                score += 3
        # 2. Sentiment bonus (pozitiv = +, negativ = -)
        sent = compute_sentiment(title + ". " + descr)
        score += sent * 2
        # 3. Recență (bonus dacă are dată, extra dacă azi/ieri)
        date = news.get("published_at", "")  # ex: "2024-05-18..."
        if date:
            score += 1
        # 4. Lungime titlu (mai scurt => știre de breaking)
        if 10 < len(title) < 60:
            score += 1
        # Poți extinde cu scor de engagement, social, etc.
        scored.append((score, news))
    scored = sorted(scored, key=lambda x: -x[0])
    # Returnează doar top_n (fără scor, doar dicturi de știre)
    return [news for (score, news) in scored[:top_n]]


def summarize_news(news_list, max_sentences=2):
    """
    Pentru fiecare știre, returnează un rezumat scurt (folosește doar primele max_sentences din descriere).
    Poți extinde cu LLM sau library de summarization dacă vrei ceva mai avansat.
    """
    summaries = []
    for news in news_list:
        descr = news.get("description") or ""
        summary = " ".join(descr.split(".")[:max_sentences]) + "."
        summaries.append(
            {
                "title": news.get("title", ""),
                "summary": summary,
                "published_at": news.get("published_at", ""),
                "url": news.get("url", "#"),
            }
        )
    return summaries


# EXTRAS: Poți crea funcții similare pentru macro, istoric, etc.
# def rank_macro_events(...): ...
# def rank_historical_impacts(...): ...
