from sqlalchemy import Column, Integer, String, Float, DateTime, Index, Text
from .database import Base
from datetime import datetime, timezone


def utcnow():
    # datetime aware, în UTC
    return datetime.now(timezone.utc)


class StockPrediction(Base):
    __tablename__ = "stock_predictions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(String)
    url = Column(String)
    analysis = Column(String)
    symbol = Column(String, index=True)
    estimated_change = Column(Float)
    eps = Column(Float, nullable=True)  # Earnings Per Share
    pe_ratio = Column(Float, nullable=True)  # P/E Ratio
    revenue = Column(Float, nullable=True)  # Venituri
    net_income = Column(Float, nullable=True)  # Profit net
    published_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    # ==== ADĂUGĂ AICI câmpurile pentru AI scoring ====
    probability = Column(Float, nullable=True)
    reward_to_risk = Column(Float, nullable=True)
    recommendation = Column(String, nullable=True)
    ai_full_text = Column(Text, nullable=True)  # Text mare, dacă vrei tot output-ul AI
    trade_outcome = Column(String, nullable=True)
    prob_justification = Column(Text, nullable=True)
    reco_reason = Column(Text, nullable=True)
    short_justification = Column(Text, nullable=True)
    actual_price = Column(Float, nullable=True)  # prețul curent la momentul predicției
    final_percent = Column(Float, nullable=True)  # % blend (LLM+ML) folosit în UI
    final_probability = Column(Float, nullable=True)  # prob. calibrată/blend folosită în UI
    ci_low_pct = Column(Float, nullable=True)  # banda inferioară (pct) pentru 7d
    ci_high_pct = Column(Float, nullable=True)  # banda superioară (pct) pentru 7d
    p20 = Column(Float, nullable=True)
    p80 = Column(Float, nullable=True)
    price_low = Column(Float, nullable=True)
    price_high = Column(Float, nullable=True)

    __table_args__ = (Index("ix_symbol_date", "symbol", "published_at"),)


class APICache(Base):
    __tablename__ = "api_cache"

    id = Column(Integer, primary_key=True, index=True)
    api_name = Column(String, index=True)
    request_key = Column(String, index=True)
    response_data = Column(Text)
    timestamp = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class HistoricalImpact(Base):
    __tablename__ = "historical_impacts"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    news_title = Column(String)
    news_date = Column(DateTime(timezone=True), index=True)
    price_before = Column(Float)
    price_after = Column(Float)
    impact_percentage = Column(Float)
    sentiment = Column(String)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
