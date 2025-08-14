import requests
import os
from dotenv import load_dotenv

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def get_historical_prices(symbol, start_date, days=7):
    # Folosim Alpha Vantage TIME_SERIES_DAILY_ADJUSTED
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"
    r = requests.get(url)
    if r.status_code != 200:
        return []
    data = r.json().get("Time Series (Daily)", {})
    prices = []
    # Extragem preturile dupa start_date pentru urmatoarele days zile
    sorted_dates = sorted(data.keys())
    for date in sorted_dates:
        if date >= start_date and len(prices) < days:
            prices.append(float(data[date]["4. close"]))
    return prices
