import requests
import os
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def fetch_news(query="stock market", page_size=10):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json().get("articles", [])
    return []
