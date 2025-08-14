import yfinance as yf

tickers = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "^VIX"]
for t in tickers:
    df = yf.Ticker(t).history(period="8y", interval="1d", auto_adjust=True)
    print(t, len(df))
