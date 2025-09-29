import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_prices(tickers, start, end, save_path="data/raw"):
    os.makedirs(save_path, exist_ok=True)
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end)
        df.reset_index(inplace=True)
        df.to_csv(f"{save_path}/{ticker}.csv")

def save_log_returns(tickers, raw_path="data/raw", save_path="data/processed"):
    os.makedirs(save_path, exist_ok=True)
    for ticker in tickers:
        print(f"Processing {ticker}...")
        path = os.path.join(raw_path, f"{ticker}.csv")
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        close = pd.to_numeric(df["Close"], errors="coerce")
        log_returns = np.log(close / close.shift(1)).dropna()
        log_returns.name = "Log Return"
        log_returns.to_csv(f"{save_path}/{ticker}_log_returns.csv")