import pandas as pd

def compute_moving_average(series, window=10):
    return series.rolling(window=window).mean()

def compute_volatility(series, window=10):
    return series.rolling(window=window).std()

def compute_momentum(series, window=5):
    return series.diff(window)

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi