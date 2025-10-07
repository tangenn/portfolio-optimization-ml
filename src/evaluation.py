import pandas as pd
import numpy as np
from src.sharpe_optimizer import optimize_sharpe

def generate_dynamic_weights(
    returns: pd.DataFrame,
    static_expected_returns: pd.Series,
    window: int = 60,
    min_periods: int = 30,
) -> pd.DataFrame:
    weights = []
    dates = []
    tickers = returns.columns.tolist()

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i - window:i]
        current_date = returns.index[i]

        if window_returns.shape[0] < min_periods:
            continue

        cov_matrix = window_returns.cov()

        try:
            mu_today = static_expected_returns.loc[current_date]  # ✅ Extract row as Series
            w = optimize_sharpe(mu_today, cov_matrix)
            weights.append(w.values)
            dates.append(current_date)
        except Exception as e:
            print(f"⚠️ Skipped {current_date}: {e}")
            continue

    return pd.DataFrame(weights, columns=tickers, index=dates)

def compute_portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    aligned_returns = returns.loc[weights.index]
    return (aligned_returns * weights).sum(axis=1)

def compute_metrics(returns: pd.Series):
    ann_return = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std()
    return {
        "Annual Return": ann_return,
        "Volatility": ann_vol,
        "Sharpe Ratio": sharpe
    }