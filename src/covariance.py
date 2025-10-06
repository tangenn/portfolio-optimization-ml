import pandas as pd
from pathlib import Path

def compute_covariance_matrix(tickers, log_return_path=Path("data/processed")):
    returns_df = pd.DataFrame()

    for ticker in tickers:
        df = pd.read_csv(log_return_path / f"{ticker}_log_returns.csv", parse_dates=["Date"])
        df = df.rename(columns={"Log Return": ticker})
        returns_df[ticker] = df.set_index("Date")[ticker]

    # Drop rows with any missing values
    returns_df.dropna(inplace=True)

    # Compute covariance matrix
    cov_matrix = returns_df.cov()

    return cov_matrix