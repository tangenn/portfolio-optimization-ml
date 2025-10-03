def create_labels(ticker, log_ret_path="data/processed", feature_path="data/features", save_path="data/modeling"):
    import pandas as pd
    import os

    os.makedirs(save_path, exist_ok=True)

    # Load log returns and features
    log_returns = pd.read_csv(f"{log_ret_path}/{ticker}_log_returns.csv", index_col="Date", parse_dates=True)
    features = pd.read_csv(f"{feature_path}/{ticker}_features.csv", index_col="Date", parse_dates=True)

    # Create target: 1 if next day's return > 0, else 0
    log_returns["Target"] = (log_returns["Log Return"].shift(-1) > 0).astype(int)
    log_returns.dropna(inplace=True)

    # Find common dates
    common_dates = log_returns.index.intersection(features.index)

    # Align both to common dates
    log_returns = log_returns.loc[common_dates]
    features = features.loc[common_dates]

    # Combine features + target
    data = features.copy()
    data["Target"] = log_returns["Target"]

    # Save for modeling
    data.to_csv(f"{save_path}/{ticker}_model_data.csv")

    return data