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

import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def generate_prediction_files(tickers, model_cls=XGBClassifier, model_name="xgb"):
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        print(f"🔍 Processing {ticker}...")
        data = pd.read_csv(f"data/modeling/{ticker}_model_data.csv", parse_dates=["Date"], index_col="Date")

        X = data.drop(columns=["Target"])
        y = data["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = model_cls(use_label_encoder=False, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)

        # Predict class and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (up)

        # Load true log returns
        ret_df = pd.read_csv(f"data/processed/{ticker}_log_returns.csv", parse_dates=["Date"], index_col="Date")
        log_returns = ret_df.loc[X_test.index, "Log Return"]

        # Save prediction file
        result = pd.DataFrame({
            "Date": X_test.index,
            "Predicted_Prob": y_prob,
            "Predicted_Class": y_pred,
            "True_Log_Return": log_returns
        })
        result.to_csv(pred_dir / f"{ticker}_predictions.csv", index=False)

        print(f"✅ Saved to data/predictions/{ticker}_predictions.csv")