import pandas as pd
from pathlib import Path

def compute_expected_returns(
    tickers,
    pred_path=Path("data/predictions"),
    return_path=Path("data/processed"),
    save_path=Path("data/expected")
):
    save_path.mkdir(parents=True, exist_ok=True)
    avg_up_returns = {}
    prob_df = pd.DataFrame()

    for ticker in tickers:
        print(f"🔍 Processing {ticker}...")

        # --- Load prediction CSV ---
        pred_file = pred_path / f"{ticker}_predictions.csv"
        pred_df = pd.read_csv(pred_file, parse_dates=["Date"])
        pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.date

        # --- Load log return CSV ---
        return_file = return_path / f"{ticker}_log_returns.csv"
        return_df = pd.read_csv(return_file, parse_dates=["Date"])
        return_df["Date"] = pd.to_datetime(return_df["Date"]).dt.date

        # Force rename the column reliably
        if "Log Return" not in return_df.columns:
            raise ValueError(f"[{ticker}] ❌ 'Log Return' column not found in: {return_file}")
        return_df = return_df.rename(columns={"Log Return": "True_Log_Return"})

        # --- Merge ---
        df = pd.merge(pred_df, return_df, on="Date", how="inner")

        print(f"✅ Merged columns: {df.columns.tolist()}")

        if "True_Log_Return_y" in df.columns:
            df = df.rename(columns={"True_Log_Return_y": "True_Log_Return"})
        elif "True_Log_Return" in df.columns:
            pass  # it's already correct
        else:
            raise ValueError(f"[{ticker}] ❌ 'True_Log_Return' column not found after merge")

        # --- Step 1: Calculate avg return on actual up days ---
        up_returns = df.loc[df["True_Log_Return"] > 0, "True_Log_Return"]
        avg_up_returns[ticker] = up_returns.mean()

        # --- Step 2: Collect predicted prob ---
        prob_df[ticker] = df["Predicted_Prob"].reset_index(drop=True)

    # --- Save to CSV ---
    avg_return_series = pd.Series(avg_up_returns)
    avg_return_series.to_csv(save_path / "avg_up_returns.csv", header=["Avg_Up_Return"])

    prob_df.to_csv(save_path / "predicted_probs.csv", index=False)

    print("✅ Done! Saved to /data/expected/")