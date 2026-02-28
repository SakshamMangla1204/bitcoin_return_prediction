import numpy as np
import pandas as pd


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(
        "data/processed/Bitcoin Historical Data Processed.csv",
        parse_dates=["Date"],
    )


def main() -> None:
    df = load_clean_data()
    df = df.sort_values("Date").reset_index(drop=True)

    # 1-2. Load and create return
    df["return"] = df["Price"].pct_change()
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. Create SMA features
    df["SMA_7"] = df["Price"].rolling(window=7).mean()
    df["SMA_14"] = df["Price"].rolling(window=14).mean()

    # 4. Create volatility feature
    df["volatility_7"] = df["return"].rolling(window=7).std()

    # Keep only rows where base features are valid before creating lags/target.
    df = df.dropna(subset=["return", "SMA_7", "SMA_14", "volatility_7"]).reset_index(drop=True)

    # 5. Create lag features
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)
    df["return_lag3"] = df["return"].shift(3)

    # 6. Create target
    df["target"] = df["return"].shift(-1)

    # 7. Drop rows with NaN only in modeling columns.
    model_cols = [
        "Date",
        "Price",
        "return",
        "SMA_7",
        "SMA_14",
        "volatility_7",
        "return_lag1",
        "return_lag2",
        "return_lag3",
        "target",
    ]
    df = df.dropna(subset=model_cols)

    # 8. Print shape
    print(df.shape)

    # 9. Save model-ready data
    df.to_csv("data/processed/model_ready_data.csv", index=False)


if __name__ == "__main__":
    main()
