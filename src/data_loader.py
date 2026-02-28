import pandas as pd


def load_processed_data(path: str = "data/processed/model_ready_data.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])

