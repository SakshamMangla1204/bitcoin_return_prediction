import pandas as pd
from typing import List, Optional


FEATURE_COLS = ["return", "volatility_7", "return_lag1", "return_lag2", "return_lag3"]


def prepare_model_frame(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    cols = feature_cols or FEATURE_COLS
    model_df = df[["Date", *cols, "target"]].copy()
    model_df = model_df.dropna().sort_values("Date").reset_index(drop=True)
    return model_df


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    split_ratio: float = 0.8,
) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split = int(split_ratio * len(X))
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    return split, X_train, X_test, y_train, y_test
