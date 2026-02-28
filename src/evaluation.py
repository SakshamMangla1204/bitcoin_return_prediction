from pathlib import Path

import numpy as np
import pandas as pd

from backtest import compute_risk_metrics, compute_strategy_returns
from models import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    MATPLOTLIB_AVAILABLE = False


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, label: str) -> float:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    direction_accuracy = float(np.mean((y_pred > 0) == (y_true.to_numpy() > 0)))
    strategy_returns = compute_strategy_returns(y_true.to_numpy(), y_pred)
    risk = compute_risk_metrics(strategy_returns)

    print(f"{label} MAE: {mae:.6f}")
    print(f"{label} RMSE: {rmse:.6f}")
    print(f"{label} Direction Accuracy: {direction_accuracy:.4f}")
    print(f"{label} Strategy Cumulative Return: {risk['cumulative_return']:.6f}")
    print(f"{label} Max Drawdown: {risk['max_drawdown']:.6f}")
    print(f"{label} Sharpe Ratio: {risk['sharpe_ratio']:.4f}")
    print(f"{label} Win Rate: {risk['win_rate']:.4f}")
    return float(mae)


def evaluate_direction_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> None:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"{label} Accuracy: {acc:.4f}")
    print(f"{label} Precision: {prec:.4f}")
    print(f"{label} Recall: {rec:.4f}")
    print(f"{label} Confusion Matrix:\n{cm}")


def run_regime_analysis(
    y_test: pd.Series,
    y_pred: np.ndarray,
    volatility_test: pd.Series,
    volatility_threshold: float,
    label: str,
) -> None:
    high_mask = volatility_test.to_numpy() >= volatility_threshold
    low_mask = ~high_mask

    print(f"{label} Regime split threshold (volatility_7): {volatility_threshold:.6f}")
    print(
        f"{label} Regime sample sizes -> high: {int(np.sum(high_mask))}, "
        f"low: {int(np.sum(low_mask))}"
    )

    if np.sum(high_mask) > 0:
        evaluate_predictions(y_test.iloc[high_mask], y_pred[high_mask], f"{label} [High Vol]")
    if np.sum(low_mask) > 0:
        evaluate_predictions(y_test.iloc[low_mask], y_pred[low_mask], f"{label} [Low Vol]")


def run_leakage_checks(model_df: pd.DataFrame, split: int) -> None:
    print(f"Leakage check - dates sorted ascending: {model_df['Date'].is_monotonic_increasing}")
    same_day_equal_ratio = np.mean(
        np.isclose(model_df["target"].to_numpy(), model_df["return"].to_numpy())
    )
    same_day_corr = model_df["target"].corr(model_df["return"])
    print("Leakage check - same-day target equality ratio:", f"{same_day_equal_ratio:.4f}")
    print(f"Leakage check - same-day target/return correlation: {same_day_corr:.4f}")
    train_last_date = model_df["Date"].iloc[split - 1]
    test_first_date = model_df["Date"].iloc[split]
    print(f"Leakage check - train end date: {train_last_date.date()}")
    print(f"Leakage check - test start date: {test_first_date.date()}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed. Skipping feature importance plot.")
        return

    plot_df = importance_df.sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 4.8))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to: {output_path}")

