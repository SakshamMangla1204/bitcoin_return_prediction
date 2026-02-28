import numpy as np


def compute_strategy_returns(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sign(y_pred) * y_true


def compute_risk_metrics(strategy_returns: np.ndarray) -> dict[str, float]:
    cumulative_return = float(np.prod(1 + strategy_returns) - 1)
    equity_curve = np.cumprod(1 + strategy_returns)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_peak - 1
    max_drawdown = float(np.min(drawdown))
    daily_std = float(np.std(strategy_returns))
    sharpe_ratio = 0.0
    if daily_std >= 1e-12:
        sharpe_ratio = float(np.mean(strategy_returns) / daily_std * np.sqrt(252))
    win_rate = float(np.mean(strategy_returns > 0))
    return {
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
    }

