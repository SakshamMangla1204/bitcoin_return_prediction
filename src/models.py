import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore[import-not-found]

    JOBLIB_AVAILABLE = True
except ModuleNotFoundError:
    JOBLIB_AVAILABLE = False

try:
    from xgboost import XGBRegressor  # type: ignore[import-not-found]

    XGBOOST_AVAILABLE = True
except ModuleNotFoundError:
    XGBRegressor = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
    from sklearn.metrics import (  # type: ignore[import-not-found]
        accuracy_score,
        confusion_matrix,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]
except ModuleNotFoundError:
    class LinearRegression:  # type: ignore[no-redef]
        def fit(self, X, y):
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            X_design = np.c_[np.ones(X_arr.shape[0]), X_arr]
            beta = np.linalg.lstsq(X_design, y_arr, rcond=1e-10)[0]
            self.intercept_ = beta[0]
            self.coef_ = np.clip(beta[1:], -1e3, 1e3)
            return self

        def predict(self, X):
            X_arr = np.asarray(X, dtype=float)
            coef = np.nan_to_num(self.coef_, nan=0.0, posinf=0.0, neginf=0.0)
            intercept = float(
                np.nan_to_num(self.intercept_, nan=0.0, posinf=0.0, neginf=0.0)
            )
            y_pred = intercept + np.sum(X_arr * coef, axis=1)
            return np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    class StandardScaler:  # type: ignore[no-redef]
        def fit(self, X):
            X_arr = np.asarray(X, dtype=float)
            self.mean_ = X_arr.mean(axis=0)
            self.scale_ = X_arr.std(axis=0)
            self.scale_[~np.isfinite(self.scale_)] = 1.0
            self.scale_[self.scale_ < 1e-8] = 1.0
            return self

        def transform(self, X):
            X_arr = np.asarray(X, dtype=float)
            z = (X_arr - self.mean_) / self.scale_
            return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    class LogisticRegression:  # type: ignore[no-redef]
        def __init__(self, max_iter=1000, random_state=None, learning_rate=0.1):
            self.max_iter = int(max_iter)
            self.random_state = random_state
            self.learning_rate = learning_rate

        def fit(self, X, y):
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            n_samples, n_features = X_arr.shape
            self.coef_ = np.zeros(n_features, dtype=float)
            self.intercept_ = 0.0
            for _ in range(self.max_iter):
                z = np.sum(X_arr * self.coef_, axis=1) + self.intercept_
                z = np.clip(z, -20, 20)
                p = 1.0 / (1.0 + np.exp(-z))
                error = p - y_arr
                grad_w = np.sum(X_arr * error[:, None], axis=0) / n_samples
                grad_b = np.mean(error)
                self.coef_ -= self.learning_rate * grad_w
                self.intercept_ -= self.learning_rate * grad_b
            return self

        def predict(self, X):
            X_arr = np.asarray(X, dtype=float)
            z = np.sum(X_arr * self.coef_, axis=1) + self.intercept_
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            z = np.clip(z, -20, 20)
            p = 1.0 / (1.0 + np.exp(-z))
            return (p >= 0.5).astype(int)

    def mean_absolute_error(y_true, y_pred):  # type: ignore[no-redef]
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.mean(np.abs(y_true - y_pred))

    def mean_squared_error(y_true, y_pred):  # type: ignore[no-redef]
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.mean((y_true - y_pred) ** 2)

    def accuracy_score(y_true, y_pred):  # type: ignore[no-redef]
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        return float(np.mean(y_true == y_pred))

    def precision_score(y_true, y_pred, zero_division=0):  # type: ignore[no-redef]
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        denom = tp + fp
        if denom == 0:
            return float(zero_division)
        return float(tp / denom)

    def recall_score(y_true, y_pred, zero_division=0):  # type: ignore[no-redef]
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        denom = tp + fn
        if denom == 0:
            return float(zero_division)
        return float(tp / denom)

    def confusion_matrix(y_true, y_pred):  # type: ignore[no-redef]
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])


def save_artifact(obj, path: Path) -> None:
    if JOBLIB_AVAILABLE:
        joblib.dump(obj, path)
        return
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def walk_forward_validate_linear(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple[float, float]:
    n = len(X)
    initial_train_size = int(0.5 * n)
    test_size = (n - initial_train_size) // n_splits
    linear_maes = []
    baseline_maes = []

    for fold in range(n_splits):
        train_end = initial_train_size + fold * test_size
        test_end = n if fold == n_splits - 1 else train_end + test_size
        X_train_fold = X.iloc[:train_end]
        y_train_fold = y.iloc[:train_end]
        X_test_fold = X.iloc[train_end:test_end]
        y_test_fold = y.iloc[train_end:test_end]
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit(X_train_fold).transform(X_train_fold)
        X_test_fold_scaled = scaler_fold.transform(X_test_fold)
        model_fold = LinearRegression()
        model_fold.fit(X_train_fold_scaled, y_train_fold)
        y_pred_fold = model_fold.predict(X_test_fold_scaled)
        linear_maes.append(mean_absolute_error(y_test_fold, y_pred_fold))
        baseline_maes.append(mean_absolute_error(y_test_fold, np.zeros(len(y_test_fold))))

    return float(np.mean(linear_maes)), float(np.mean(baseline_maes))


def walk_forward_mae_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int = 4,
) -> tuple[float, float]:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost is not available in this environment.")

    n = len(X)
    initial_train_size = int(0.5 * n)
    test_size = (n - initial_train_size) // n_splits
    maes = []
    for fold in range(n_splits):
        train_end = initial_train_size + fold * test_size
        test_end = n if fold == n_splits - 1 else train_end + test_size
        X_train_fold = X.iloc[:train_end]
        y_train_fold = y.iloc[:train_end]
        X_test_fold = X.iloc[train_end:test_end]
        y_test_fold = y.iloc[train_end:test_end]
        model_fold = XGBRegressor(
            objective="reg:squarederror",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            **params,
        )
        model_fold.fit(X_train_fold, y_train_fold)
        y_pred_fold = model_fold.predict(X_test_fold)
        maes.append(mean_absolute_error(y_test_fold, y_pred_fold))
    return float(np.mean(maes)), float(np.std(maes))


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    max_depth_grid = [2, 3, 4, 5, 6]
    learning_rate_grid = [0.01, 0.03, 0.05, 0.1]
    n_estimators_grid = [100, 200, 300, 500]
    results = []

    for max_depth, learning_rate, n_estimators in product(
        max_depth_grid, learning_rate_grid, n_estimators_grid
    ):
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
        }
        avg_mae, std_mae = walk_forward_mae_xgb(X_train, y_train, params, n_splits=4)
        results.append({**params, "wf_avg_mae": avg_mae, "wf_std_mae": std_mae})

    results_df = pd.DataFrame(results).sort_values(
        ["wf_avg_mae", "wf_std_mae"], ascending=[True, True]
    )
    best_params = (
        results_df.iloc[0][["max_depth", "learning_rate", "n_estimators"]].to_dict()
    )
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["learning_rate"] = float(best_params["learning_rate"])
    return best_params, results_df
