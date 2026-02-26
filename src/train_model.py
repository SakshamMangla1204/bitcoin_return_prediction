import math
import numpy as np
import pandas as pd
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ModuleNotFoundError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    class LinearRegression:  # type: ignore[no-redef]
        def fit(self, X, y):
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            X_design = np.c_[np.ones(X_arr.shape[0]), X_arr]
            beta = np.linalg.lstsq(X_design, y_arr, rcond=None)[0]
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
            return self

        def predict(self, X):
            X_arr = np.asarray(X, dtype=float)
            return self.intercept_ + X_arr @ self.coef_

    class StandardScaler:  # type: ignore[no-redef]
        def fit(self, X):
            X_arr = np.asarray(X, dtype=float)
            self.mean_ = X_arr.mean(axis=0)
            self.scale_ = X_arr.std(axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            X_arr = np.asarray(X, dtype=float)
            return (X_arr - self.mean_) / self.scale_

    def mean_absolute_error(y_true, y_pred):  # type: ignore[no-redef]
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.mean(np.abs(y_true - y_pred))

    def mean_squared_error(y_true, y_pred):  # type: ignore[no-redef]
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.mean((y_true - y_pred) ** 2)


def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(
        "data/processed/model_ready_data.csv",
        parse_dates=["Date"],
    )


def main() -> None:
    df = load_processed_data()
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    y = df["target"]
    X = df.drop(columns=["target", "Date"])
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    n = len(df)
    split = int(0.8 * n)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    model_unscaled = LinearRegression()
    model_unscaled.fit(X_train, y_train)
    y_pred_unscaled = model_unscaled.predict(X_test)
    mae_unscaled = mean_absolute_error(y_test, y_pred_unscaled)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit(X_train).transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_scaled = LinearRegression()
    model_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
    rmse_scaled = math.sqrt(mean_squared_error(y_test, y_pred_scaled))

    baseline_pred = np.zeros(len(y_test))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print("First 5 predictions (scaled model):", y_pred_scaled[:5])
    print(f"MAE (unscaled): {mae_unscaled:.6f}")
    print(f"MAE (scaled): {mae_scaled:.6f}")
    print(f"RMSE (scaled): {rmse_scaled:.6f}")
    print(f"Baseline MAE (predict zeros): {baseline_mae:.6f}")

    if mae_scaled < baseline_mae:
        print("Model has signal.")
    else:
        print("Model is useless.")

    coef_df = pd.DataFrame(
        {"feature": X.columns, "coefficient": model_scaled.coef_}
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    top_feature = coef_df.iloc[0]
    print(
        "Strongest feature influence:",
        f"{top_feature['feature']} (coef={top_feature['coefficient']:.6f})",
    )
    print("Top 5 features by absolute coefficient:")
    print(coef_df[["feature", "coefficient"]].head(5).to_string(index=False))

    if XGBOOST_AVAILABLE:
        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
        xgb_rmse = math.sqrt(mean_squared_error(y_test, y_pred_xgb))
        print(f"XGBoost MAE: {xgb_mae:.6f}")
        print(f"XGBoost RMSE: {xgb_rmse:.6f}")

        if xgb_mae < mae_scaled:
            print("Better model by MAE: XGBoost")
        else:
            print("Better model by MAE: Linear Regression")
    else:
        print("XGBoost not installed. Install `xgboost` to run tree-based comparison.")

    full_scaler = StandardScaler()
    full_scaler.fit(X)
    print("Final scaler fitted on full feature dataset.")


if __name__ == "__main__":
    main()
