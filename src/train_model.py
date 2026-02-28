from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_processed_data
from evaluation import (
    evaluate_direction_classification,
    evaluate_predictions,
    plot_feature_importance,
    run_leakage_checks,
    run_regime_analysis,
)
from features import FEATURE_COLS, chronological_split, prepare_model_frame
from models import (
    LinearRegression,
    LogisticRegression,
    StandardScaler,
    XGBOOST_AVAILABLE,
    XGBRegressor,
    mean_absolute_error,
    save_artifact,
    tune_xgboost,
    walk_forward_validate_linear,
)


def main() -> None:
    df = load_processed_data()
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    model_df = prepare_model_frame(df, FEATURE_COLS)
    X = model_df[FEATURE_COLS]
    y = model_df["target"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Using features:", FEATURE_COLS)

    split, X_train, X_test, y_train, y_test = chronological_split(X, y, split_ratio=0.8)
    run_leakage_checks(model_df, split)
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit(X_train).transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    y_pred_linear = linear_model.predict(X_test_scaled)
    mae_linear = evaluate_predictions(y_test, y_pred_linear, "Linear (scaled)")

    vol_threshold = float(X_train["volatility_7"].median())
    run_regime_analysis(
        y_test,
        y_pred_linear,
        X_test["volatility_7"],
        vol_threshold,
        "Linear (scaled)",
    )

    baseline_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))
    buy_hold_return = float(np.prod(1 + y_test.to_numpy()) - 1)
    print("First 5 linear predictions:", y_pred_linear[:5])
    print(f"Baseline MAE (predict zeros): {baseline_mae:.6f}")
    print(f"Buy-and-hold return (test period): {buy_hold_return:.6f}")
    print("Linear model has signal." if mae_linear < baseline_mae else "Linear model is useless.")

    coef_df = pd.DataFrame({"feature": X.columns, "coefficient": linear_model.coef_})
    coef_df["importance"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("importance", ascending=False)
    print("Top 5 features by absolute coefficient:")
    print(coef_df[["feature", "coefficient"]].head(5).to_string(index=False))
    plot_feature_importance(
        coef_df[["feature", "importance"]],
        "Linear Regression Feature Importance (|coefficient|)",
        Path("reports/linear_feature_importance.png"),
    )
    top_feature = str(coef_df.iloc[0]["feature"])
    if top_feature == "volatility_7":
        print("Interpretation hint: volatility dominates -> regime-style behavior.")
    elif top_feature == "return_lag1":
        print("Interpretation hint: lag1 dominates -> momentum-style behavior.")

    y_dir_train = (y_train.to_numpy() > 0).astype(int)
    y_dir_test = (y_test.to_numpy() > 0).astype(int)
    dir_model = LogisticRegression(max_iter=1000, random_state=42)
    dir_model.fit(X_train_scaled, y_dir_train)
    y_dir_pred = dir_model.predict(X_test_scaled)
    evaluate_direction_classification(y_dir_test, y_dir_pred, "Direction Classifier")

    best_model = linear_model
    best_model_name = "linear_regression_scaled"
    best_mae = float(mae_linear)
    best_scaler = scaler
    best_xgb_params_output = None

    if XGBOOST_AVAILABLE:
        print("Running XGBoost hyperparameter tuning (walk-forward on train split)...")
        best_xgb_params, tuning_results = tune_xgboost(X_train, y_train)
        print("Best XGBoost params:", best_xgb_params)
        print("Top 5 XGBoost configs by walk-forward MAE:")
        print(tuning_results.head(5).to_string(index=False))

        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            **best_xgb_params,
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        mae_xgb = evaluate_predictions(y_test, y_pred_xgb, "XGBoost")
        run_regime_analysis(
            y_test,
            y_pred_xgb,
            X_test["volatility_7"],
            vol_threshold,
            "XGBoost",
        )
        print("First 5 XGBoost predictions:", y_pred_xgb[:5])

        xgb_importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": xgb_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        print("Top 5 XGBoost feature importances:")
        print(xgb_importance_df.head(5).to_string(index=False))
        plot_feature_importance(
            xgb_importance_df,
            "XGBoost Feature Importance",
            Path("reports/xgboost_feature_importance.png"),
        )

        if mae_xgb < mae_linear:
            print("Better model by MAE: XGBoost")
            best_model = xgb_model
            best_model_name = "xgboost"
            best_mae = float(mae_xgb)
            best_scaler = None
        else:
            print("Better model by MAE: Linear Regression")
        best_xgb_params_output = best_xgb_params
    else:
        print("XGBoost not installed. Install `xgboost` to run tree-based comparison.")

    full_scaler = StandardScaler()
    full_scaler.fit(X)
    print("Final scaler fitted on full feature dataset.")

    print("Running walk-forward validation (Linear Regression)...")
    wf_linear_mae, wf_baseline_mae = walk_forward_validate_linear(X, y, n_splits=5)
    print(f"Walk-forward avg Linear MAE: {wf_linear_mae:.6f}")
    print(f"Walk-forward avg Baseline MAE: {wf_baseline_mae:.6f}")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_artifact(best_model, models_dir / "best_model.pkl")
    save_artifact(best_scaler, models_dir / "scaler.pkl")
    save_artifact(
        {
            "model_name": best_model_name,
            "mae": best_mae,
            "best_xgb_params": best_xgb_params_output,
        },
        models_dir / "model_metadata.pkl",
    )
    print("Saved best model to: models/best_model.pkl")
    print("Saved scaler to: models/scaler.pkl")
    print("Saved metadata to: models/model_metadata.pkl")


if __name__ == "__main__":
    main()

