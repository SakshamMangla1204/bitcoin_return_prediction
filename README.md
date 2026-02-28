# Bitcoin Return Prediction

End-to-end pipeline for Bitcoin next-day return modeling with:
- data cleaning
- feature engineering
- regression + direction classification
- backtest + risk metrics
- regime analysis
- modular production-style code structure

## Project Structure

```text
src/
├── backtest.py
├── data_loader.py
├── data_preprocessing.py
├── evaluation.py
├── feature_engineering.py
├── features.py
├── models.py
└── train_model.py

data/
  raw/
  processed/
models/
reports/
```

## Pipeline

1. **Data preprocessing**  
   Loads raw Bitcoin CSV, cleans numeric fields, converts dates, sorts chronologically, saves:
   - `data/processed/Bitcoin Historical Data Processed.csv`

2. **Feature engineering**  
   Builds:
   - `return`
   - `SMA_7`, `SMA_14`
   - `volatility_7`
   - `return_lag1`, `return_lag2`, `return_lag3`
   - `target` = next-day return  
   Saves:
   - `data/processed/model_ready_data.csv`

3. **Model training/evaluation (`train_model.py`)**
   - Chronological 80/20 split (no shuffle)
   - Linear regression baseline model (scaled)
   - Optional XGBoost tuning (if installed)
   - Direction classifier metrics
   - Regime analysis (high/low volatility)
   - Backtest + risk metrics (cumulative return, max drawdown, Sharpe, win rate)
   - Walk-forward validation
   - Saves best artifacts:
     - `models/best_model.pkl`
     - `models/scaler.pkl`
     - `models/model_metadata.pkl`

## What Each Module Now Represents

- `data_loader.py`  
  Responsible only for:
  reading raw or processed data, and returning a DataFrame.  
  This follows the Single Responsibility Principle.

- `feature_engineering.py`  
  Handles:
  lag features, rolling volatility, and target shift.  
  This separates modeling logic from feature logic, which is important in industry.

- `features.py`  
  Contains:
  `FEATURE_COLS`, helper functions like `chronological_split`, and frame preparation logic.  
  This centralizes feature configuration.

- `models.py`  
  Contains:
  linear regression wrapper, logistic regression, XGBoost, scaling, model tuning, and artifact save logic.  
  This isolates the modeling layer with clean separation.

- `evaluation.py`  
  Handles:
  MAE, RMSE, direction accuracy, and strategy-focused evaluation.  
  Separating evaluation is a mature design pattern.

- `backtest.py`  
  Holds financial backtest logic separate from ML logic (returns path and risk metrics).  
  This is a strong production/research structure signal.

- `train_model.py`  
  Acts as the orchestrator.
  It connects:
  data, features, models, evaluation, and saving.  
  This is how production training scripts are typically structured.

## Run

From project root:

```bash
python3 src/data_preprocessing.py
python3 src/feature_engineering.py
python3 src/train_model.py
```

## Metrics Reported

- Regression: MAE, RMSE
- Direction: Direction Accuracy, Accuracy, Precision, Recall, Confusion Matrix
- Backtest: Strategy cumulative return, Buy & Hold return
- Risk: Max Drawdown, Sharpe Ratio, Win Rate
- Stability: Walk-forward average MAE vs baseline MAE
- Regime: Separate model performance in high-vol and low-vol regimes

## Optional Dependencies

The code has fallbacks for missing ML libraries, but for full capability install:

```bash
pip install scikit-learn xgboost matplotlib joblib
```
