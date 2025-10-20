"""
train_and_forecast.py
---------------------
AI-Based Rice Growth & Yield Forecasting for Mwea Irrigation Scheme
Author: [Your Name]
Date: [Auto Generated]

This script:
✅ Reads yield data (CSV)
✅ Extracts optional raster-derived features
✅ Creates lag/time-based features
✅ Trains SARIMAX and LightGBM models
✅ Forecasts monthly rice yield through December 2030
✅ Saves metrics, trained models, and forecast CSV outputs
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import lightgbm as lgb
import rasterio

# ========== Utility Functions ==========

def safe_rmse(y_true, y_pred):
    """Compute RMSE safely."""
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))

def metrics_df(y_true, y_pred):
    """Return MAE, RMSE, and MAPE."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = safe_rmse(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}


# ========== Raster Helper ==========
def Raster_mean_value(Raster_path):
    """Compute mean pixel value of a raster band."""
    try:
        with rasterio.open(Raster_path) as src:
            arr = src.read(1, masked=True)
            data = arr.compressed()
            return float(np.nanmean(data))
    except Exception as e:
        print(f"⚠️ Could not read raster {Raster_path}: {e}")
        return np.nan


# ========== Data Preprocessing ==========

def preprocess_data(csv_path, date_col="date", target="yield_tonnes"):
    """Load and clean yield CSV."""
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    df = df.asfreq("MS")  # ensure monthly frequency
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["t"] = np.arange(len(df))
    return df


def create_lag_features(df, target="yield_tonnes", lags=[1, 2, 3], rolls=[3, 6]):
    """Create lag and rolling mean features."""
    df = df.copy()
    for l in lags:
        df[f"{target}_lag_{l}"] = df[target].shift(l)
    for r in rolls:
        df[f"{target}_roll_{r}"] = df[target].shift(1).rolling(r, min_periods=1).mean()
    return df


# ========== Models ==========

def persistence_forecast(train, val_index, target):
    """Simple baseline: last known value."""
    preds = []
    for _ in val_index:
        preds.append(train[target].iloc[-1])
    return np.array(preds)


def sarimax_forecast(train, val, target, seasonal_period=12):
    """Train a simple SARIMAX model."""
    y_train = train[target].astype(float)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, seasonal_period)
    model = sm.tsa.SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.get_forecast(steps=len(val)).predicted_mean
    return res, preds


def train_lightgbm(train, val, features, target):
    """Train a LightGBM regressor."""
    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "verbosity": -1,
        "n_estimators": 200
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(train[features], train[target],
              eval_set=[(val[features], val[target])],
              eval_metric="rmse", verbose=False)
    return model


def iterative_lightgbm_forecast(model, history, features, target, future_dates):
    """Generate multi-step forecasts iteratively."""
    df_hist = history.copy()
    preds = []

    for date in future_dates:
        row = {
            "month": date.month,
            "year": date.year,
            "t": df_hist["t"].iloc[-1] + 1
        }
        for l in [1, 2, 3]:
            row[f"{target}_lag_{l}"] = df_hist[target].iloc[-l]
        for r in [3, 6]:
            row[f"{target}_roll_{r}"] = df_hist[target].iloc[-r:].mean()

        X = pd.DataFrame([row])[features]
        y_pred = model.predict(X)[0]
        preds.append(y_pred)
        row[target] = y_pred
        df_hist = pd.concat([df_hist, pd.DataFrame(row, index=[date])])

    return np.array(preds)


# ========== Pipeline Runner ==========

def run_pipeline():
    """Main pipeline: preprocess, train, forecast."""
    print("🚀 Starting Rice Yield Forecasting Pipeline")

    csv_path = "CSV/Mwea_data.csv"
    raster_dir = "Rasters"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = preprocess_data(csv_path)
    df = create_lag_features(df, "yield_tonnes")

    # Optional raster feature extraction
    Raster_features = {}
    if os.path.exists(raster_dir):
        for file in os.listdir(raster_dir):
            if file.lower().endswith(".tif"):
                mean_val = Raster_mean_value(os.path.join(raster_dir, file))
                feature_name = os.path.splitext(file)[0].replace(" ", "_")
                Raster_features[feature_name] = mean_val
                df[feature_name] = mean_val
        print(f"🛰️ Extracted {len(Raster_features)} raster features")

    # Drop NA rows for modeling
    df = df.dropna()

    # Split data (last 12 months validation)
    val_months = 12
    train = df.iloc[:-val_months]
    val = df.iloc[-val_months:]

    target = "yield_tonnes"

    # --- Persistence baseline ---
    pers_preds = persistence_forecast(train, val.index, target)
    pers_metrics = metrics_df(val[target], pers_preds)

    # --- SARIMAX ---
    sarimax_model, sarimax_preds = sarimax_forecast(train, val, target)
    sarimax_metrics = metrics_df(val[target], sarimax_preds)
    joblib.dump(sarimax_model, os.path.join(out_dir, "sarimax_model.pkl"))

    # --- LightGBM ---
    features = ['month', 'year', 't'] + [f"{target}_lag_{i}" for i in [1,2,3]] + [f"{target}_roll_{i}" for i in [3,6]]
    features += list(Raster_features.keys())
    model_lgb = train_lightgbm(train, val, features, target)
    joblib.dump(model_lgb, os.path.join(out_dir, "lightgbm_model.pkl"))

    lgb_preds = model_lgb.predict(val[features])
    lgb_metrics = metrics_df(val[target], lgb_preds)

    # --- Forecast future until 2030-12 ---
    last_date = df.index.max()
    forecast_index = pd.date_range(last_date + pd.offsets.MonthBegin(), "2030-12-01", freq="MS")

    future_preds = iterative_lightgbm_forecast(model_lgb, df, features, target, forecast_index)

    forecast_df = pd.DataFrame({
        "forecast_date": forecast_index,
        "forecast_tonnes": future_preds
    })
    forecast_df.to_csv(os.path.join(out_dir, "forecast_to_2030.csv"), index=False)

    # --- Save Metrics ---
    metrics_all = pd.DataFrame({
        "Model": ["Persistence", "SARIMAX", "LightGBM"],
        "MAE": [pers_metrics["MAE"], sarimax_metrics["MAE"], lgb_metrics["MAE"]],
        "RMSE": [pers_metrics["RMSE"], sarimax_metrics["RMSE"], lgb_metrics["RMSE"]],
        "MAPE(%)": [pers_metrics["MAPE(%)"], sarimax_metrics["MAPE(%)"], lgb_metrics["MAPE(%)"]],
    })
    metrics_all.to_csv(os.path.join(out_dir, "model_metrics.csv"), index=False)

    print("\n✅ Forecasting complete!")
    print("Results saved in:", out_dir)
    print(metrics_all)
    print("\nForecast CSV → outputs/forecast_to_2030.csv")

if __name__ == "__main__":
    run_pipeline()
