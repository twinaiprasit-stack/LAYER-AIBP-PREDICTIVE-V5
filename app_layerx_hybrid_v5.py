# ==========================================================
# Layer-X Hybrid Dashboard (v5) — Prophet + XGBoost + LightGBM + RandomForest
# CPF Edition / 2025
# ==========================================================

import os, io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Layer X — Hybrid Forecast v5", page_icon="🥚", layout="wide")

# ---------- helper ----------
def _asset(path, fallback=None):
    if os.path.exists(path): return path
    if fallback and os.path.exists(fallback): return fallback
    return None

# ---------- load artifacts ----------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    models = {}
    try:
        models["prophet"] = joblib.load(_asset("prophet_model.pkl", "/mnt/data/prophet_model.pkl"))
        models["xgb"] = joblib.load(_asset("xgboost_model.pkl", "/mnt/data/xgboost_model.pkl"))
        models["lgbm"] = joblib.load(_asset("lightgbm_model.pkl", "/mnt/data/lightgbm_model.pkl"))
        models["rf"] = joblib.load(_asset("randomforest_model.pkl", "/mnt/data/randomforest_model.pkl"))
        scaler_bundle = joblib.load(_asset("scaler_and_features.pkl", "/mnt/data/scaler_and_features.pkl"))
        if isinstance(scaler_bundle, dict):
            models["scaler"] = scaler_bundle.get("scaler")
            models["features"] = scaler_bundle.get("feature_names")
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
    return models

MODELS = load_artifacts()

# ---------- load datasets ----------
@st.cache_data
def load_data():
    forecast_csv = _asset("prophet_forecast.csv", "/mnt/data/prophet_forecast.csv")
    test_csv = _asset("Predict Egg Price 2022-25 with Date - Test_Pmhoo + Layinghen.csv",
                      "/mnt/data/Predict Egg Price 2022-25 with Date - Test_Pmhoo + Layinghen.csv")
    if forecast_csv is None or test_csv is None:
        st.error("Missing required CSV files."); return None, None
    forecast_df = pd.read_csv(forecast_csv)
    test_df = pd.read_csv(test_csv)
    test_df["Date"] = pd.to_datetime(test_df["Date"], dayfirst=True)
    test_df = test_df.rename(columns={"Date": "ds", "PriceMarket": "Actual"})
    return forecast_df, test_df

FORECAST, TEST = load_data()

# ---------- hybrid compute ----------
def compute_hybrid_predictions():
    if FORECAST is None or TEST is None:
        return None
    df = FORECAST.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.merge(TEST[["ds", "Actual"]], on="ds", how="left")

    feature_names = MODELS.get("features", [])
    scaler = MODELS.get("scaler")
    XGB = MODELS.get("xgb")
    LGBM = MODELS.get("lgbm")
    RF = MODELS.get("rf")

    # Prophet prediction
    df["Prophet"] = df["yhat_original"] if "yhat_original" in df.columns else df["yhat"]

    # Generate base model predictions if possible
    if scaler is not None and feature_names:
        try:
            X = df[feature_names].fillna(method="ffill")
            X_scaled = scaler.transform(X)
            df["XGBoost"] = XGB.predict(X_scaled)
            df["LightGBM"] = LGBM.predict(X_scaled)
            df["RandomForest"] = RF.predict(X_scaled)
        except Exception as e:
            st.warning(f"Unable to compute base model predictions: {e}")

    # Combine hybrid (mean of available)
    available_cols = [c for c in ["Prophet", "XGBoost", "LightGBM", "RandomForest"] if c in df.columns]
    df["Hybrid"] = df[available_cols].mean(axis=1)

    return df

RESULT = compute_hybrid_predictions()

# ---------- evaluation ----------
def evaluate_model(df):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    df_eval = df.dropna(subset=["Hybrid", "Actual"])
    if df_eval.empty: return None
    mae = mean_absolute_error(df_eval["Actual"], df_eval["Hybrid"])
    mse = mean_squared_error(df_eval["Actual"], df_eval["Hybrid"])
    rmse = np.sqrt(mse)
    r2 = r2_score(df_eval["Actual"], df_eval["Hybrid"])
    return dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2)

METRICS = evaluate_model(RESULT) if RESULT is not None else None

# ---------- visualization ----------
st.title("🚀 Layer-X Hybrid Forecast (Prophet + XGB + LGBM + RF)")
st.markdown("CPF Layer Business Intelligence — v5.0")

if RESULT is not None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=RESULT["ds"], y=RESULT["Actual"], mode="lines+markers",
                             name="Actual", line=dict(color="red", width=2)))
    for name, color in zip(["Prophet","XGBoost","LightGBM","RandomForest","Hybrid"],
                           ["#00E0FF","#FFD166","#8DFF6A","#FFA3E0","#00FFAA"]):
        if name in RESULT.columns:
            fig.add_trace(go.Scatter(x=RESULT["ds"], y=RESULT[name], mode="lines",
                                     name=name, line=dict(color=color, width=2)))
    fig.update_layout(template="plotly_dark", height=520,
                      title="Actual vs Hybrid Forecast (Stacked Models)",
                      xaxis_title="Date", yaxis_title="Egg Price (฿/ฟอง)")
    st.plotly_chart(fig, use_container_width=True)

    if METRICS:
        st.subheader("📊 Evaluation Metrics")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("MAE", f"{METRICS['MAE']:.4f}")
        c2.metric("RMSE", f"{METRICS['RMSE']:.4f}")
        c3.metric("MSE", f"{METRICS['MSE']:.4f}")
        c4.metric("R² Score", f"{METRICS['R2']*100:.2f}%")

        csv = RESULT[["ds","Actual","Hybrid"]].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast Result", data=csv,
                           file_name="hybrid_v5_result.csv", mime="text/csv")
else:
    st.warning("No data available for display. Check CSV or model files.")
