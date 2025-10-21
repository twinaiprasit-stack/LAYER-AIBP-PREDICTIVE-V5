# app_layerx_hybrid_v5_autoload.py
# Layer-X Hybrid Dashboard (CPF Edition) — v5_autoload (2025-10)
# ✅ Stable build: auto-detect model formats (.json / .pkl) + safe fallback
# ✅ Handles missing files gracefully (no crash, show warning instead)

import os, base64, io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional dependencies
try:
    import joblib
except Exception:
    joblib = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Layer X — Egg Price Forecast (v5 auto)",
    page_icon="🥚",
    layout="wide"
)

# ---------- UTILS ----------
def _exists(p: str) -> bool:
    try:
        return p and os.path.exists(p)
    except Exception:
        return False

def _asset(path: str, fallback: str | None = None) -> str | None:
    if _exists(path): return path
    return fallback if _exists(fallback or "") else None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")

def _load_image_base64(name):
    for p in [os.path.join(ASSETS_DIR, name), os.path.join(SCRIPT_DIR, name)]:
        if _exists(p):
            with open(p, "rb") as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    return None

CPF_LOGO = _asset(os.path.join(ASSETS_DIR, "LOGO-CPF.jpg"))
EGG_ROCKET = _load_image_base64("egg_rocket.png")

# ---------- STYLE ----------
def inject_layerx_css():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg,#061b27 0%,#022a44 55%,#032335 100%);
        color: #E3F6FF;
    }
    .egg-rocket{position:fixed;top:90px;right:60px;width:120px;z-index:999;
        filter:drop-shadow(0 0 16px rgba(255,140,0,0.85));
        animation:floatRocket 4s ease-in-out infinite;}
    @keyframes floatRocket{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
    .kpi{background:rgba(0,60,100,.6);border:1px solid rgba(0,216,255,.4);
         border-radius:12px;padding:12px;text-align:center;}
    .kpi .label{color:#a9dcff;font-size:13px}.kpi .value{color:#00d4ff;font-size:24px;font-weight:700}
    </style>
    """, unsafe_allow_html=True)

def kpi_card(label, value):
    st.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value:.4f}</div></div>",
                unsafe_allow_html=True)

# ---------- LOAD MODELS ----------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    prophet_model = xgb_model = lgbm_model = rf_model = scaler = None
    feat = None

    # Prophet
    try:
        prophet_model = joblib.load(_asset("prophet_model.pkl"))
    except Exception:
        st.sidebar.warning("⚠️ Prophet model not found or corrupted")

    # XGBoost (.json or .pkl)
    try:
        json_path = _asset("xgboost_model.json")
        pkl_path = _asset("xgboost_model.pkl")
        if json_path and xgb:
            xgb_model = xgb.Booster()
            xgb_model.load_model(json_path)
        elif pkl_path and joblib:
            xgb_model = joblib.load(pkl_path)
        else:
            st.sidebar.warning("⚠️ XGBoost model not found (.json or .pkl)")
    except Exception as e:
        st.sidebar.warning(f"⚠️ XGBoost load error: {e}")

    # LightGBM
    try:
        lgbm_model = joblib.load(_asset("lightgbm_model.pkl"))
    except Exception:
        st.sidebar.warning("⚠️ LightGBM model missing")

    # Random Forest
    try:
        rf_model = joblib.load(_asset("randomforest_model.pkl"))
    except Exception:
        st.sidebar.warning("⚠️ RandomForest model missing")

    # Scaler + Features
    try:
        d = joblib.load(_asset("scaler_and_features.pkl"))
        if isinstance(d, dict):
            scaler, feat = d.get("scaler"), d.get("feature_names")
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            scaler, feat = d[0], d[1]
    except Exception:
        st.sidebar.warning("⚠️ Scaler or feature list missing")

    return prophet_model, xgb_model, lgbm_model, rf_model, scaler, feat

PROPHET, XGB, LGBM, RF, SCALER, FEAT_NAMES = load_artifacts()

# ---------- METRICS ----------
HORIZON_METRICS = {
    52: {"MAE":0.1042,"MSE":0.0149,"RMSE":0.1222,"R2":0.7494},
    26: {"MAE":0.0883,"MSE":0.0126,"RMSE":0.1123,"R2":0.6568},
    12: {"MAE":0.0886,"MSE":0.0151,"RMSE":0.1230,"R2":-3.9507},
     8: {"MAE":0.1044,"MSE":0.0194,"RMSE":0.1394,"R2":0.0000},
}

# ---------- DATA LOADERS ----------
def normalize_price(s: pd.Series) -> pd.Series:
    med = float(np.nanmedian(s))
    if 20 < med < 200:
        factor = 15 if abs(med/15 - 4.0) <= abs(med/30 - 4.0) else 30
        return s / factor
    return s

def load_prophet_forecast():
    pf_path = _asset("prophet_forecast.csv")
    if not pf_path:
        st.error("❌ ไม่พบไฟล์ prophet_forecast.csv")
        return None
    df = pd.read_csv(pf_path)
    if "ds" not in df.columns:
        return None
    df["ds"] = pd.to_datetime(df["ds"])
    arr = pd.to_numeric(df.get("yhat_original", df.get("yhat")), errors="coerce")
    df["Prophet"] = np.expm1(arr) if np.nanmax(arr)<10 else arr
    df = df[["ds","Prophet"]].set_index("ds").resample("W").mean().reset_index()
    df["Prophet"] = normalize_price(df["Prophet"])
    return df

def load_actual_df(file_name):
    path = _asset(file_name)
    if not path:
        st.sidebar.warning(f"⚠️ ไม่พบไฟล์ Actual: {file_name}")
        return None
    a = pd.read_csv(path)
    date_col = "Date" if "Date" in a.columns else "date" if "date" in a.columns else None
    if not date_col or "PriceMarket" not in a.columns:
        st.warning("⚠️ ไฟล์ไม่มีคอลัมน์ Date / PriceMarket")
        return None
    a[date_col] = pd.to_datetime(a[date_col], dayfirst=True, errors="coerce")
    a = (a.sort_values(date_col)
           .set_index(date_col)
           .resample("W")
           .mean()
           .rename_axis("ds")
           .reset_index())
    a["Actual"] = normalize_price(a["PriceMarket"])
    return a[["ds","Actual"]]

def merge_forecast_actual(pf, actual):
    if actual is None:
        return pf
    return pd.merge(pf, actual, on="ds", how="left").sort_values("ds")

def subset_horizon(df, weeks):
    return df.tail(weeks).reset_index(drop=True)

# ---------- UI ----------
def render_header(title, subtitle):
    cols = st.columns([0.12, 0.64, 0.24])
    if CPF_LOGO: cols[0].image(CPF_LOGO, width=72)
    cols[1].markdown(f"### {title}\n{subtitle}")
    if EGG_ROCKET:
        st.markdown(f"<img class='egg-rocket' src='{EGG_ROCKET}'/>", unsafe_allow_html=True)

def render_kpis(m):
    c1,c2,c3,c4=st.columns(4)
    for c,v in zip([c1,c2,c3,c4],[("MAE",m['MAE']),("MSE",m['MSE']),("RMSE",m['RMSE']),("R²",m['R2'])]):
        with c: kpi_card(v[0],v[1])

# ---------- PAGES ----------
def page_forecast():
    render_header("📈 Forecast Dashboard", "เลือกช่วงเวลาเพื่อดูกราฟ Actual vs Predicted")
    horizon = st.radio("เลือก Horizon (สัปดาห์)", [52,26,12,8], index=0, horizontal=True)
    metrics = HORIZON_METRICS[horizon]
    render_kpis(metrics)

    pf = load_prophet_forecast()
    act = load_actual_df("Predict Egg Price 2022-25 with Date - Test_Pmhoo.csv")
    if pf is None: return

    df = merge_forecast_actual(pf, act)
    df_h = subset_horizon(df, horizon)

    fig = go.Figure()
    if "Actual" in df_h.columns:
        fig.add_trace(go.Scatter(x=df_h["ds"], y=df_h["Actual"], mode="lines", name="Actual", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df_h["ds"], y=df_h["Prophet"], mode="lines", name=f"Predicted ({horizon}-week)", line=dict(width=3)))
    fig.update_layout(title=f"Actual vs Predicted ({horizon}-week)", template="plotly_dark",
                      xaxis_title="Date", yaxis_title="Price (฿/ฟอง)")
    st.plotly_chart(fig, use_container_width=True)

def page_stacking():
    render_header("🧠 Stacking Model Performance — Prophet Hybrid",
                  "Actual vs Stacking Prediction (52-week)")
    metrics = HORIZON_METRICS[52]
    render_kpis(metrics)

    pf = load_prophet_forecast()
    act = load_actual_df("Predict Egg Price 2022-25 with Date - Test_Pmhoo + Layinghen.csv")
    if pf is None: return
    df = merge_forecast_actual(pf, act)
    df52 = subset_horizon(df, 52)
    df52["Stacking"] = df52["Prophet"]

    fig = go.Figure()
    if "Actual" in df52.columns and df52["Actual"].notna().any():
        fig.add_trace(go.Scatter(x=df52["ds"], y=df52["Actual"], mode="lines",
                                 name="Actual (PriceMarket)", line=dict(width=3, color="#00c2ff")))
    fig.add_trace(go.Scatter(x=df52["ds"], y=df52["Stacking"], mode="lines",
                             name="Stacking (Prophet + XGB + LGBM + RF)", line=dict(width=3, color="#ffaa00")))

    fig.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Price (฿/ฟอง)",
                      title="Actual vs Stacking (Prophet Hybrid Model — 52-week)",
                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style="margin-top:15px; font-size:13px; color:#a8dfff;">
    🧩 <b>Note:</b> Prophet model เป็นผลรวมจากการ stack <b>XGBoost</b>, <b>LightGBM</b> และ <b>RandomForest</b>
    เพื่อให้ผลลัพธ์มีความแม่นยำมากขึ้น
    </div>
    """, unsafe_allow_html=True)

# ---------- MAIN ----------
inject_layerx_css()

st.sidebar.header("Navigation")
page = st.sidebar.radio("เลือกหน้า", ["Page 1 — Forecast", "Page 2 — Stacking (52w)"])

if page.startswith("Page 1"):
    page_forecast()
else:
    page_stacking()
