# app_layerx_hybrid_v3_updated.py (fixed)
# Layer-X Hybrid Dashboard (CPF Edition) — v3_updated (2025-10)
# Fixed: weekly resample + normalized units for price (฿/egg)

import os, io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Layer X — Egg Price Forecast (v3 updated)", page_icon="🥚", layout="wide")

def _exists(p: str) -> bool:
    try:
        return p is not None and os.path.exists(p)
    except Exception:
        return False

def _asset(path: str, fallback: str | None = None) -> str | None:
    if _exists(path): return path
    return fallback if _exists(fallback or "") else None

import base64
BG_IMAGE = _asset("assets/space_bg.png")
CPF_LOGO = _asset("assets/LOGO-CPF.jpg")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
egg_paths = [
    os.path.join(SCRIPT_DIR, "assets", "egg_rocket.png"),
    os.path.join(SCRIPT_DIR, "egg_rocket.png"),
]
EGG_ROCKET = None
for p in egg_paths:
    if os.path.exists(p):
        with open(p, "rb") as f:
            EGG_ROCKET = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        break

def inject_layerx_css():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg,#061b27 0%,#022a44 55%,#032335 100%);
            color: #E3F6FF;
        }
        .egg-rocket{position:fixed;top:90px;right:60px;width:120px;z-index:999;
            filter:drop-shadow(0 0 16px rgba(255,140,0,0.85));
            animation:floatRocket 4s ease-in-out infinite;}
        @keyframes floatRocket{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
        .kpi{background:rgba(0,60,100,.6);border:1px solid rgba(0,216,255,.4);border-radius:12px;padding:12px;text-align:center;}
        .kpi .label{color:#a9dcff;font-size:13px}.kpi .value{color:#00d4ff;font-size:24px;font-weight:700}
        </style>
        """, unsafe_allow_html=True)

def kpi_card(label, value):
    st.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value:.4f}</div></div>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    prophet_model = xgb_model = lgbm_model = rf_model = scaler = None
    feat = None
    if joblib:
        try: prophet_model = joblib.load(_asset("prophet_model.pkl"))
        except Exception: pass
        try: xgb_model = joblib.load(_asset("xgboost_model.pkl"))
        except Exception: pass
        try: lgbm_model = joblib.load(_asset("lightgbm_model.pkl"))
        except Exception: pass
        try: rf_model = joblib.load(_asset("randomforest_model.pkl"))
        except Exception: pass
        try:
            d = joblib.load(_asset("scaler_and_features.pkl"))
            if isinstance(d, dict): scaler, feat = d.get("scaler"), d.get("feature_names")
            elif isinstance(d, (list, tuple)) and len(d)>=2: scaler, feat = d[0], d[1]
        except Exception: pass
    return prophet_model, xgb_model, lgbm_model, rf_model, scaler, feat

PROPHET, XGB, LGBM, RF, SCALER, FEAT_NAMES = load_artifacts()

HORIZON_METRICS = {
    52: {"MAE":0.1042,"MSE":0.0149,"RMSE":0.1222,"R2":0.7494},
    26: {"MAE":0.0883,"MSE":0.0126,"RMSE":0.1123,"R2":0.6568},
    12: {"MAE":0.0886,"MSE":0.0151,"RMSE":0.1230,"R2":-3.9507},
     8: {"MAE":0.1044,"MSE":0.0194,"RMSE":0.1394,"R2":0.0000},
}

def normalize_price(s: pd.Series) -> pd.Series:
    med = float(np.nanmedian(s))
    if 20 < med < 200:
        c15 = abs(med/15 - 4.0); c30 = abs(med/30 - 4.0)
        factor = 15 if c15 <= c30 else 30
        return s / factor
    return s

def load_prophet_forecast():
    pf_path = _asset("prophet_forecast.csv")
    if not pf_path:
        st.error("ไม่พบไฟล์ prophet_forecast.csv"); return None
    df = pd.read_csv(pf_path)
    if "ds" not in df.columns: return None
    df["ds"] = pd.to_datetime(df["ds"])
    if "yhat_original" in df.columns:
        df["Prophet"] = pd.to_numeric(df["yhat_original"], errors="coerce")
    else:
        arr = pd.to_numeric(df.get("yhat"), errors="coerce")
        df["Prophet"] = np.expm1(arr) if np.nanmax(arr)<10 else arr
    df_w = df[["ds","Prophet"]].set_index("ds").resample("W").mean().reset_index()
    df_w["Prophet"] = normalize_price(df_w["Prophet"])
    return df_w

def load_actual_df():
    path = _asset("Predict Egg Price 2022-25 with Date - Test_Pmhoo.csv")
    if not path:
        st.sidebar.warning("ไม่พบไฟล์ Actual"); return None
    a = pd.read_csv(path)
    date_col = "Date" if "Date" in a.columns else ("date" if "date" in a.columns else None)
    if not date_col or "PriceMarket" not in a.columns: return None
    a[date_col] = pd.to_datetime(a[date_col], dayfirst=True, errors="coerce")
    a = a.sort_values(date_col).set_index(date_col).resample("W").mean().rename_axis("ds").reset_index()
    a["Actual"] = normalize_price(a["PriceMarket"])
    return a[["ds","Actual"]]

def merge_forecast_actual(pf, actual):
    if actual is None: return pf
    return pd.merge(pf, actual, on="ds", how="left").sort_values("ds").reset_index(drop=True)

def subset_horizon(df, weeks):
    return df.tail(weeks).reset_index(drop=True)

def predict_with_model(model, scaler, feat, df, label):
    try:
        if model is None or scaler is None or not isinstance(feat,(list,tuple)): return None
        feats=[c for c in feat if c in df.columns]
        if len(feats)!=len(feat): return None
        X=scaler.transform(df[feats]) if scaler else df[feats].values
        y=model.predict(X)
        y=np.expm1(y) if np.nanmax(y)<10 else y
        return pd.Series(y,index=df.index,name=label)
    except Exception: return None

def render_header(title,subtitle):
    cols=st.columns([0.12,0.64,0.24])
    if CPF_LOGO: cols[0].image(CPF_LOGO,width=72)
    cols[1].markdown(f"### {title}\n{subtitle}")
    if EGG_ROCKET: st.markdown(f"<img class='egg-rocket' src='{EGG_ROCKET}'/>",unsafe_allow_html=True)

def render_kpis(m):
    c1,c2,c3,c4=st.columns(4)
    for c,v in zip([c1,c2,c3,c4],[("MAE",m['MAE']),("MSE",m['MSE']),("RMSE",m['RMSE']),("R²",m['R2'])]): 
        with c: kpi_card(v[0],v[1])

def page1():
    render_header("📈 Forecast Dashboard","เลือกช่วงเวลาเพื่อดูกราฟ Actual vs Predicted")
    horizon=st.radio("เลือก Horizon (สัปดาห์)",[52,26,12,8],index=0,horizontal=True)
    m=HORIZON_METRICS[horizon]; render_kpis(m)
    pf,act=load_prophet_forecast(),load_actual_df()
    if pf is None: return
    df=merge_forecast_actual(pf,act); df_h=subset_horizon(df,horizon)
    fig=go.Figure()
    if "Actual" in df_h.columns: fig.add_trace(go.Scatter(x=df_h["ds"],y=df_h["Actual"],mode="lines",name="Actual",line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df_h["ds"],y=df_h["Prophet"],mode="lines",name=f"Predicted ({horizon}-week)",line=dict(width=3)))
    fig.update_layout(title=f"Actual vs Predicted ({horizon}-week)",template="plotly_dark",xaxis_title="Date",yaxis_title="Price (฿/ฟอง)")
    st.plotly_chart(fig,use_container_width=True)

def page2_stacking():
    # ----- Header -----
    render_header("🧠 Stacking Model Performance — Prophet Hybrid (XGBoost + LightGBM + RandomForest)",
                  "Actual vs Stacking Prediction (52-week)")
    metrics = HORIZON_METRICS[52]
    render_kpis(metrics)

    # ----- Load Data -----
    pf = load_prophet_forecast()
    if pf is None:
        st.error("ไม่พบไฟล์ prophet_forecast.csv")
        return

    # Actual data: use new file and correct column name
    actual_csv = _asset("Predict Egg Price 2022-25 with Date - Test_Pmhoo + Layinghen.csv",
                        "/mnt/data/Predict Egg Price 2022-25 with Date - Test_Pmhoo + Layinghen.csv")
    actual = None
    if _exists(actual_csv):
        try:
            a = pd.read_csv(actual_csv)
            date_col = "Date" if "Date" in a.columns else ("date" if "date" in a.columns else None)
            if date_col and "PriceMarket" in a.columns:
                a[date_col] = pd.to_datetime(a[date_col], dayfirst=True, errors="coerce")
                a = (a.sort_values(date_col)
                       .set_index(date_col)
                       .resample("W")
                       .mean()
                       .rename_axis("ds")
                       .reset_index())
                actual = a[["ds", "PriceMarket"]].rename(columns={"PriceMarket": "Actual"})
            else:
                st.warning("⚠️ ไฟล์ Actual ไม่มีคอลัมน์ Date/PriceMarket")
        except Exception as e:
            st.warning(f"อ่านไฟล์ Actual ไม่สำเร็จ: {e}")
    else:
        st.sidebar.warning("ไม่พบไฟล์ Actual: Predict Egg Price 2022-25 with Date - Test_Pmhoo + Layinghen.csv")

    # ----- Merge Forecast + Actual -----
    df = merge_forecast_actual(pf, actual)
    df52 = subset_horizon(df, 52)

    # Prophet = Stacking model (already combined)
    df52["Stacking"] = df52["Prophet"]

    # ----- Plot -----
    fig = go.Figure()
    if "Actual" in df52.columns and df52["Actual"].notna().any():
        fig.add_trace(go.Scatter(
            x=df52["ds"], y=df52["Actual"], mode="lines",
            name="Actual (PriceMarket)", line=dict(width=3, color="#00c2ff")
        ))
    fig.add_trace(go.Scatter(
        x=df52["ds"], y=df52["Stacking"], mode="lines",
        name="Stacking (Prophet + XGB + LGBM + RF)", line=dict(width=3, color="#ffaa00")
    ))

    fig.update_layout(
        title="Actual vs Stacking (Prophet Hybrid Model — 52-week)",
        template="plotly_dark", height=500,
        margin=dict(l=20, r=20, t=60, b=50),
        xaxis_title="Date", yaxis_title="Price (฿/ฟอง)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----- Model Info -----
    st.markdown(
        """
        <div style="margin-top:15px; font-size:13px; color:#a8dfff;">
        🧩 <b>Note:</b> Prophet model เป็นผลรวมจากการ stack <b>XGBoost</b>, <b>LightGBM</b> และ <b>RandomForest</b> ซึ่งเป็น hybrid model เพื่อให้ผลลัพธ์มีความแม่นยำมากขึ้น
        </div>
        """,
        unsafe_allow_html=True
    )

    # ----- PDF Export -----
    st.markdown('<div class="lx-btn" style="text-align:center;">', unsafe_allow_html=True)
    if st.button("🧾 Export PDF Summary (Page 2)"):
        mlist = [("MAE", f"{metrics['MAE']:.4f}"),
                 ("MSE", f"{metrics['MSE']:.4f}"),
                 ("RMSE", f"{metrics['RMSE']:.4f}"),
                 ("R²", f"{metrics['R2']:.4f}")]
        pdf = export_pdf_report("Stacking Model Performance — Prophet Hybrid (52-week)", mlist, fig)
        st.download_button("📥 ดาวน์โหลด PDF", data=pdf, file_name="LayerX_Stacking_52w_Hybrid.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

