# %%
# Create app_layerx_hybrid_v3_updated.py according to the user's requirements.
# The script renders two pages:
# Page 1: Dynamic Horizon Forecast (52/26/12/8 weeks) with KPIs per horizon and Plotly chart using prophet_forecast.csv + actuals
# Page 2: 52-week Stacking Model (Prophet, XGBoost, LightGBM, RandomForest) vs Actual with KPIs equal to Page 1 (52-week metrics)
# Styling/theme and PDF export are inspired by v4.2 provided by the user.


code = r'''# app_layerx_hybrid_v3_updated.py
# Layer-X Hybrid Dashboard (CPF Edition) — v3_updated (2025-10)
# Page 1: Dynamic Horizon Forecast (52/26/12/8 weeks)
# Page 2: Stacking Model Performance (52-week) — Prophet + XGBoost + LightGBM + RandomForest
# Notes:
# - Uses prophet_forecast.csv for predicted values (yhat / yhat_original)
# - Uses "Predict Egg Price 2022-25 with Date - Test_Pmhoo.csv" for actuals (resampled weekly)
# - Stacking is an average of available model predictions among: Prophet, XGBoost, LightGBM, RandomForest
# - KPIs on Page 2 are identical to Page 1 (52-week horizon), per user's request

import os, io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- optional joblib ----------
try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Layer X — Egg Price Forecast (v3 updated)", page_icon="🥚", layout="wide")

# ---------- helpers ----------
def _exists(p: str) -> bool:
    try:
        return p is not None and os.path.exists(p)
    except Exception:
        return False

def _asset(path: str, fallback: str | None = None) -> str | None:
    if _exists(path): return path
    return fallback if _exists(fallback or "") else None

# ---------- assets ----------
import base64

BG_IMAGE   = _asset("assets/space_bg.png", "/mnt/data/7cc2db54-4b0f-4179-9fd0-4e0411da902c.png")
CPF_LOGO   = _asset("assets/LOGO-CPF.jpg", "/mnt/data/LOGO-CPF.jpg")

# Encode egg_rocket.png as base64 for universal rendering
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
egg_paths = [
    os.path.join(SCRIPT_DIR, "assets", "egg_rocket.png"),
    os.path.join(SCRIPT_DIR, "egg_rocket.png"),
    "/mnt/data/egg_rocket.png"
]

EGG_ROCKET = None
for p in egg_paths:
    if os.path.exists(p):
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            EGG_ROCKET = f"data:image/png;base64,{encoded}"
        break

if not EGG_ROCKET:
    st.sidebar.warning("🚫 ไม่พบไฟล์ egg_rocket.png ใน assets หรือ /mnt/data/")

# ---------- theme ----------
def inject_layerx_css():
    bg_layer = f"url('{BG_IMAGE}')" if BG_IMAGE else "none"
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(1200px 600px at 20% 0%, rgba(0,216,255,0.10), rgba(0,0,0,0) 70%),
                linear-gradient(180deg, #061b27 0%, #022a44 55%, #032335 100%);
            color: #E3F6FF;
        }}
        .space-bg {{
            position: fixed; inset: 0; z-index: -2;
            background-image: {bg_layer};
            background-size: cover; background-position: center;
            opacity: .22; filter: saturate(120%);
        }}
        .egg-rocket {{
            position: fixed;
            top: 90px;
            right: 60px;
            z-index: 999;
            width: 140px; max-width: 22vw;
            transform: rotate(-8deg);
            filter: drop-shadow(0 0 16px rgba(255,140,0,0.85));
            pointer-events: none;
            animation: floatRocket 4s ease-in-out infinite;
        }}

        @keyframes floatRocket {{
            0%, 100% {{ transform: translateY(0) rotate(-8deg); }}
            50% {{ transform: translateY(-8px) rotate(-8deg); }}
        }}

        .lx-title {{ font-size: 32px; font-weight: 800; color: #ccf3ff; margin: 2px 0 0; }}
        .lx-sub   {{ font-size: 13px; color: #8fd9ff; opacity: .95; }}
        .kpi {{
            background: linear-gradient(145deg, rgba(0,80,120,.70), rgba(0,20,40,.70));
            border: 1px solid rgba(0,216,255,.40); border-radius: 16px;
            padding: 16px 20px; text-align: center;
            box-shadow: 0 0 16px rgba(0,216,255,.22), inset 0 0 10px rgba(0,216,255,.12);
        }}
        .kpi .label {{ font-size: 13px; color: #a9dcff; opacity: .95; margin: 0 0 4px; }}
        .kpi .value {{ font-size: 28px; color: #00d4ff; font-weight: 800; margin: 0; }}
        .lx-btn > button {{
            border-radius: 12px !important; border: 1px solid rgba(0,216,255,.65) !important;
            background: #00bfff !important; color: #fff !important; font-weight: 700 !important;
            box-shadow: 0 10px 26px rgba(0,216,255,.18) !important;
        }}
        .lx-btn > button:hover {{ filter: brightness(1.08); transform: translateY(-1px); }}
        </style>
        <div class="space-bg"></div>
        """, unsafe_allow_html=True
    )

def kpi_card(label: str, value: str, unit: str = ""):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">{label}</div>
          <div class="value">{value}{unit}</div>
        </div>
        """, unsafe_allow_html=True
    )

# ---------- artifacts ----------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    # Load base models + scaler/feature names when available
    prophet_model = xgb_model = lgbm_model = rf_model = scaler = None
    feature_names = None
    if joblib:
        try: prophet_model = joblib.load(_asset("prophet_model.pkl", "/mnt/data/prophet_model.pkl"))
        except Exception as e: st.sidebar.warning(f"Prophet model not loaded: {e}")
        try: xgb_model = joblib.load(_asset("xgboost_model.pkl", "/mnt/data/xgboost_model.pkl"))
        except Exception as e: st.sidebar.warning(f"XGBoost model not loaded: {e}")
        try: lgbm_model = joblib.load(_asset("lightgbm_model.pkl", "/mnt/data/lightgbm_model.pkl"))
        except Exception as e: st.sidebar.warning(f"LightGBM model not loaded: {e}")
        try: rf_model = joblib.load(_asset("randomforest_model.pkl", "/mnt/data/randomforest_model.pkl"))
        except Exception as e: st.sidebar.warning(f"RandomForest model not loaded: {e}")
        try:
            d = joblib.load(_asset("scaler_and_features.pkl", "/mnt/data/scaler_and_features.pkl"))
            if isinstance(d, dict):
                scaler, feature_names = d.get("scaler"), d.get("feature_names")
            elif isinstance(d, (list, tuple)) and len(d) >= 2:
                scaler, feature_names = d[0], d[1]
        except Exception as e: st.sidebar.warning(f"Scaler/feature_names not loaded: {e}")
    return prophet_model, xgb_model, lgbm_model, rf_model, scaler, feature_names

PROPHET, XGB, LGBM, RF, SCALER, FEAT_NAMES = load_artifacts()

# ---------- PDF export ----------
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader

def export_pdf_report(title_text: str, metrics: list[tuple[str, str]], fig=None) -> io.BytesIO:
    buf = io.BytesIO()
    try: pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    except Exception: pass
    c = canvas.Canvas(buf, pagesize=landscape(A4))
    w, h = landscape(A4)
    c.setFillColorRGB(0.03,0.08,0.13); c.rect(0,0,w,h,fill=True,stroke=False)
    try:
        if CPF_LOGO: c.drawImage(ImageReader(CPF_LOGO), 40, h-100, width=80, preserveAspectRatio=True, mask='auto')
    except Exception: pass
    c.setFillColorRGB(0.0,0.84,1.0)
    try: c.setFont("HYSMyeongJo-Medium", 20)
    except Exception: c.setFont("Helvetica-Bold", 20)
    c.drawString(140, h-70, title_text)
    c.setFillColorRGB(0.70,0.90,1.0)
    try: c.setFont("HYSMyeongJo-Medium", 12)
    except Exception: c.setFont("Helvetica", 12)
    c.drawString(140, h-92, f"วันที่ออกรายงาน: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y = h-150
    for k,v in metrics:
        c.setFillColorRGB(0.52,0.92,1.0); c.drawString(100, y, f"{k}: {v}"); y -= 22
    if fig is not None:
        try:
            img = io.BytesIO(fig.to_image(format="png"))
            c.drawImage(ImageReader(img), 80, 110, width=700, height=360, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            c.setFillColorRGB(1,0.6,0.6); c.drawString(100, 120, f"[WARN] ฝังกราฟไม่ได้: {e}")
    try:
        if EGG_ROCKET: c.drawImage(ImageReader(EGG_ROCKET), w-180, h-180, width=120, preserveAspectRatio=True, mask='auto')
    except Exception: pass
    c.setFillColorRGB(0.45,0.85,1.0)
    try: c.setFont("HYSMyeongJo-Medium", 10)
    except Exception: c.setFont("Helvetica", 10)
    c.drawRightString(w-30, 24, "Layer-X Confidential — © 2025")
    c.showPage(); c.save(); buf.seek(0); return buf

# ---------- metrics per horizon (provided by user) ----------
HORIZON_METRICS = {
    52: {"MAE": 0.1042, "MSE": 0.0149, "RMSE": 0.1222, "R2": 0.7494},
    26: {"MAE": 0.0883, "MSE": 0.0126, "RMSE": 0.1123, "R2": 0.6568},
    12: {"MAE": 0.0886, "MSE": 0.0151, "RMSE": 0.1230, "R2": -3.9507},
     8: {"MAE": 0.1044, "MSE": 0.0194, "RMSE": 0.1394, "R2": 0.0000},
}

# ---------- data loaders ----------
def load_prophet_forecast() -> pd.DataFrame | None:
    pf_path = _asset("prophet_forecast.csv", "/mnt/data/prophet_forecast.csv")
    if not pf_path:
        st.error("ไม่พบไฟล์ prophet_forecast.csv"); return None
    df = pd.read_csv(pf_path)
    if "ds" not in df.columns:
        st.error("prophet_forecast.csv ต้องมีคอลัมน์ 'ds'"); return None
    df["ds"] = pd.to_datetime(df["ds"])
    # choose yhat column
    if "yhat_original" in df.columns:
        df["Prophet"] = df["yhat_original"].astype(float)
    else:
        # assume log if values small
        yh = df.get("yhat")
        if yh is not None:
            arr = pd.to_numeric(yh, errors="coerce").values
            df["Prophet"] = np.expm1(arr) if np.nanmax(arr) < 10 else arr
        else:
            st.error("ไม่พบคอลัมน์ yhat/yhat_original ใน prophet_forecast.csv")
            return None
    return df

def load_actual_df() -> pd.DataFrame | None:
    actual_csv = _asset("Predict Egg Price 2022-25 with Date - Test_Pmhoo.csv",
                        "/mnt/data/Predict Egg Price 2022-25 with Date - Test_Pmhoo.csv")
    if not actual_csv: 
        st.sidebar.warning("ไม่พบไฟล์ Actual (Test_Pmhoo.csv) จะแสดงเฉพาะเส้น Predicted")
        return None
    try:
        a = pd.read_csv(actual_csv)
        date_col = "Date" if "Date" in a.columns else ("date" if "date" in a.columns else None)
        if not date_col or "PriceMarket" not in a.columns:
            st.sidebar.warning("ไฟล์ Actual ไม่มีคอลัมน์ Date/PriceMarket ครบถ้วน"); return None
        a[date_col] = pd.to_datetime(a[date_col], dayfirst=True, errors="coerce")
        a = a.sort_values(date_col).set_index(date_col).resample("W").mean().rename_axis("ds").reset_index()
        return a[["ds","PriceMarket"]].rename(columns={"PriceMarket":"Actual"})
    except Exception as e:
        st.sidebar.warning(f"อ่านไฟล์ Actual ไม่สำเร็จ: {e}"); return None

def merge_forecast_actual(pf: pd.DataFrame, actual: pd.DataFrame | None) -> pd.DataFrame:
    if actual is None: 
        out = pf.copy()
    else:
        out = pd.merge(pf, actual, on="ds", how="left")
    return out.sort_values("ds").reset_index(drop=True)

def subset_horizon(df: pd.DataFrame, weeks: int) -> pd.DataFrame:
    # Use the most recent 'weeks' rows where Prophet is not NaN
    df_valid = df[df["Prophet"].notna()].copy()
    # Keep alignment for Actual if available
    return df_valid.tail(weeks).reset_index(drop=True)

# ---------- model predictions for stacking ----------
def predict_with_model(model, scaler, feat_names: list[str], df: pd.DataFrame, label: str) -> pd.Series | None:
    try:
        if model is None or scaler is None or not isinstance(feat_names, (list, tuple)):
            return None
        # ensure all features exist
        feats = [c for c in feat_names if c in df.columns]
        if len(feats) != len(feat_names):
            return None
        X = df[feat_names].copy()
        try:
            X_scaled = scaler.transform(X)
        except Exception:
            X_scaled = X.values
        y_pred = model.predict(X_scaled)
        # heuristics: if small numeric range, assume log and expm1 back
        yhat = np.expm1(y_pred) if np.nanmax(y_pred) < 10 else y_pred
        return pd.Series(yhat, index=df.index, name=label)
    except Exception:
        return None

# ---------- pages ----------
def render_header(title: str, subtitle: str):
    cols = st.columns([0.12, 0.64, 0.24])
    with cols[0]:
        if CPF_LOGO: st.image(CPF_LOGO, width=72)
    with cols[1]:
        st.markdown(f'<div class="lx-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="lx-sub">{subtitle}</div>', unsafe_allow_html=True)
    if EGG_ROCKET: st.markdown(f"<img class='egg-rocket' src='{EGG_ROCKET}'/>", unsafe_allow_html=True)

def render_kpis(metrics: dict):
    c1,c2,c3,c4 = st.columns(4, gap="large")
    with c1: kpi_card("MAE", f"{metrics['MAE']:.4f}")
    with c2: kpi_card("MSE", f"{metrics['MSE']:.4f}")
    with c3: kpi_card("RMSE", f"{metrics['RMSE']:.4f}")
    with c4: kpi_card("R²", f"{metrics['R2']:.4f}")

def page1_forecast():
    render_header("📈 Forecast Dashboard", "เลือกช่วงเวลาเพื่อดูกราฟ Actual vs Predicted")
    # Controls
    horizon = st.radio("เลือก Horizon (สัปดาห์)", [52,26,12,8], index=0, horizontal=True)
    metrics = HORIZON_METRICS.get(horizon, HORIZON_METRICS[52])
    render_kpis(metrics)

    pf = load_prophet_forecast()
    actual = load_actual_df()
    if pf is None: return
    df = merge_forecast_actual(pf, actual)
    df_h = subset_horizon(df, horizon)

    # Plot
    fig = go.Figure()
    if "Actual" in df_h.columns and df_h["Actual"].notna().any():
        fig.add_trace(go.Scatter(x=df_h["ds"], y=df_h["Actual"], mode="lines",
                                 name="Actual", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df_h["ds"], y=df_h["Prophet"], mode="lines",
                             name=f"Predicted ({horizon}-week)", line=dict(width=3)))
    fig.update_layout(title=f"Actual vs Predicted PriceMarket ({horizon}-week Horizon)",
                      template="plotly_dark", height=500,
                      margin=dict(l=20,r=20,t=60,b=50),
                      xaxis_title="Date", yaxis_title="Price (฿/ฟอง)")
    st.plotly_chart(fig, use_container_width=True)

    # PDF Export
    st.markdown('<div class="lx-btn" style="text-align:center;">', unsafe_allow_html=True)
    if st.button("🧾 Export PDF Report (Page 1)"):
        mlist = [("MAE", f"{metrics['MAE']:.4f}"),
                 ("MSE", f"{metrics['MSE']:.4f}"),
                 ("RMSE", f"{metrics['RMSE']:.4f}"),
                 ("R²", f"{metrics['R2']:.4f}")]
        pdf = export_pdf_report(f"Forecast Dashboard — {horizon}-Week", mlist, fig)
        st.download_button("📥 ดาวน์โหลด PDF", data=pdf, file_name=f"LayerX_Forecast_{horizon}w.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

def page2_stacking():
    render_header("🧠 Stacking Model Performance — Prophet, XGBoost, LightGBM, RandomForest",
                  "เปรียบเทียบ Actual vs Stacking Prediction (52-week)")
    metrics = HORIZON_METRICS[52]
    render_kpis(metrics)

    pf = load_prophet_forecast()
    actual = load_actual_df()
    if pf is None: return
    df = merge_forecast_actual(pf, actual)

    # base: Prophet prediction already in "Prophet"
    preds = []
    if "Prophet" in df.columns:
        preds.append(df["Prophet"].rename("Prophet"))

    # try to compute XGB/LGBM/RF predictions if features are present
    unavailable = []
    if XGB is not None and SCALER is not None and isinstance(FEAT_NAMES, (list, tuple)):
        s = predict_with_model(XGB, SCALER, FEAT_NAMES, df, "XGBoost")
        if s is not None: preds.append(s)
        else: unavailable.append("XGBoost")
    else:
        unavailable.append("XGBoost")

    if LGBM is not None and SCALER is not None and isinstance(FEAT_NAMES, (list, tuple)):
        s = predict_with_model(LGBM, SCALER, FEAT_NAMES, df, "LightGBM")
        if s is not None: preds.append(s)
        else: unavailable.append("LightGBM")
    else:
        unavailable.append("LightGBM")

    if RF is not None and SCALER is not None and isinstance(FEAT_NAMES, (list, tuple)):
        s = predict_with_model(RF, SCALER, FEAT_NAMES, df, "RandomForest")
        if s is not None: preds.append(s)
        else: unavailable.append("RandomForest")
    else:
        unavailable.append("RandomForest")

    if len(preds) == 0:
        st.warning("ไม่สามารถคำนวณ Stacking ได้ — จะแสดงเฉพาะเส้น Prophet")
        df["Stacking"] = df["Prophet"]
    else:
        pred_df = pd.concat(preds, axis=1)
        df["Stacking"] = pred_df.mean(axis=1)

    # subset to 52-week
    df52 = subset_horizon(df, 52)

    # Plot (Actual vs Stacking)
    fig = go.Figure()
    if "Actual" in df52.columns and df52["Actual"].notna().any():
        fig.add_trace(go.Scatter(x=df52["ds"], y=df52["Actual"], mode="lines",
                                 name="Actual (52w)", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df52["ds"], y=df52["Stacking"], mode="lines",
                             name="Predicted (Stacking 52w)", line=dict(width=3)))
    fig.update_layout(title="Actual vs Predicted (Stacking 52-week)",
                      template="plotly_dark", height=500,
                      margin=dict(l=20,r=20,t=60,b=50),
                      xaxis_title="Date", yaxis_title="Price (฿/ฟอง)")
    st.plotly_chart(fig, use_container_width=True)

    if unavailable:
        st.info("ℹ️ โมเดลที่ยังใช้งานไม่ได้ใน Stacking: " + ", ".join(sorted(set(unavailable))))

    st.markdown('<div class="lx-btn" style="text-align:center;">', unsafe_allow_html=True)
    if st.button("🧾 Export PDF Summary (Page 2)"):
        mlist = [("MAE", f"{metrics['MAE']:.4f}"),
                 ("MSE", f"{metrics['MSE']:.4f}"),
                 ("RMSE", f"{metrics['RMSE']:.4f}"),
                 ("R²", f"{metrics['R2']:.4f}")]
        pdf = export_pdf_report("Stacking Model Performance — 52-week", mlist, fig)
        st.download_button("📥 ดาวน์โหลด PDF", data=pdf, file_name="LayerX_Stacking_52w.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- main ----------
inject_layerx_css()
st.sidebar.title("Navigation")
page = st.sidebar.radio("เลือกหน้า", ["Page 1 — Forecast", "Page 2 — Stacking (52w)"], index=0)
if page.startswith("Page 1"):
    page1_forecast()
else:
    page2_stacking()
'''
with open('/mnt/data/app_layerx_hybrid_v3_updated.py', 'w', encoding='utf-8') as f:
    f.write(code)

'/mnt/data/app_layerx_hybrid_v3_updated.py'
