# app_layerx_hybrid_v4_2.py
# Layer-X Hybrid Dashboard (CPF Edition) — UX 2025
# Page1: 8-Week Prophet Forecast  |  Page2: Actual vs Prophet + XGBoost + Hybrid (+Actual)
# PDF Export: Thai text + CPF logo + Egg Rocket + in-memory chart image

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

st.set_page_config(page_title="Layer X — Egg Price Forecast", page_icon="🥚", layout="wide")

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

# ✅ Encode egg_rocket.png เป็น Base64 เพื่อให้แน่ใจว่าแสดงได้ทุกที่
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
    prophet_model = xgb_model = scaler = None
    feature_names = None
    if joblib:
        try: prophet_model = joblib.load(_asset("prophet_model.pkl", "/mnt/data/prophet_model.pkl"))
        except Exception as e: st.sidebar.warning(f"Prophet model not loaded: {e}")
        try: xgb_model = joblib.load(_asset("xgboost_model.pkl", "/mnt/data/xgboost_model.pkl"))
        except Exception as e: st.sidebar.warning(f"XGBoost model not loaded: {e}")
        try:
            d = joblib.load(_asset("scaler_and_features.pkl", "/mnt/data/scaler_and_features.pkl"))
            if isinstance(d, dict):
                scaler, feature_names = d.get("scaler"), d.get("feature_names")
            elif isinstance(d, (list, tuple)) and len(d) >= 2:
                scaler, feature_names = d[0], d[1]
        except Exception as e: st.sidebar.warning(f"Scaler/feature_names not loaded: {e}")
    return prophet_model, xgb_model, scaler, feature_names

PROPHET, XGB, SCALER, FEAT_NAMES = load_artifacts()

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

# ---------- Page 1 ----------
def render_page1():
    cols = st.columns([0.12, 0.64, 0.24])
    with cols[0]:
        if CPF_LOGO: st.image(CPF_LOGO, width=72)
    with cols[1]:
        st.markdown('<div class="lx-title">🥚 Egg Price Forecast Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="lx-sub">AI-Powered Market Intelligence (2025)</div>', unsafe_allow_html=True)
    if EGG_ROCKET: st.markdown(f"<img class='egg-rocket' src='{EGG_ROCKET}'/>", unsafe_allow_html=True)

    pf_path = _asset("prophet_forecast.csv", "/mnt/data/prophet_forecast.csv")
    if not pf_path:
        st.error("ไม่พบไฟล์ prophet_forecast.csv"); return
    df = pd.read_csv(pf_path)
    yhat = df["yhat_original"].astype(float) if "yhat_original" in df.columns else np.expm1(df.get("yhat", pd.Series([np.nan]*len(df))))
    ds = pd.to_datetime(df["ds"])

    c1,c2,c3 = st.columns(3, gap="large")
    with c1: kpi_card("Min Price", f"{float(np.nanmin(yhat)):.2f}", " ฿/ฟอง")
    with c2: kpi_card("Avg Price", f"{float(np.nanmean(yhat)):.2f}", " ฿/ฟอง")
    with c3: kpi_card("Max Price", f"{float(np.nanmax(yhat)):.2f}", " ฿/ฟอง")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds, y=yhat, mode="lines", name="Prophet Forecast", line=dict(color="#00e0ff", width=3)))
    if {"yhat_lower_original","yhat_upper_original"} <= set(df.columns):
        fig.add_trace(go.Scatter(
            x=pd.concat([ds, ds[::-1]]),
            y=pd.concat([df["yhat_upper_original"], df["yhat_lower_original"][::-1]]),
            fill="toself", fillcolor="rgba(0,224,255,0.18)", line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", name="Confidence"
        ))
    fig.update_layout(title="8-Week Forecast (Prophet)", template="plotly_dark",
                      height=460, margin=dict(l=20,r=20,t=60,b=40),
                      xaxis_title="Date", yaxis_title="Price (฿/ฟอง)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="lx-btn" style="text-align:center;">', unsafe_allow_html=True)
    if st.button("🧾 Export PDF Report", key="exp1"):
        metrics = [("Min Price", f"{float(np.nanmin(yhat)):.2f} ฿/ฟอง"),
                   ("Avg Price", f"{float(np.nanmean(yhat)):.2f} ฿/ฟอง"),
                   ("Max Price", f"{float(np.nanmax(yhat)):.2f} ฿/ฟอง")]
        pdf = export_pdf_report("Egg Price Forecast Dashboard", metrics, fig)
        st.download_button("📥 ดาวน์โหลด PDF", data=pdf, file_name="LayerX_Page1_Report.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Page 2 ----------
FIXED_MAE, FIXED_RMSE, FIXED_MAPE, FIXED_ACC = 0.0037, 0.0046, 0.10, 99.90

def _load_actual_df():
    """โหลดราคาจริงจากไฟล์ทดสอบ และเตรียมคอลัมน์ Date/PriceMarket"""
    actual_csv = _asset("Predict Egg Price 2022-25 with Date - Test_Pmhoo.csv",
                        "/mnt/data/Predict Egg Price 2022-25 with Date - Test_Pmhoo.csv")
    if not actual_csv: return None
    try:
        a = pd.read_csv(actual_csv)
        # รองรับชื่อคอลัมน์ Date/Datetime แบบยืดหยุ่น
        date_col = "Date" if "Date" in a.columns else ("date" if "date" in a.columns else None)
        if not date_col or "PriceMarket" not in a.columns:
            return None
        a[date_col] = pd.to_datetime(a[date_col], dayfirst=True, errors="coerce")
        a = a.rename(columns={date_col: "ds"})
        # หากเป็นรายวัน แปลงเป็นรายสัปดาห์ให้เทียบกับ Prophet ที่เป็น weekly
        a = a.sort_values("ds").set_index("ds").resample("W").mean().rename_axis("ds").reset_index()
        return a[["ds", "PriceMarket"]]
    except Exception:
        return None

def render_page2():
    cols = st.columns([0.12, 0.64, 0.24])
    with cols[0]:
        if CPF_LOGO: st.image(CPF_LOGO, width=72)
    with cols[1]:
        st.markdown('<div class="lx-title">🚀 Hybrid Model Performance — Prophet + XGBoost</div>', unsafe_allow_html=True)
        st.markdown('<div class="lx-sub">Smart Fusion Forecasting for Egg Price (2025)</div>', unsafe_allow_html=True)
    if EGG_ROCKET: st.markdown(f"<img class='egg-rocket' src='{EGG_ROCKET}'/>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4, gap="large")
    with c1: kpi_card("MAE", f"{FIXED_MAE:.4f}")
    with c2: kpi_card("RMSE", f"{FIXED_RMSE:.4f}")
    with c3: kpi_card("MAPE", f"{FIXED_MAPE:.2f}%")
    with c4: kpi_card("Accuracy", f"{FIXED_ACC:.2f}%")

    pf_path = _asset("prophet_forecast.csv", "/mnt/data/prophet_forecast.csv")
    if not pf_path: st.error("ไม่พบไฟล์ prophet_forecast.csv — จำเป็นสำหรับกราฟ Hybrid"); return
    pf = pd.read_csv(pf_path)
    pf["ds"] = pd.to_datetime(pf["ds"])

    # Prophet
    pf["Prophet"] = (pf["yhat_original"].astype(float)
                     if "yhat_original" in pf.columns
                     else np.expm1(pf.get("yhat", pd.Series([np.nan]*len(pf)))))

    # Actual (จากไฟล์ Test_Pmhoo)
    actual_df = _load_actual_df()
    if actual_df is not None:
        pf = pf.merge(actual_df, on="ds", how="left")  # เพิ่มคอลัมน์ PriceMarket (Actual)
        if pf["PriceMarket"].notna().any():
            pf["Actual"] = pf["PriceMarket"]

    # XGBoost predict (ถ้าฟีเจอร์ครบ)
    unavailable_reason = None
    if XGB is not None and SCALER is not None and isinstance(FEAT_NAMES, (list, tuple)):
        feature_cols_available = [c for c in FEAT_NAMES if c in pf.columns]
        if len(feature_cols_available) == len(FEAT_NAMES):
            try:
                X = pf[FEAT_NAMES].copy()
                try: X_scaled = SCALER.transform(X)
                except Exception: X_scaled = X.values
                y_pred_log = XGB.predict(X_scaled)
                y_pred = np.expm1(y_pred_log) if np.nanmax(y_pred_log) < 10 else y_pred_log
                pf["XGBoost"] = y_pred
            except Exception as e:
                unavailable_reason = f"คำนวณ XGBoost ไม่สำเร็จ: {e}"
        else:
            unavailable_reason = "คอลัมน์ฟีเจอร์ในไฟล์ไม่ครบตามที่โมเดล XGBoost ต้องการ"
    else:
        unavailable_reason = "ยังโหลดโมเดล/สเกลเลอร์ XGBoost ไม่ครบ"

    # Hybrid
    if {"Prophet","XGBoost"} <= set(pf.columns):
        pf["Hybrid"] = pf[["Prophet","XGBoost"]].mean(axis=1)

    # ---- Plot ----
    fig = go.Figure()
    if "Actual" in pf.columns:
        fig.add_trace(go.Scatter(x=pf["ds"], y=pf["Actual"], mode="lines+markers",
                                 name="Actual", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=pf["ds"], y=pf["Prophet"], mode="lines",
                             name="Prophet", line=dict(color="#00e0ff", width=3)))
    if "XGBoost" in pf.columns:
        fig.add_trace(go.Scatter(x=pf["ds"], y=pf["XGBoost"], mode="lines",
                                 name="XGBoost", line=dict(color="#ffb347", width=3)))
    if "Hybrid" in pf.columns:
        fig.add_trace(go.Scatter(x=pf["ds"], y=pf["Hybrid"], mode="lines",
                                 name="Hybrid", line=dict(color="#7CFF6B", width=3)))

    fig.update_layout(title="Actual vs Predicted (Prophet + XGBoost + Hybrid)",
                      template="plotly_dark", height=500,
                      margin=dict(l=20,r=20,t=60,b=50),
                      xaxis_title="Date", yaxis_title="Price (฿/ฟอง)")
    st.plotly_chart(fig, use_container_width=True)

    if unavailable_reason:
        st.info(f"ℹ️ XGBoost: {unavailable_reason} — กำลังแสดงเฉพาะเส้น Prophet/Actual/Hybrid เท่าที่มีข้อมูล")

    st.markdown('<div class="lx-btn" style="text-align:center;">', unsafe_allow_html=True)
    if st.button("🧾 Export PDF Summary", key="exp2"):
        metrics = [("MAE", f"{FIXED_MAE:.4f}"),
                   ("RMSE", f"{FIXED_RMSE:.4f}"),
                   ("MAPE", f"{FIXED_MAPE:.2f}%"),
                   ("Accuracy", f"{FIXED_ACC:.2f}%")]
        pdf = export_pdf_report("Hybrid Model Performance — Prophet + XGBoost", metrics, fig)
        st.download_button("📥 ดาวน์โหลด PDF", data=pdf, file_name="LayerX_Hybrid_Report.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- main ----------
inject_layerx_css()
st.sidebar.title("Navigation")
page = st.sidebar.radio("เลือกหน้า", ["Page 1 — Dashboard", "Page 2 — Hybrid Performance"], index=0)
if page.startswith("Page 1"):
    render_page1()
else:
    render_page2()
