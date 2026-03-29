import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Light rose/pink background */
    .stApp { background: linear-gradient(150deg, #FFF0F5 0%, #FFFFFF 55%, #FFE4EE 100%); }

    /* Hero — light rose gradient */
    .hero-card {
        background: linear-gradient(135deg, #F48FB1 0%, #F06292 50%, #EC407A 100%);
        border-radius: 24px; padding: 48px 40px; color: white;
        margin-bottom: 32px; box-shadow: 0 16px 48px rgba(240,98,146,0.3); text-align: center;
    }
    .hero-icon { font-size: 72px; margin-bottom: 16px; }
    .hero-title { font-size: 42px; font-weight: 800; margin: 0; letter-spacing: -1px; }
    .hero-subtitle { font-size: 18px; opacity: 0.95; margin-top: 12px; font-weight: 600; }

    /* Section cards */
    .section-card {
        background: #FFFFFF; border-radius: 20px; padding: 28px 32px;
        margin-bottom: 24px; box-shadow: 0 4px 20px rgba(240,98,146,0.12);
        border-left: 6px solid #F48FB1;
    }
    .section-title {
        font-size: 22px; font-weight: 800; color: #C2185B;
        margin-bottom: 20px; padding-bottom: 12px;
        border-bottom: 2px solid #FCE4EC;
    }

    /* Field labels — big and bold */
    .field-label {
        font-size: 17px; font-weight: 800; color: #880E4F;
        margin-top: 20px; margin-bottom: 4px; display: block;
        letter-spacing: 0.2px;
    }
    .field-desc { font-size: 13px; color: #AD7090; margin-bottom: 6px; display: block; font-weight: 500; }

    /* Progress bar */
    .bar-track { background: #FCE4EC; border-radius: 50px; height: 16px; margin: 8px 0 4px 0; overflow: hidden; }
    .bar-fill  { height: 16px; border-radius: 50px; }
    .bar-val   { font-size: 13px; font-weight: 700; color: #C2185B; text-align: right; margin-bottom: 14px; }

    /* Info cards */
    .info-card {
        background: #FFF5F8; border-radius: 16px; padding: 20px 24px; margin-bottom: 16px;
        box-shadow: 0 2px 12px rgba(240,98,146,0.1); border-left: 4px solid #F48FB1;
    }
    .info-card h4 { color: #C2185B; margin: 0 0 6px 0; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; font-weight: 800; }
    .info-card p  { color: #6D4C41; margin: 0; font-size: 14px; line-height: 1.6; }

    .helper-text {
        background: #E8F4FD; border-radius: 10px; padding: 14px 18px;
        font-size: 14px; color: #1565C0; margin-bottom: 20px;
        border-left: 4px solid #42A5F5; font-weight: 500;
    }

    /* Results */
    .result-malignant {
        background: linear-gradient(135deg, #C2185B, #E91E63);
        border-radius: 20px; padding: 36px; text-align: center; color: white;
        box-shadow: 0 16px 48px rgba(194,24,91,0.3); margin-top: 24px;
    }
    .result-benign {
        background: linear-gradient(135deg, #2E7D32, #43A047);
        border-radius: 20px; padding: 36px; text-align: center; color: white;
        box-shadow: 0 16px 48px rgba(46,125,50,0.3); margin-top: 24px;
    }
    .result-emoji { font-size: 64px; margin-bottom: 12px; }
    .result-title { font-size: 30px; font-weight: 800; margin: 8px 0; }
    .result-subtitle { font-size: 16px; opacity: 0.95; font-weight: 500; }

    /* Confidence bars */
    .prob-track { background: #FCE4EC; border-radius: 50px; height: 24px; margin: 10px 0 6px 0; overflow: hidden; }
    .prob-fill  { height: 24px; border-radius: 50px; display: flex; align-items: center;
                  padding-left: 14px; font-size: 14px; font-weight: 800; color: white; }
    .metric-card {
        background: #FFF5F8; border-radius: 16px; padding: 22px 28px;
        box-shadow: 0 4px 16px rgba(240,98,146,0.1); margin-bottom: 14px;
    }
    .metric-label { font-size: 15px; color: #880E4F; font-weight: 700; margin-bottom: 4px; }
    .metric-value { font-size: 34px; font-weight: 800; margin-top: 6px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #FCE4EC 0%, #FFF0F5 100%); border-right: 1px solid #F8BBD0; }
    [data-testid="stSidebar"] * { color: #880E4F !important; }

    .disclaimer {
        background: #FFF0F5; border-radius: 12px; padding: 16px 20px;
        margin-top: 24px; border: 1px solid #F8BBD0; font-size: 13px; color: #880E4F; font-weight: 500;
    }

    /* Force Streamlit slider/select labels bold */
    div[data-testid="stSlider"] label,
    div[data-testid="stSelectbox"] label {
        font-size: 17px !important; font-weight: 800 !important; color: #880E4F !important;
    }
</style>
""", unsafe_allow_html=True)

MODELS = {
    "Support Vector Machine (SVM)": "models/breast_cancer_svm.pkl",
    "Decision Tree": "models/breast_cancer_decision_tree.pkl",
    "K-Nearest Neighbors (KNN)": "models/breast_cancer_knn.pkl",
    "Logistic Regression": "models/breast_cancer_logistic_regression.pkl",
    "Naive Bayes": "models/breast_cancer_naive_bayes.pkl",
}

def bar(value, mn, mx, color="#EC407A"):
    pct = max(0, min(100, int((value - mn) / (mx - mn) * 100)))
    st.markdown(f"""
    <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color};"></div></div>
    <div class="bar-val">{value} &nbsp;·&nbsp; {pct}% of range</div>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎗️ Breast Cancer")
    st.markdown("---")
    st.markdown("### Choose Algorithm")
    selected_model = st.selectbox("Select Algorithm", list(MODELS.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.page_link("app.py", label="🏠 Back to Home")
    st.page_link("pages/lung_cancer.py", label="🫁 Lung Cancer")
    st.page_link("pages/liver_cancer.py", label="🫀 Liver Cancer")
    st.markdown("---")
    st.markdown("### Results Guide")
    st.markdown("🟢 **Benign (B)** — Non-cancerous")
    st.markdown("🔴 **Malignant (M)** — Cancerous")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div class="hero-icon">🎗️</div>
    <div class="hero-title">Breast Cancer Detection</div>
    <div class="hero-subtitle">Enter tumor measurements below — each bar shows where the value sits in the expected range</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="info-card"><h4>🎯 Purpose</h4><p>Classify breast tumors as Benign or Malignant using cell nucleus measurements.</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="info-card"><h4>📊 Dataset</h4><p>Wisconsin Breast Cancer Dataset — 569 records, 30 features.</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="info-card"><h4>🏥 Note</h4><p>Measurements come from fine needle aspirate (FNA) of a breast mass.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="helper-text">ℹ️ These values are tumor cell nucleus measurements from medical imaging — typically provided by a radiologist or pathologist report.</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Helper: one vertical field ────────────────────────────────────────────────
def slider_field(label, desc, key, mn, mx, default, step, color):
    st.markdown(f'<span class="field-label">{label}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="field-desc">{desc}</span>', unsafe_allow_html=True)
    val = st.slider(label, mn, mx, default, step, label_visibility="collapsed", key=key)
    bar(val, mn, mx, color)
    return val

# ── Section 1: Mean Measurements ─────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📐 Mean Measurements</div>', unsafe_allow_html=True)

radius_mean         = slider_field("📏 Radius Mean",          "Mean distance from center to perimeter (6–30)",          "rm",  6.0,   30.0,   14.0,  0.1,    "#EC407A")
texture_mean        = slider_field("🔲 Texture Mean",         "Standard deviation of gray-scale values (9–40)",         "tm",  9.0,   40.0,   19.0,  0.1,    "#F06292")
perimeter_mean      = slider_field("📐 Perimeter Mean",       "Mean size of the core tumor (40–190)",                   "pm",  40.0,  190.0,  91.0,  0.1,    "#EC407A")
area_mean           = slider_field("📦 Area Mean",            "Mean area of the tumor nucleus (140–2500)",              "am",  140.0, 2500.0, 654.0, 1.0,    "#F06292")
smoothness_mean     = slider_field("〰️ Smoothness Mean",      "Local variation in radius lengths (0.05–0.17)",          "sm",  0.05,  0.17,   0.10,  0.001,  "#EC407A")
compactness_mean    = slider_field("🔵 Compactness Mean",     "Perimeter² / area − 1.0 (0.02–0.35)",                   "cm",  0.02,  0.35,   0.10,  0.001,  "#F06292")
concavity_mean      = slider_field("🌀 Concavity Mean",       "Severity of concave portions of contour (0–0.43)",       "cnm", 0.0,   0.43,   0.09,  0.001,  "#EC407A")
concave_points_mean = slider_field("📍 Concave Points Mean",  "Number of concave portions of contour (0–0.20)",         "cpm", 0.0,   0.20,   0.05,  0.001,  "#F06292")
symmetry_mean       = slider_field("⚖️ Symmetry Mean",        "Symmetry of the tumor (0.10–0.30)",                      "sym", 0.10,  0.30,   0.18,  0.001,  "#EC407A")
fractal_mean        = slider_field("🔬 Fractal Dimension Mean","Coastline approximation (0.05–0.10)",                   "fdm", 0.05,  0.10,   0.063, 0.001,  "#F06292")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: SE Measurements ────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📏 Standard Error (SE) Measurements</div>', unsafe_allow_html=True)

radius_se         = slider_field("📏 Radius SE",          "Standard error of radius (0.1–3.0)",              "rse",  0.1,    3.0,   0.4,   0.01,   "#E91E63")
texture_se        = slider_field("🔲 Texture SE",         "Standard error of texture (0.3–5.0)",             "tse",  0.3,    5.0,   1.2,   0.01,   "#F06292")
perimeter_se      = slider_field("📐 Perimeter SE",       "Standard error of perimeter (0.7–22.0)",          "pse",  0.7,    22.0,  2.9,   0.1,    "#E91E63")
area_se           = slider_field("📦 Area SE",            "Standard error of area (6–550)",                  "ase",  6.0,    550.0, 40.0,  1.0,    "#F06292")
smoothness_se     = slider_field("〰️ Smoothness SE",      "Standard error of smoothness (0.001–0.03)",       "sse",  0.001,  0.03,  0.007, 0.0001, "#E91E63")
compactness_se    = slider_field("🔵 Compactness SE",     "Standard error of compactness (0.002–0.14)",      "cse",  0.002,  0.14,  0.025, 0.001,  "#F06292")
concavity_se      = slider_field("🌀 Concavity SE",       "Standard error of concavity (0–0.40)",            "cvse", 0.0,    0.40,  0.032, 0.001,  "#E91E63")
concave_points_se = slider_field("📍 Concave Points SE",  "Standard error of concave points (0–0.053)",      "cpse", 0.0,    0.053, 0.012, 0.001,  "#F06292")
symmetry_se       = slider_field("⚖️ Symmetry SE",        "Standard error of symmetry (0.007–0.08)",         "syse", 0.007,  0.08,  0.020, 0.001,  "#E91E63")
fractal_se        = slider_field("🔬 Fractal Dimension SE","Standard error of fractal dimension (0.0008–0.03)","fdse",0.0008, 0.030, 0.004, 0.0001, "#F06292")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Worst Measurements ────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Worst Measurements</div>', unsafe_allow_html=True)

radius_worst         = slider_field("📏 Radius Worst",          "Largest mean radius value (7–37)",                    "rw",  7.0,   37.0,   16.0,  0.1,   "#C2185B")
texture_worst        = slider_field("🔲 Texture Worst",         "Largest mean texture value (12–50)",                  "tw",  12.0,  50.0,   25.0,  0.1,   "#E91E63")
perimeter_worst      = slider_field("📐 Perimeter Worst",       "Largest mean perimeter value (50–252)",               "pw",  50.0,  252.0,  107.0, 0.1,   "#C2185B")
area_worst           = slider_field("📦 Area Worst",            "Largest mean area value (185–4250)",                  "aw",  185.0, 4250.0, 880.0, 1.0,   "#E91E63")
smoothness_worst     = slider_field("〰️ Smoothness Worst",      "Largest mean smoothness value (0.07–0.22)",           "sw",  0.07,  0.22,   0.13,  0.001, "#C2185B")
compactness_worst    = slider_field("🔵 Compactness Worst",     "Largest mean compactness value (0.02–1.10)",          "cw",  0.02,  1.10,   0.25,  0.01,  "#E91E63")
concavity_worst      = slider_field("🌀 Concavity Worst",       "Largest mean concavity value (0–1.25)",               "cvw", 0.0,   1.25,   0.27,  0.01,  "#C2185B")
concave_points_worst = slider_field("📍 Concave Points Worst",  "Largest mean concave points value (0–0.29)",          "cpw", 0.0,   0.29,   0.11,  0.001, "#E91E63")
symmetry_worst       = slider_field("⚖️ Symmetry Worst",        "Largest mean symmetry value (0.15–0.66)",             "syw", 0.15,  0.66,   0.29,  0.01,  "#C2185B")
fractal_worst        = slider_field("🔬 Fractal Dimension Worst","Largest mean fractal dimension value (0.05–0.21)",   "fdw", 0.05,  0.21,   0.08,  0.001, "#E91E63")

st.markdown('</div>', unsafe_allow_html=True)

# ── Build input array ─────────────────────────────────────────────────────────
input_data = np.array([[
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_mean,
    radius_se, texture_se, perimeter_se, area_se, smoothness_se,
    compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_se,
    radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
    compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_worst,
]])

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Tumor Type", use_container_width=True)

if predict_clicked:
    model_path = MODELS[selected_model]
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found. Please run `python train_breast_cancer_models.py` first.")
    else:
        with st.spinner("Analyzing tumor measurements..."):
            saved = joblib.load(model_path)
            model = saved["model"]
            le = saved["label_encoder"]
            X = input_data.copy()
            if "imputer" in saved:
                X = saved["imputer"].transform(X)
            if "scaler" in saved:
                X = saved["scaler"].transform(X)
            pred_enc = model.predict(X)[0]
            prediction = le.inverse_transform([pred_enc])[0]
            proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

        if prediction == "M":
            st.markdown('<div class="result-malignant"><div class="result-emoji">⚠️</div><div class="result-title">Malignant Tumor Detected</div><div class="result-subtitle">The model predicts this tumor is MALIGNANT (cancerous). Please seek immediate attention from an oncologist.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-benign"><div class="result-emoji">✅</div><div class="result-title">Benign Tumor Detected</div><div class="result-subtitle">The model predicts this tumor is BENIGN (non-cancerous). Continue regular check-ups with your doctor.</div></div>', unsafe_allow_html=True)

        if proba is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<strong style='color:#C2185B;font-size:20px;'>📊 Prediction Confidence</strong>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            classes = le.classes_
            labels = {"B": "✅ Benign — Non-cancerous", "M": "⚠️ Malignant — Cancerous"}
            colors = {"B": "#43A047", "M": "#C2185B"}
            for i, cls in enumerate(classes):
                pct = max(0, min(100, int(proba[i] * 100)))
                c = colors.get(cls, "#C2185B")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{labels.get(cls, cls)}</div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{pct}%;background:{c};">{pct}%</div>
                    </div>
                    <div class="metric-value" style="color:{c}">{proba[i]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">⚕️ <strong>Medical Disclaimer:</strong> This prediction is for educational purposes only. Always consult a qualified healthcare provider.</div>', unsafe_allow_html=True)
