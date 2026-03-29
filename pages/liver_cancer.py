import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Liver Cancer Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(150deg, #FFFDE7 0%, #FFFFFF 60%, #FFF8E1 100%); }

    .hero-card {
        background: linear-gradient(135deg, #FFD54F 0%, #FFCA28 50%, #FFE082 100%);
        border-radius: 24px; padding: 48px 40px; color: #4E342E;
        margin-bottom: 32px; box-shadow: 0 20px 60px rgba(255,193,7,0.3); text-align: center;
    }
    .hero-icon { font-size: 72px; margin-bottom: 16px; }
    .hero-title { font-size: 42px; font-weight: 800; margin: 0; letter-spacing: -1px; color: #4E342E; }
    .hero-subtitle { font-size: 18px; margin-top: 12px; font-weight: 500; color: #6D4C41; }

    .section-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(255,193,7,0.15);
        border-left: 6px solid #FFCA28;
    }
    .section-title {
        font-size: 24px; font-weight: 800; color: #4E342E;
        margin-bottom: 20px; padding-bottom: 12px;
        border-bottom: 3px solid #FFE082;
        display: flex; align-items: center; gap: 8px;
        letter-spacing: -0.3px;
    }

    .field-label {
        font-size: 18px; font-weight: 800; color: #3E2723;
        margin-bottom: 4px; margin-top: 20px; display: block;
        letter-spacing: 0.1px;
    }
    .field-desc { font-size: 13px; color: #A1887F; margin-bottom: 4px; font-weight: 500; }

    .bar-wrap { background: #FFF8E1; border-radius: 50px; height: 14px; margin: 6px 0 2px 0; overflow: hidden; }
    .bar-fill { height: 14px; border-radius: 50px; transition: width 0.3s; }

    .bar-val { font-size: 13px; font-weight: 700; color: #E08000; text-align: right; margin-bottom: 12px; }

    .info-card {
        background: white; border-radius: 16px; padding: 20px 24px; margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(255,193,7,0.12); border-left: 4px solid #FFCA28;
    }
    .info-card h4 { color: #E08000; margin: 0 0 6px 0; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
    .info-card p { color: #5D4037; margin: 0; font-size: 14px; line-height: 1.6; }

    .result-cancer {
        background: linear-gradient(135deg, #B71C1C, #E53935);
        border-radius: 20px; padding: 36px; text-align: center; color: white;
        box-shadow: 0 16px 48px rgba(183,28,28,0.3); margin-top: 24px;
    }
    .result-no-cancer {
        background: linear-gradient(135deg, #1B5E20, #388E3C);
        border-radius: 20px; padding: 36px; text-align: center; color: white;
        box-shadow: 0 16px 48px rgba(27,94,32,0.3); margin-top: 24px;
    }
    .result-emoji { font-size: 64px; margin-bottom: 12px; }
    .result-title { font-size: 28px; font-weight: 800; margin: 8px 0; }
    .result-subtitle { font-size: 16px; opacity: 0.9; }

    .prob-bar-wrap { background: #FFF8E1; border-radius: 50px; height: 22px; margin: 8px 0 4px 0; overflow: hidden; }
    .prob-bar-fill { height: 22px; border-radius: 50px; display: flex; align-items: center; padding-left: 12px; font-size: 13px; font-weight: 700; color: white; }

    .metric-card {
        background: white; border-radius: 14px; padding: 20px;
        text-align: center; box-shadow: 0 4px 16px rgba(255,193,7,0.15);
        margin-bottom: 12px;
    }
    .metric-value { font-size: 32px; font-weight: 800; }
    .metric-label { font-size: 14px; color: #A1887F; margin-top: 4px; font-weight: 600; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #FFF8E1 0%, #FFFDE7 100%); border-right: 1px solid #FFE082; }
    [data-testid="stSidebar"] * { color: #5D4037 !important; }

    .disclaimer {
        background: #FFFDE7; border-radius: 12px; padding: 16px 20px;
        margin-top: 24px; border: 1px solid #FFE082; font-size: 13px; color: #E08000;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        font-size: 18px !important; font-weight: 800 !important; color: #3E2723 !important;
    }

    .how-to-box {
        background: #FFFDE7; border-radius: 14px; padding: 16px 20px;
        margin: 10px 0 16px 0; border-left: 5px solid #FFCA28;
        font-size: 13px; color: #5D4037; line-height: 1.8;
    }
    .how-to-box strong { color: #E08000; font-size: 14px; display: block; margin-bottom: 6px; }
    .how-to-step { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 4px; }
    .how-to-step .step-num {
        background: #FFCA28; color: #4E342E; border-radius: 50%;
        width: 20px; height: 20px; min-width: 20px;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 800; margin-top: 1px;
    }
</style>
""", unsafe_allow_html=True)

MODELS = {
    "Support Vector Machine (SVM)": "models/liver_cancer_svm.pkl",
    "Decision Tree": "models/liver_cancer_decision_tree.pkl",
    "K-Nearest Neighbors (KNN)": "models/liver_cancer_knn.pkl",
    "Logistic Regression": "models/liver_cancer_logistic_regression.pkl",
    "Naive Bayes": "models/liver_cancer_naive_bayes.pkl",
}

def render_bar(value, min_val, max_val, color="#FFCA28"):
    pct = int((value - min_val) / (max_val - min_val) * 100)
    st.markdown(f"""
    <div class="bar-wrap"><div class="bar-fill" style="width:{pct}%;background:{color};"></div></div>
    <div class="bar-val">{value} &nbsp;|&nbsp; {pct}% of range</div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🫀 Liver Cancer")
    st.markdown("---")
    st.markdown("### Choose Algorithm")
    selected_model = st.selectbox("Select Algorithm", list(MODELS.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.page_link("app.py", label="🏠 Back to Home")
    st.page_link("pages/lung_cancer.py", label="🫁 Lung Cancer")
    st.page_link("pages/breast_cancer.py", label="🎗️ Breast Cancer")
    st.markdown("---")
    st.markdown("### Key Risk Factors")
    st.markdown("🔴 Hepatitis B/C infection")
    st.markdown("🔴 Cirrhosis history")
    st.markdown("🔴 Heavy alcohol use")
    st.markdown("🔴 Family history of cancer")
    st.markdown("🟡 Diabetes & high BMI")

st.markdown("""
<div class="hero-card">
    <div class="hero-icon">🫀</div>
    <div class="hero-title">Liver Cancer Risk Prediction</div>
    <div class="hero-subtitle">Fill in the patient details below — each section guides you step by step</div>
</div>
""", unsafe_allow_html=True)

# Info cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="info-card"><h4>🎯 Purpose</h4><p>Predict liver cancer risk based on clinical and lifestyle factors.</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="info-card"><h4>📊 Dataset</h4><p>Trained on 5,000 patient records covering demographics, lifestyle, and medical history.</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="info-card"><h4>🏥 Key Factors</h4><p>Hepatitis, cirrhosis, alcohol use, AFP levels, and family history are major predictors.</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Section 1: Patient Demographics ──────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Patient Demographics</div>', unsafe_allow_html=True)

st.markdown('<span class="field-label">🎂 Age</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Patient age in years (18–90)</span>', unsafe_allow_html=True)
age = st.slider("Age", 18, 90, 50, label_visibility="collapsed")
render_bar(age, 18, 90, "#FFCA28")

st.markdown('<span class="field-label">⚧ Gender</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Biological sex of the patient</span>', unsafe_allow_html=True)
gender = st.selectbox("Gender", ["Male", "Female"], label_visibility="collapsed")

st.markdown('<span class="field-label">⚖️ BMI (Body Mass Index)</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Body mass index — higher BMI increases liver cancer risk</span>', unsafe_allow_html=True)
bmi = st.slider("BMI", 10.0, 45.0, 25.0, 0.1, label_visibility="collapsed")
render_bar(bmi, 10.0, 45.0, "#FFB300")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Lifestyle Factors ─────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🍺 Lifestyle Factors</div>', unsafe_allow_html=True)

st.markdown('<span class="field-label">🍷 Alcohol Consumption</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">How often does the patient consume alcohol?</span>', unsafe_allow_html=True)
alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasional", "Regular"], label_visibility="collapsed")

st.markdown('<span class="field-label">🚬 Smoking Status</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Current smoking habits of the patient</span>', unsafe_allow_html=True)
smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"], label_visibility="collapsed")

st.markdown('<span class="field-label">🏃 Physical Activity Level</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">How active is the patient on a regular basis?</span>', unsafe_allow_html=True)
physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"], label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Medical History ────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧬 Medical History</div>', unsafe_allow_html=True)

st.markdown('<span class="field-label">🦠 Hepatitis B</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Has the patient been diagnosed with Hepatitis B?</span>', unsafe_allow_html=True)
hepatitis_b = st.selectbox("Hepatitis B", ["No", "Yes"], label_visibility="collapsed")

st.markdown('<span class="field-label">🦠 Hepatitis C</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Has the patient been diagnosed with Hepatitis C?</span>', unsafe_allow_html=True)
hepatitis_c = st.selectbox("Hepatitis C", ["No", "Yes"], label_visibility="collapsed")

st.markdown('<span class="field-label">🫀 Cirrhosis History</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Does the patient have a history of liver cirrhosis?</span>', unsafe_allow_html=True)
cirrhosis = st.selectbox("Cirrhosis History", ["No", "Yes"], label_visibility="collapsed")

st.markdown('<span class="field-label">👨‍👩‍👧 Family History of Cancer</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Any first-degree relatives diagnosed with cancer?</span>', unsafe_allow_html=True)
family_history = st.selectbox("Family History of Cancer", ["No", "Yes"], label_visibility="collapsed")

st.markdown('<span class="field-label">🩸 Diabetes</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Has the patient been diagnosed with diabetes?</span>', unsafe_allow_html=True)
diabetes = st.selectbox("Diabetes", ["No", "Yes"], label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 4: Lab Values ─────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔬 Lab Values</div>', unsafe_allow_html=True)

st.markdown('<span class="field-label">🧪 Liver Function Score</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Higher scores may indicate liver dysfunction (range: 10–120)</span>', unsafe_allow_html=True)
st.markdown("""
<div class="how-to-box">
    <strong>🩺 How to find your Liver Function Score?</strong>
    <div class="how-to-step"><div class="step-num">1</div><span>Ask your doctor for a <b>Liver Function Test (LFT)</b> — a routine blood test.</span></div>
    <div class="how-to-step"><div class="step-num">2</div><span>The test measures enzymes like <b>ALT, AST, ALP</b> and proteins like <b>albumin and bilirubin</b>.</span></div>
    <div class="how-to-step"><div class="step-num">3</div><span>Your lab report will show a combined score or individual values — ask your doctor to summarise it as a score out of 120.</span></div>
    <div class="how-to-step"><div class="step-num">4</div><span><b>Normal range:</b> 10–40 &nbsp;|&nbsp; <b>Elevated:</b> 40–80 &nbsp;|&nbsp; <b>High concern:</b> 80–120</span></div>
</div>
""", unsafe_allow_html=True)
liver_score = st.slider("Liver Function Score", 10.0, 120.0, 65.0, 0.1, label_visibility="collapsed")
render_bar(liver_score, 10.0, 120.0, "#F57C00")

st.markdown('<span class="field-label">📈 Alpha-Fetoprotein (AFP) Level</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">AFP > 400 ng/mL is strongly associated with liver cancer (range: 0–120)</span>', unsafe_allow_html=True)
st.markdown("""
<div class="how-to-box">
    <strong>🩺 How to find your AFP Level?</strong>
    <div class="how-to-step"><div class="step-num">1</div><span>AFP is measured via a simple <b>blood test</b> — request an <b>Alpha-Fetoprotein (AFP) test</b> from your doctor.</span></div>
    <div class="how-to-step"><div class="step-num">2</div><span><b>Normal AFP:</b> below 10 ng/mL in healthy adults.</span></div>
    <div class="how-to-step"><div class="step-num">3</div><span><b>Elevated AFP (10–400 ng/mL)</b> may indicate liver disease or other conditions.</span></div>
    <div class="how-to-step"><div class="step-num">4</div><span><b>AFP above 400 ng/mL</b> is strongly associated with hepatocellular carcinoma (liver cancer).</span></div>
</div>
""", unsafe_allow_html=True)
afp_level = st.slider("AFP Level", 0.0, 120.0, 10.0, 0.01, label_visibility="collapsed")
render_bar(afp_level, 0.0, 120.0, "#E65100")

st.markdown('</div>', unsafe_allow_html=True)

# ── Encode inputs ─────────────────────────────────────────────────────────────
alcohol_map = {"Never": 0, "Occasional": 1, "Regular": 2}
smoking_map = {"Current": 0, "Former": 1, "Never": 2}
gender_map = {"Female": 0, "Male": 1}
activity_map = {"High": 0, "Low": 1, "Moderate": 2}

input_data = np.array([[
    age, gender_map[gender], bmi, alcohol_map[alcohol], smoking_map[smoking],
    1 if hepatitis_b == "Yes" else 0,
    1 if hepatitis_c == "Yes" else 0,
    liver_score, afp_level,
    1 if cirrhosis == "Yes" else 0,
    1 if family_history == "Yes" else 0,
    activity_map[physical_activity],
    1 if diabetes == "Yes" else 0,
]])

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Liver Cancer Risk", use_container_width=True)

if predict_clicked:
    model_path = MODELS[selected_model]
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found. Please run `python train_liver_cancer_models.py` first.")
    else:
        with st.spinner("Analyzing patient data..."):
            saved = joblib.load(model_path)
            model = saved["model"]
            X = input_data.copy()
            if "imputer" in saved:
                X = saved["imputer"].transform(X)
            if "scaler" in saved:
                X = saved["scaler"].transform(X)
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.markdown('<div class="result-cancer"><div class="result-emoji">⚠️</div><div class="result-title">Liver Cancer Risk Detected</div><div class="result-subtitle">The model predicts a HIGH likelihood of liver cancer. Please consult a hepatologist or oncologist immediately.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-no-cancer"><div class="result-emoji">✅</div><div class="result-title">Low Liver Cancer Risk</div><div class="result-subtitle">The model predicts a LOW likelihood of liver cancer. Maintain healthy habits and attend regular check-ups.</div></div>', unsafe_allow_html=True)

        if proba is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<strong style='color:#E08000;font-size:20px;'>📊 Prediction Confidence</strong>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            no_cancer_pct = int(proba[0] * 100)
            cancer_pct = int(proba[1] * 100)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">✅ No Cancer Probability</div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{no_cancer_pct}%;background:#388E3C;">{no_cancer_pct}%</div>
                </div>
                <div class="metric-value" style="color:#388E3C">{proba[0]*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">⚠️ Cancer Probability</div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{cancer_pct}%;background:#E53935;">{cancer_pct}%</div>
                </div>
                <div class="metric-value" style="color:#E53935">{proba[1]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">⚕️ <strong>Medical Disclaimer:</strong> This prediction is for educational purposes only. Always consult a qualified healthcare provider.</div>', unsafe_allow_html=True)
