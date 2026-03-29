import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(150deg, #E3F2FD 0%, #FFFFFF 55%, #E1F5FE 100%); }

    .hero-card {
        background: linear-gradient(135deg, #64B5F6 0%, #42A5F5 50%, #29B6F6 100%);
        border-radius: 24px; padding: 48px 40px; color: white;
        margin-bottom: 32px; box-shadow: 0 16px 48px rgba(66,165,245,0.3); text-align: center;
    }
    .hero-icon { font-size: 72px; margin-bottom: 16px; }
    .hero-title { font-size: 42px; font-weight: 800; margin: 0; letter-spacing: -1px; }
    .hero-subtitle { font-size: 18px; opacity: 0.9; margin-top: 12px; font-weight: 500; }

    .section-card {in present m bola th
        background: #FFFFFF;
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(66,165,245,0.12);
        border-left: 6px solid #64B5F6;
    }
    .section-title {
        font-size: 22px; font-weight: 800; color: #1565C0;
        margin-bottom: 20px; padding-bottom: 10px;
        border-bottom: 2px solid #E3F2FD;
        display: flex; align-items: center; gap: 8px;
    }

    .field-label {
        font-size: 17px; font-weight: 800; color: #0D47A1;
        margin-bottom: 6px; margin-top: 20px; display: block;
        letter-spacing: 0.2px;
    }
    .field-desc { font-size: 13px; color: #78909C; margin-bottom: 6px; font-weight: 500; }

    .bar-wrap { background: #E3F2FD; border-radius: 50px; height: 16px; margin: 8px 0 4px 0; overflow: hidden; }
    .bar-fill { height: 16px; border-radius: 50px; }
    .bar-val { font-size: 13px; font-weight: 700; color: #1976D2; text-align: right; margin-bottom: 14px; }

    .info-card {
        background: #F0F8FF; border-radius: 16px; padding: 20px 24px; margin-bottom: 16px;
        box-shadow: 0 2px 12px rgba(66,165,245,0.1); border-left: 4px solid #64B5F6;
    }
    .info-card h4 { color: #1565C0; margin: 0 0 6px 0; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; font-weight: 800; }
    .info-card p { color: #37474F; margin: 0; font-size: 14px; line-height: 1.6; }

    .result-high {
        background: linear-gradient(135deg, #B71C1C, #E53935);
        border-radius: 20px; padding: 36px; text-align: center; color: white;
        box-shadow: 0 16px 48px rgba(183,28,28,0.3); margin-top: 24px;
    }
    .result-medium {
        background: linear-gradient(135deg, #E65100, #F57C00);
        border-radius: 20px; padding: 36px; text-align: center; color: white;
        box-shadow: 0 16px 48px rgba(230,81,0,0.3); margin-top: 24px;
    }
    .result-low {
        background: linear-gradient(135deg, #1B5E20, #388E3C);
        border-radius: 20px; padding: 36px; text-align: center; color: white;
        box-shadow: 0 16px 48px rgba(27,94,32,0.3); margin-top: 24px;
    }
    .result-emoji { font-size: 64px; margin-bottom: 12px; }
    .result-title { font-size: 28px; font-weight: 800; margin: 8px 0; }
    .result-subtitle { font-size: 16px; opacity: 0.9; }

    .prob-bar-wrap { background: #E3F2FD; border-radius: 50px; height: 24px; margin: 10px 0 6px 0; overflow: hidden; }
    .prob-bar-fill { height: 24px; border-radius: 50px; display: flex; align-items: center; padding-left: 14px; font-size: 14px; font-weight: 800; color: white; }

    .metric-card {
        background: #F0F8FF; border-radius: 16px; padding: 22px 28px;
        box-shadow: 0 4px 16px rgba(66,165,245,0.1); margin-bottom: 14px;
    }
    .metric-value { font-size: 34px; font-weight: 800; }
    .metric-label { font-size: 15px; color: #1565C0; margin-top: 4px; font-weight: 700; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #E3F2FD 0%, #E1F5FE 100%); border-right: 1px solid #BBDEFB; }
    [data-testid="stSidebar"] * { color: #0D47A1 !important; }

    .disclaimer {
        background: #E3F2FD; border-radius: 12px; padding: 16px 20px;
        margin-top: 24px; border: 1px solid #BBDEFB; font-size: 13px; color: #1565C0; font-weight: 500;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        font-size: 17px !important; font-weight: 800 !important; color: #0D47A1 !important;
    }
</style>
""", unsafe_allow_html=True)

MODELS = {
    "Support Vector Machine (SVM)": "models/lung_cancer_svm.pkl",
    "Decision Tree": "models/lung_cancer_decision_tree.pkl",
    "K-Nearest Neighbors (KNN)": "models/lung_cancer_knn.pkl",
    "Logistic Regression": "models/lung_cancer_logistic_regression.pkl",
    "Naive Bayes": "models/lung_cancer_naive_bayes.pkl",
}

def render_bar(value, min_val, max_val, color="#42A5F5"):
    pct = max(0, min(100, int((value - min_val) / (max_val - min_val) * 100)))
    st.markdown(f"""
    <div class="bar-wrap"><div class="bar-fill" style="width:{pct}%;background:{color};"></div></div>
    <div class="bar-val">{value} &nbsp;·&nbsp; {pct}% of range</div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🫁 Lung Cancer")
    st.markdown("---")
    st.markdown("### Choose Algorithm")
    selected_model = st.selectbox("Select Algorithm", list(MODELS.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.page_link("app.py", label="🏠 Back to Home")
    st.page_link("pages/breast_cancer.py", label="🎗️ Breast Cancer")
    st.page_link("pages/liver_cancer.py", label="🫀 Liver Cancer")
    st.markdown("---")
    st.markdown("### Risk Levels")
    st.markdown("🟢 **Low** — Minimal risk")
    st.markdown("🟡 **Medium** — Moderate risk")
    st.markdown("🔴 **High** — High risk")

st.markdown("""
<div class="hero-card">
    <div class="hero-icon">🫁</div>
    <div class="hero-title">Lung Cancer Risk Prediction</div>
    <div class="hero-subtitle">Fill in each section below — the bars show where your values sit in the range</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="info-card"><h4>🎯 Purpose</h4><p>Predict lung cancer risk as Low, Medium, or High based on lifestyle and symptom data.</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="info-card"><h4>📊 Dataset</h4><p>Trained on 1,000+ patient records with 23 features covering symptoms and risk factors.</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="info-card"><h4>⚡ Algorithm</h4><p>Choose from 5 ML algorithms. SVM and Decision Tree typically achieve the highest accuracy.</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Section 1: Patient Information ───────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)

st.markdown('<span class="field-label">🎂 Age</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Patient age in years (1–100)</span>', unsafe_allow_html=True)
age = st.slider("Age", 1, 100, 35, label_visibility="collapsed")
render_bar(age, 1, 100, "#42A5F5")

st.markdown('<span class="field-label">⚧ Gender</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Biological sex of the patient</span>', unsafe_allow_html=True)
gender = st.selectbox("Gender", ["Male", "Female"], label_visibility="collapsed")

st.markdown('<span class="field-label">🚬 Smoking Intensity</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">1 = Very low / non-smoker &nbsp;→&nbsp; 8 = Heavy smoker</span>', unsafe_allow_html=True)
smoking = st.slider("Smoking", 1, 8, 3, label_visibility="collapsed")
render_bar(smoking, 1, 8, "#1E88E5")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Environmental & Lifestyle ─────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌿 Environmental & Lifestyle Factors</div>', unsafe_allow_html=True)

st.markdown('<span class="field-label">🌫️ Air Pollution Exposure</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Level of exposure to polluted air (1=Low → 8=High)</span>', unsafe_allow_html=True)
air_pollution = st.slider("Air Pollution", 1, 8, 3, label_visibility="collapsed")
render_bar(air_pollution, 1, 8, "#42A5F5")

st.markdown('<span class="field-label">🍷 Alcohol Use</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Frequency and amount of alcohol consumption (1=Low → 8=High)</span>', unsafe_allow_html=True)
alcohol = st.slider("Alcohol Use", 1, 8, 3, label_visibility="collapsed")
render_bar(alcohol, 1, 8, "#64B5F6")

st.markdown('<span class="field-label">🤧 Dust Allergy</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Severity of dust allergy (1=None → 8=Severe)</span>', unsafe_allow_html=True)
dust_allergy = st.slider("Dust Allergy", 1, 8, 3, label_visibility="collapsed")
render_bar(dust_allergy, 1, 8, "#42A5F5")

st.markdown('<span class="field-label">🏭 Occupational Hazards</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Exposure to workplace chemicals/toxins (1=None → 8=High)</span>', unsafe_allow_html=True)
occupational = st.slider("Occupational Hazards", 1, 8, 3, label_visibility="collapsed")
render_bar(occupational, 1, 8, "#1E88E5")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Medical History ────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧬 Medical History</div>', unsafe_allow_html=True)

st.markdown('<span class="field-label">🧬 Genetic Risk</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Family/genetic predisposition to lung cancer (1=Low → 8=High)</span>', unsafe_allow_html=True)
genetic_risk = st.slider("Genetic Risk", 1, 8, 3, label_visibility="collapsed")
render_bar(genetic_risk, 1, 8, "#0288D1")

st.markdown('<span class="field-label">🫁 Chronic Lung Disease</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Severity of existing chronic lung conditions (1=None → 8=Severe)</span>', unsafe_allow_html=True)
chronic_lung = st.slider("Chronic Lung Disease", 1, 8, 3, label_visibility="collapsed")
render_bar(chronic_lung, 1, 8, "#1565C0")

st.markdown('<span class="field-label">🥗 Balanced Diet</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Quality of diet — higher means healthier diet (1=Poor → 8=Excellent)</span>', unsafe_allow_html=True)
balanced_diet = st.slider("Balanced Diet", 1, 8, 3, label_visibility="collapsed")
render_bar(balanced_diet, 1, 8, "#26C6DA")

st.markdown('<span class="field-label">⚖️ Obesity</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Degree of obesity (1=Normal weight → 8=Severely obese)</span>', unsafe_allow_html=True)
obesity = st.slider("Obesity", 1, 8, 3, label_visibility="collapsed")
render_bar(obesity, 1, 8, "#0288D1")

st.markdown('<span class="field-label">💨 Passive Smoker</span>', unsafe_allow_html=True)
st.markdown('<span class="field-desc">Exposure to secondhand smoke (1=None → 8=Heavy exposure)</span>', unsafe_allow_html=True)
passive_smoker = st.slider("Passive Smoker", 1, 8, 3, label_visibility="collapsed")
render_bar(passive_smoker, 1, 8, "#1565C0")

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 4: Symptoms ───────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🩺 Symptoms</div>', unsafe_allow_html=True)

symptoms = [
    ("chest_pain",      "💢 Chest Pain",               "Frequency/severity of chest pain (1=None → 9=Severe)", 1, 9, 3, "#E53935"),
    ("coughing_blood",  "🩸 Coughing of Blood",         "Presence of blood when coughing (1=None → 9=Frequent)", 1, 9, 3, "#C62828"),
    ("fatigue",         "😴 Fatigue",                   "Level of tiredness/exhaustion (1=None → 9=Extreme)", 1, 9, 3, "#0288D1"),
    ("weight_loss",     "📉 Weight Loss",               "Unexplained weight loss severity (1=None → 8=Severe)", 1, 8, 3, "#1565C0"),
    ("shortness_breath","😮‍💨 Shortness of Breath",     "Difficulty breathing (1=None → 9=Severe)", 1, 9, 3, "#0288D1"),
    ("wheezing",        "🌬️ Wheezing",                  "Whistling sound when breathing (1=None → 8=Severe)", 1, 8, 3, "#26C6DA"),
    ("swallowing",      "😣 Swallowing Difficulty",     "Trouble swallowing food/liquids (1=None → 8=Severe)", 1, 8, 3, "#1565C0"),
    ("clubbing",        "🖐️ Clubbing of Finger Nails",  "Nail clubbing severity (1=None → 9=Severe)", 1, 9, 3, "#0288D1"),
    ("frequent_cold",   "🤧 Frequent Cold",             "How often the patient gets colds (1=Rarely → 7=Very often)", 1, 7, 3, "#26C6DA"),
    ("dry_cough",       "😤 Dry Cough",                 "Frequency of dry cough (1=None → 8=Constant)", 1, 8, 3, "#0288D1"),
    ("snoring",         "😴 Snoring",                   "Severity of snoring (1=None → 7=Severe)", 1, 7, 3, "#1565C0"),
]

vals = {}
for key, label, desc, mn, mx, default, color in symptoms:
    st.markdown(f'<span class="field-label">{label}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="field-desc">{desc}</span>', unsafe_allow_html=True)
    vals[key] = st.slider(label, mn, mx, default, label_visibility="collapsed", key=f"lung_{key}")
    render_bar(vals[key], mn, mx, color)

chest_pain      = vals["chest_pain"]
coughing_blood  = vals["coughing_blood"]
fatigue         = vals["fatigue"]
weight_loss     = vals["weight_loss"]
shortness_breath= vals["shortness_breath"]
wheezing        = vals["wheezing"]
swallowing      = vals["swallowing"]
clubbing        = vals["clubbing"]
frequent_cold   = vals["frequent_cold"]
dry_cough       = vals["dry_cough"]
snoring         = vals["snoring"]

st.markdown('</div>', unsafe_allow_html=True)

gender_val = 1 if gender == "Male" else 2
input_data = np.array([[age, gender_val, air_pollution, alcohol, dust_allergy,
                         occupational, genetic_risk, chronic_lung, balanced_diet,
                         obesity, smoking, passive_smoker, chest_pain, coughing_blood,
                         fatigue, weight_loss, shortness_breath, wheezing, swallowing,
                         clubbing, frequent_cold, dry_cough, snoring]])

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Lung Cancer Risk", use_container_width=True)

if predict_clicked:
    model_path = MODELS[selected_model]
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found. Please run `python train_lung_cancer_models.py` first.")
    else:
        with st.spinner("Analyzing patient data..."):
            saved = joblib.load(model_path)
            model = saved["model"]
            le = saved["label_encoder"]
            X = input_data.copy()
            if "imputer" in saved:
                X = saved["imputer"].transform(X)
            if "scaler" in saved and selected_model != "Decision Tree":
                X = saved["scaler"].transform(X)
            pred_enc = model.predict(X)[0]
            prediction = le.inverse_transform([pred_enc])[0]
            proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

        if prediction == "High":
            st.markdown('<div class="result-high"><div class="result-emoji">⚠️</div><div class="result-title">High Risk Detected</div><div class="result-subtitle">The model predicts a HIGH risk of lung cancer. Please consult a medical professional immediately.</div></div>', unsafe_allow_html=True)
        elif prediction == "Medium":
            st.markdown('<div class="result-medium"><div class="result-emoji">🔶</div><div class="result-title">Medium Risk Detected</div><div class="result-subtitle">The model predicts a MEDIUM risk. Regular monitoring and lifestyle changes are recommended.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-low"><div class="result-emoji">✅</div><div class="result-title">Low Risk Detected</div><div class="result-subtitle">The model predicts a LOW risk of lung cancer. Maintain your healthy lifestyle!</div></div>', unsafe_allow_html=True)

        if proba is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<strong style='color:#0288D1;font-size:20px;'>📊 Prediction Confidence</strong>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            classes = le.classes_
            colors = {"Low": "#388E3C", "Medium": "#F57C00", "High": "#E53935"}
            for i, cls in enumerate(classes):
                pct = int(proba[i] * 100)
                c = colors.get(cls, "#0288D1")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{cls} Risk</div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-fill" style="width:{pct}%;background:{c};">{pct}%</div>
                    </div>
                    <div class="metric-value" style="color:{c}">{proba[i]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">⚕️ <strong>Medical Disclaimer:</strong> This prediction is for educational purposes only. Always consult a qualified healthcare provider.</div>', unsafe_allow_html=True)
