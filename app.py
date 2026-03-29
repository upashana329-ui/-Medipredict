import streamlit as st

st.set_page_config(
    page_title="MediPredict — Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(150deg, #FFF0F5 0%, #FFFFFF 50%, #FFE4EE 100%); }

    .hero {
        background: linear-gradient(135deg, #FFB6C1 0%, #FFC0CB 50%, #FFD6E0 100%);
        border-radius: 28px;
        padding: 64px 48px;
        color: #6B2D4E;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 16px 60px rgba(255, 182, 193, 0.45);
        position: relative;
        overflow: hidden;
        border: 1px solid #FFD6E0;
    }
    .hero::before {
        content: '';
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.25); border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute; bottom: -80px; left: -40px;
        width: 280px; height: 280px;
        background: rgba(255,255,255,0.15); border-radius: 50%;
    }
    .hero-icon { font-size: 80px; margin-bottom: 20px; }
    .hero-title { font-size: 52px; font-weight: 800; margin: 0; letter-spacing: -2px; color: #5C1A3A; }
    .hero-tagline { font-size: 20px; opacity: 0.85; margin-top: 14px; font-weight: 400; color: #7A3055; }
    .hero-stats { display: flex; justify-content: center; gap: 48px; margin-top: 36px; flex-wrap: wrap; }
    .stat { text-align: center; }
    .stat-num { font-size: 32px; font-weight: 700; color: #5C1A3A; }
    .stat-label { font-size: 13px; color: #9E5070; margin-top: 2px; }

    .search-container {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 32px;
        box-shadow: 0 4px 24px rgba(255, 182, 193, 0.25);
        border: 1px solid #FFD6E0;
    }

    .disease-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 32px 28px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 182, 193, 0.2);
        border-top: 5px solid;
        height: 100%;
    }
    .card-icon { font-size: 56px; margin-bottom: 16px; }
    .card-title { font-size: 22px; font-weight: 700; margin-bottom: 10px; }
    .card-desc { font-size: 14px; color: #8A7080; line-height: 1.7; margin-bottom: 20px; }
    .card-tag { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin: 3px; }

    .lung-card { border-top-color: #89CFF0; }
    .lung-title { color: #4A90B8; }
    .breast-card { border-top-color: #FFB6C1; }
    .breast-title { color: #C2185B; }
    .liver-card { border-top-color: #FFCA28; }
    .liver-title { color: #C68000; }

    .symptom-result {
        background: #FFF0F5;
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 16px;
        border-left: 4px solid #FFB6C1;
    }
    .symptom-result-title { font-size: 15px; font-weight: 600; color: #7A3055; margin-bottom: 10px; }

    .match-chip {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px;
        color: white;
    }

    .no-match {
        background: #FFF8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 12px;
        color: #C06000;
        font-size: 14px;
    }

    .section-header {
        font-size: 28px; font-weight: 800; color: #7A3055;
        margin: 40px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #FFD6E0;
    }

    .how-card {
        background: #FFFFFF; border-radius: 16px; padding: 28px;
        text-align: center; box-shadow: 0 4px 16px rgba(255,182,193,0.2);
        border-bottom: 4px solid #FFB6C1;
    }
    .how-num { font-size: 40px; font-weight: 800; color: #D4708A; }
    .how-title { font-size: 17px; font-weight: 700; color: #7A3055; margin: 8px 0 6px 0; }
    .how-desc { font-size: 13px; color: #9E8090; line-height: 1.7; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFE4EE 0%, #FFF0F5 100%);
        border-right: 1px solid #FFD6E0;
    }
    [data-testid="stSidebar"] * { color: #6B2D4E !important; }

    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #C2185B; }
    .stTabs [aria-selected="true"] { background: #FFF0F5; border-radius: 8px 8px 0 0; }

    .welcome-banner {
        background: linear-gradient(135deg, #FFF0F5, #FFFFFF);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 28px;
        border-left: 5px solid #FFB6C1;
        font-size: 15px;
        color: #7A3055;
        font-weight: 500;
    }

    .footer {
        text-align: center; padding: 40px 32px; color: #B06080; font-size: 13px;
        background: #FFF5F8; border-radius: 20px; margin-top: 48px;
        border: 1px solid #FFD6E0;
    }

    @media (max-width: 768px) {
        .hero-title { font-size: 32px; }
        .hero-stats { gap: 24px; }
        .search-container { padding: 24px 20px; }
    }
</style>
""", unsafe_allow_html=True)

SYMPTOM_MAP = {
    "cough": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "dry cough": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "coughing blood": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "chest pain": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "shortness of breath": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "wheezing": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "snoring": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "swallowing difficulty": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "clubbing": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "fatigue": {"diseases": ["Lung Cancer", "Liver Cancer"], "colors": ["#0288D1", "#F57C00"]},
    "weight loss": {"diseases": ["Lung Cancer", "Liver Cancer"], "colors": ["#0288D1", "#F57C00"]},
    "smoking": {"diseases": ["Lung Cancer", "Liver Cancer"], "colors": ["#0288D1", "#F57C00"]},
    "air pollution": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "dust allergy": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "occupational hazard": {"diseases": ["Lung Cancer"], "colors": ["#0288D1"]},
    "genetic": {"diseases": ["Lung Cancer", "Breast Cancer", "Liver Cancer"], "colors": ["#0288D1", "#C2185B", "#F57C00"]},
    "family history": {"diseases": ["Breast Cancer", "Liver Cancer"], "colors": ["#C2185B", "#F57C00"]},
    "lump": {"diseases": ["Breast Cancer"], "colors": ["#C2185B"]},
    "breast lump": {"diseases": ["Breast Cancer"], "colors": ["#C2185B"]},
    "nipple discharge": {"diseases": ["Breast Cancer"], "colors": ["#C2185B"]},
    "breast pain": {"diseases": ["Breast Cancer"], "colors": ["#C2185B"]},
    "tumor": {"diseases": ["Breast Cancer"], "colors": ["#C2185B"]},
    "radius": {"diseases": ["Breast Cancer"], "colors": ["#C2185B"]},
    "texture": {"diseases": ["Breast Cancer"], "colors": ["#C2185B"]},
    "concavity": {"diseases": ["Breast Cancer"], "colors": ["#C2185B"]},
    "jaundice": {"diseases": ["Liver Cancer"], "colors": ["#F57C00"]},
    "abdominal pain": {"diseases": ["Liver Cancer"], "colors": ["#F57C00"]},
    "hepatitis": {"diseases": ["Liver Cancer"], "colors": ["#F57C00"]},
    "cirrhosis": {"diseases": ["Liver Cancer"], "colors": ["#F57C00"]},
    "alcohol": {"diseases": ["Liver Cancer", "Lung Cancer"], "colors": ["#F57C00", "#0288D1"]},
    "bmi": {"diseases": ["Liver Cancer"], "colors": ["#F57C00"]},
    "diabetes": {"diseases": ["Liver Cancer"], "colors": ["#F57C00"]},
    "liver": {"diseases": ["Liver Cancer"], "colors": ["#F57C00"]},
    "afp": {"diseases": ["Liver Cancer"], "colors": ["#F57C00"]},
    "alpha-fetoprotein": {"diseases": ["Liver Cancer"], "colors": ["#F57C00"]},
    "obesity": {"diseases": ["Lung Cancer", "Liver Cancer"], "colors": ["#0288D1", "#F57C00"]},
}

DISEASE_PAGES = {
    "Lung Cancer": "pages/lung_cancer.py",
    "Breast Cancer": "pages/breast_cancer.py",
    "Liver Cancer": "pages/liver_cancer.py",
}

DISEASE_KEYWORDS = {
    "lung": "Lung Cancer", "lung cancer": "Lung Cancer",
    "breast": "Breast Cancer", "breast cancer": "Breast Cancer",
    "liver": "Liver Cancer", "liver cancer": "Liver Cancer",
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:64px; line-height:1;">🏥</div>
        <div style="font-size:22px; font-weight:800; color:#5C1A3A; margin-top:8px; letter-spacing:-0.5px;">MediPredict</div>
        <div style="font-size:11px; color:#9E5070; margin-top:4px; font-weight:500;">AI-Powered Cancer Risk Assessment</div>
        <div style="margin-top:12px; padding-top:12px; border-top:1px solid #FFD6E0;">
            <div style="font-size:12px; color:#B06080; font-weight:600;">Created by</div>
            <div style="font-size:14px; font-weight:800; color:#7A3055; margin-top:2px;">Upasana Joshi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Navigation")
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/lung_cancer.py", label="🫁 Lung Cancer")
    st.page_link("pages/breast_cancer.py", label="🎗️ Breast Cancer")
    st.page_link("pages/liver_cancer.py", label="🫀 Liver Cancer")
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.markdown("🤖 **5** ML Algorithms")
    st.markdown("🦠 **3** Disease Types")
    st.markdown("📊 **6,500+** Training Records")
    st.markdown("---")
    st.markdown("*Always consult a qualified doctor for medical advice.*")

st.markdown("""
<div class="hero">
    <div class="hero-icon">🏥</div>
    <div class="hero-title">MediPredict</div>
    <div class="hero-tagline">Early Detection Saves Lives — AI-Powered Cancer Risk Assessment</div>
    <div class="hero-stats">
        <div class="stat"><div class="stat-num">3</div><div class="stat-label">Disease Predictions</div></div>
        <div class="stat"><div class="stat-num">5</div><div class="stat-label">ML Algorithms</div></div>
        <div class="stat"><div class="stat-num">6,500+</div><div class="stat-label">Training Records</div></div>
        <div class="stat"><div class="stat-num">95%+</div><div class="stat-label">Model Accuracy</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="welcome-banner">💗 Welcome to <strong>MediPredict</strong>! You have taken a brave step towards understanding your health. Use the search below to find your disease or describe your symptoms, then get an instant AI-powered risk assessment. <strong>Remember — early detection saves lives.</strong></div>', unsafe_allow_html=True)

st.markdown('<div class="search-container">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["🔍 Search by Disease Name", "🩺 Search by Symptom"])

with tab1:
    st.markdown("**Type the name of a disease to quickly find its prediction page.**")
    disease_search = st.text_input("Search disease", placeholder="e.g. lung, breast, liver cancer...", key="disease_search", label_visibility="collapsed")
    if disease_search:
        query = disease_search.strip().lower()
        matches = []
        for kw, disease in DISEASE_KEYWORDS.items():
            if query in kw or kw in query:
                if disease not in matches:
                    matches.append(disease)
        if matches:
            st.markdown(f"**Found {len(matches)} result(s) for '{disease_search}':**")
            icons = {"Lung Cancer": "🫁", "Breast Cancer": "🎗️", "Liver Cancer": "🫀"}
            for m in matches:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {icons[m]} {m}")
                with col2:
                    st.page_link(DISEASE_PAGES[m], label=f"Go to {m} →")
        else:
            st.markdown(f'<div class="no-match">❌ No disease found for "<strong>{disease_search}</strong>". Try: lung, breast, or liver.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#AD1457; font-size:14px;">Start typing to search for a disease...</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("**Enter a symptom or risk factor to find which disease it may be associated with.**")
    symptom_search = st.text_input("Search symptom", placeholder="e.g. cough, jaundice, lump, fatigue, hepatitis...", key="symptom_search", label_visibility="collapsed")
    if symptom_search:
        query = symptom_search.strip().lower()
        matched_diseases = {}
        for symptom, info in SYMPTOM_MAP.items():
            if query in symptom or symptom in query:
                for disease, color in zip(info["diseases"], info["colors"]):
                    matched_diseases[disease] = color
        if matched_diseases:
            icons = {"Lung Cancer": "🫁", "Breast Cancer": "🎗️", "Liver Cancer": "🫀"}
            st.markdown(f'<div class="symptom-result"><div class="symptom-result-title">💡 "{symptom_search}" is associated with:</div>', unsafe_allow_html=True)
            chips = ""
            for disease, color in matched_diseases.items():
                chips += f'<span class="match-chip" style="background:{color}">{icons[disease]} {disease}</span>'
            st.markdown(chips + "</div>", unsafe_allow_html=True)
            st.markdown("**Go directly to prediction page:**")
            cols = st.columns(len(matched_diseases))
            for i, (disease, color) in enumerate(matched_diseases.items()):
                with cols[i]:
                    st.page_link(DISEASE_PAGES[disease], label=f"{icons[disease]} Open {disease}")
        else:
            st.markdown(f'<div class="no-match">❌ No match for "<strong>{symptom_search}</strong>". Try: cough, lump, jaundice, hepatitis, afp, fatigue, alcohol...</div>', unsafe_allow_html=True)
    else:
        examples = ["cough", "jaundice", "lump", "fatigue", "hepatitis", "bmi", "weight loss"]
        st.markdown(f'<div style="color:#AD1457; font-size:14px;">💡 Try searching: {", ".join(examples)}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-header">🦠 Choose a Disease to Predict</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="disease-card lung-card">
        <div class="card-icon">🫁</div>
        <div class="card-title lung-title">Lung Cancer</div>
        <div class="card-desc">Assess lung cancer risk level (Low / Medium / High) based on lifestyle, environmental exposure, and symptoms.</div>
        <div>
            <span class="card-tag" style="background:#E3F2FD; color:#0288D1">23 Features</span>
            <span class="card-tag" style="background:#E3F2FD; color:#0288D1">1000+ Records</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/lung_cancer.py", label="🫁 Open Lung Cancer Prediction →")

with col2:
    st.markdown("""
    <div class="disease-card breast-card">
        <div class="card-icon">🎗️</div>
        <div class="card-title breast-title">Breast Cancer</div>
        <div class="card-desc">Classify breast tumors as Benign or Malignant using 30 cell nucleus measurements from medical imaging.</div>
        <div>
            <span class="card-tag" style="background:#FCE4EC; color:#C2185B">30 Features</span>
            <span class="card-tag" style="background:#FCE4EC; color:#C2185B">569 Records</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/breast_cancer.py", label="🎗️ Open Breast Cancer Prediction →")

with col3:
    st.markdown("""
    <div class="disease-card liver-card">
        <div class="card-icon">🫀</div>
        <div class="card-title liver-title">Liver Cancer</div>
        <div class="card-desc">Predict liver cancer likelihood using patient demographics, lifestyle habits, lab values, and medical history.</div>
        <div>
            <span class="card-tag" style="background:#FFFDE7; color:#C68000">13 Features</span>
            <span class="card-tag" style="background:#FFFDE7; color:#C68000">5000+ Records</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/liver_cancer.py", label="🫀 Open Liver Cancer Prediction →")

st.markdown('<div class="section-header">⚙️ How It Works</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
steps = [
    ("1", "Choose Disease", "Select the type of cancer prediction from the cards above or sidebar navigation."),
    ("2", "Select Algorithm", "Pick from 5 ML algorithms — SVM, Decision Tree, KNN, Logistic Regression, or Naive Bayes."),
    ("3", "Enter Details", "Fill in patient health data using the interactive sliders and dropdowns."),
    ("4", "Get Prediction", "Click Predict to receive an instant risk assessment with confidence scores."),
]
for col, (num, title, desc) in zip([c1, c2, c3, c4], steps):
    with col:
        st.markdown(f"""
        <div class="how-card">
            <div class="how-num">{num}</div>
            <div class="how-title">{title}</div>
            <div class="how-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">

    <div style="font-size:36px; margin-bottom:12px;">🌸</div>

    <div style="font-size:20px; font-weight:800; color:#7A3055; margin-bottom:16px;">
        Your Health Is Your Greatest Wealth
    </div>

    <div style="font-size:15px; color:#9E5070; line-height:1.9; max-width:640px; margin:0 auto 24px auto;">
        Every small step you take today — whether it's getting a check-up, understanding your symptoms,
        or simply being aware — is a powerful act of self-love. You are not alone on this journey.
        <br><br>
        <em>"The greatest gift you can give your family and the world is a healthy you."</em>
    </div>

    <div style="border-top:1px solid #FFD6E0; padding-top:20px; margin-top:8px;">
        <div style="font-size:18px; font-weight:800; color:#5C1A3A; margin-bottom:6px;">
            🙏 Thank You for Using MediPredict
        </div>
        <div style="font-size:13px; color:#B06080; margin-bottom:16px;">
            We hope this tool brings you clarity, comfort, and confidence in your health journey.
        </div>

        <div style="font-size:13px; color:#9E5070; margin-bottom:4px;">
            💗 Created with care by <strong style="color:#7A3055;">Upasana Joshi</strong>
        </div>
        <div style="font-size:13px; color:#9E5070; margin-bottom:16px;">
            📧 <a href="mailto:upashana329@gmail.com" style="color:#C2185B; text-decoration:none; font-weight:600;">upashana329@gmail.com</a>
        </div>

        <div style="font-size:12px; color:#C0A0B0;">
            ⚕️ <em>For educational purposes only. Always consult a qualified healthcare professional for medical advice.</em>
            <br>Built with Streamlit · Powered by Scikit-learn
        </div>
    </div>

</div>
""", unsafe_allow_html=True)
