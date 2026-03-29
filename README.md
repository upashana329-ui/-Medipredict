# 🏥 MediPredict — AI-Powered Cancer Risk Assessment

MediPredict is a Streamlit web application that predicts the risk of three types of cancer — **Lung**, **Breast**, and **Liver** — using five machine learning algorithms trained on real medical datasets.

> ⚕️ **Disclaimer:** This application is for educational purposes only. Always consult a qualified healthcare professional for medical advice.

---

## Features

- Predict cancer risk for Lung, Breast, and Liver cancer
- Choose from 5 ML algorithms: SVM, Decision Tree, KNN, Logistic Regression, Naive Bayes
- Visual progress bars showing where each input value sits in its range
- Confidence scores with probability bar charts after each prediction
- Symptom-based and disease-name search on the homepage
- Clean baby-pink and white UI designed for ease of use

---

## Project Structure

```
medipredict/
│
├── app.py                          # Homepage — search, disease cards, how it works
│
├── pages/
│   ├── lung_cancer.py              # Lung cancer prediction page
│   ├── breast_cancer.py            # Breast cancer prediction page
│   └── liver_cancer.py             # Liver cancer prediction page
│
├── algorithms/
│   ├── __init__.py                 # Exports all train_* functions
│   ├── decision_tree.py            # Decision Tree wrapper
│   ├── knn.py                      # KNN wrapper
│   ├── logistic_regression.py      # Logistic Regression wrapper
│   ├── naive_bayes.py              # Naive Bayes wrapper
│   └── svm.py                      # SVM wrapper
│
├── utils/
│   ├── __init__.py                 # Exports key utility functions
│   ├── data_preprocessing.py       # Input encoding and validation
│   ├── feature_extraction.py       # Feature metadata for all diseases
│   ├── model_evaluation.py         # Metrics, risk labels, model comparison
│   └── predictions.py              # Model loading and prediction pipeline
│
├── config/
│   ├── __init__.py                 # Exports all settings
│   └── settings.py                 # Paths, hyperparameters, app constants
│
├── data/
│   ├── lung_cancer.csv
│   ├── breast_cancer.csv
│   └── liver_cancer.csv
│
├── models/                         # Saved .pkl model bundles (auto-created)
│
├── train_lung_cancer_models.py     # Train all 5 lung cancer models
├── train_breast_cancer_models.py   # Train all 5 breast cancer models
├── train_liver_cancer_models.py    # Train all 5 liver cancer models
│
├── .streamlit/config.toml          # Streamlit theme configuration
├── .env                            # Local environment variables (gitignored)
├── .env.example                    # Template for .env
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medipredict.git
cd medipredict
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` if needed (defaults work fine for local development).

### 5. Train the models

Run each training script once to generate the `.pkl` files in the `models/` folder:

```bash
python train_lung_cancer_models.py
python train_breast_cancer_models.py
python train_liver_cancer_models.py
```

### 6. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Machine Learning Algorithms

| Algorithm | Needs Scaled Data | Best For |
|---|---|---|
| Decision Tree | No | Interpretability, feature importance |
| K-Nearest Neighbors | Yes | Small datasets, non-linear boundaries |
| Logistic Regression | Yes | Fast, interpretable baseline |
| Naive Bayes | No | Quick training, probabilistic output |
| SVM (RBF kernel) | Yes | High accuracy on medical data |

---

## Datasets

| Disease | Records | Features | Output |
|---|---|---|---|
| Lung Cancer | 1,000+ | 23 | Low / Medium / High risk |
| Breast Cancer | 569 | 30 | Benign / Malignant |
| Liver Cancer | 5,000+ | 13 | Cancer / No Cancer |

---

## Tech Stack

- [Streamlit](https://streamlit.io) — web framework
- [scikit-learn](https://scikit-learn.org) — machine learning
- [pandas](https://pandas.pydata.org) — data handling
- [NumPy](https://numpy.org) — numerical computing
- [joblib](https://joblib.readthedocs.io) — model serialisation
