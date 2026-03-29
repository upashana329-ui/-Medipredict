# config/settings.py
# Central configuration for the MediPredict application.
# All paths, model parameters, UI constants, and feature ranges live here.
# Import from this file instead of hardcoding values across the project.

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # root of the project

# ── Directory paths ───────────────────────────────────────────────────────────
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PAGES_DIR  = BASE_DIR / "pages"

# ── Dataset file paths ────────────────────────────────────────────────────────
DATASETS = {
    "lung":   str(DATA_DIR / "lung_cancer.csv"),
    "breast": str(DATA_DIR / "breast_cancer.csv"),
    "liver":  str(DATA_DIR / "liver_cancer.csv"),
}

# ── Trained model file paths ──────────────────────────────────────────────────
MODEL_PATHS = {
    "lung": {
        "Support Vector Machine (SVM)": str(MODELS_DIR / "lung_cancer_svm.pkl"),
        "Decision Tree":                str(MODELS_DIR / "lung_cancer_decision_tree.pkl"),
        "K-Nearest Neighbors (KNN)":    str(MODELS_DIR / "lung_cancer_knn.pkl"),
        "Logistic Regression":          str(MODELS_DIR / "lung_cancer_logistic_regression.pkl"),
        "Naive Bayes":                  str(MODELS_DIR / "lung_cancer_naive_bayes.pkl"),
    },
    "breast": {
        "Support Vector Machine (SVM)": str(MODELS_DIR / "breast_cancer_svm.pkl"),
        "Decision Tree":                str(MODELS_DIR / "breast_cancer_decision_tree.pkl"),
        "K-Nearest Neighbors (KNN)":    str(MODELS_DIR / "breast_cancer_knn.pkl"),
        "Logistic Regression":          str(MODELS_DIR / "breast_cancer_logistic_regression.pkl"),
        "Naive Bayes":                  str(MODELS_DIR / "breast_cancer_naive_bayes.pkl"),
    },
    "liver": {
        "Support Vector Machine (SVM)": str(MODELS_DIR / "liver_cancer_svm.pkl"),
        "Decision Tree":                str(MODELS_DIR / "liver_cancer_decision_tree.pkl"),
        "K-Nearest Neighbors (KNN)":    str(MODELS_DIR / "liver_cancer_knn.pkl"),
        "Logistic Regression":          str(MODELS_DIR / "liver_cancer_logistic_regression.pkl"),
        "Naive Bayes":                  str(MODELS_DIR / "liver_cancer_naive_bayes.pkl"),
    },
}

# ── Training parameters (used by train_*.py scripts) ─────────────────────────
TRAIN_CONFIG = {
    "test_size":    0.2,        # 80/20 train-test split
    "random_state": 42,         # reproducibility seed
    "cv_folds":     5,          # stratified k-fold cross-validation

    # Decision Tree
    "dt_max_depth":         10,
    "dt_min_samples_split": 5,
    "dt_min_samples_leaf":  2,
    "dt_criterion":         "gini",

    # KNN
    "knn_n_neighbors": 5,
    "knn_weights":     "uniform",
    "knn_metric":      "minkowski",

    # Logistic Regression
    "lr_C":        1.0,
    "lr_max_iter": 1000,
    "lr_solver":   "lbfgs",

    # Naive Bayes
    "nb_var_smoothing": 1e-9,

    # SVM
    "svm_kernel": "rbf",
    "svm_C":      1.0,
    "svm_gamma":  "scale",
}

# ── Risk thresholds (used by utils/model_evaluation.py) ──────────────────────
RISK_THRESHOLDS = {
    "low":    0.35,     # probability < 0.35  → Low Risk
    "medium": 0.65,     # probability < 0.65  → Medium Risk
                        # probability >= 0.65 → High Risk
}

RISK_COLORS = {
    "Low Risk":    "#43A047",
    "Medium Risk": "#FB8C00",
    "High Risk":   "#E53935",
}

# ── App metadata ──────────────────────────────────────────────────────────────
APP_CONFIG = {
    "title":       "MediPredict — Disease Prediction System",
    "icon":        "🏥",
    "layout":      "wide",
    "description": "AI-powered cancer risk assessment for Lung, Breast, and Liver cancer.",
    "version":     "1.0.0",
    "disclaimer":  (
        "This application is for educational purposes only. "
        "Always consult a qualified healthcare professional for medical advice."
    ),
}

# ── Disease display metadata (used by homepage cards) ────────────────────────
DISEASE_CONFIG = {
    "lung": {
        "label":       "Lung Cancer",
        "icon":        "🫁",
        "page":        "pages/lung_cancer.py",
        "color":       "#42A5F5",
        "features":    23,
        "records":     "1,000+",
        "output":      "Low / Medium / High risk",
        "train_script":"train_lung_cancer_models.py",
    },
    "breast": {
        "label":       "Breast Cancer",
        "icon":        "🎗️",
        "page":        "pages/breast_cancer.py",
        "color":       "#EC407A",
        "features":    30,
        "records":     "569",
        "output":      "Benign / Malignant",
        "train_script":"train_breast_cancer_models.py",
    },
    "liver": {
        "label":       "Liver Cancer",
        "icon":        "🫀",
        "page":        "pages/liver_cancer.py",
        "color":       "#FFCA28",
        "features":    13,
        "records":     "5,000+",
        "output":      "Cancer / No Cancer",
        "train_script":"train_liver_cancer_models.py",
    },
}

# ── Environment / secrets (read from .env if present) ────────────────────────
# Add any API keys or secrets to .env — never hardcode them here.
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
