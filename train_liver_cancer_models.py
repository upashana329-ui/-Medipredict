import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

os.makedirs("models", exist_ok=True)

print("=" * 60)
print("   LIVER CANCER PREDICTION - MODEL TRAINING SYSTEM")
print("=" * 60)

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------
print("\n[1] Loading dataset...")
data = pd.read_csv("data/liver_cancer.csv")
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
data = data.dropna(axis=1, how="all")

print(f"    Records         : {len(data)}")
print(f"    Columns         : {len(data.columns)}")
print(f"    Missing values  : {data.isnull().sum().sum()}")

# ----------------------------------------------------------
# 2. ENCODE CATEGORICAL COLUMNS
# ----------------------------------------------------------
print("\n[2] Encoding categorical columns...")

categorical_cols = ["gender", "alcohol_consumption", "smoking_status", "physical_activity_level"]
label_encoders = {}

for col in categorical_cols:
    if col in data.columns:
        le_col = LabelEncoder()
        data[col] = le_col.fit_transform(data[col].astype(str))
        label_encoders[col] = le_col
        print(f"    {col}: {list(le_col.classes_)}")

# ----------------------------------------------------------
# 3. PREPARE FEATURES & TARGET
# ----------------------------------------------------------
target_col = "liver_cancer"
X = data.drop(columns=[target_col])
y = data[target_col]
feature_names = list(X.columns)
class_names = ["No Cancer", "Cancer"]

print(f"\n    Features        : {len(feature_names)}")
print(f"    Class distribution:")
print(f"      No Cancer (0) : {(y == 0).sum()}")
print(f"      Cancer    (1) : {(y == 1).sum()}")

# ----------------------------------------------------------
# 4. HANDLE MISSING VALUES & SCALE
# ----------------------------------------------------------
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n    Training samples: {len(X_train)}")
print(f"    Testing samples : {len(X_test)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred    = model.predict(X_te)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")

    metrics = {
        "accuracy"  : round(accuracy_score(y_te, y_pred), 4),
        "precision" : round(precision_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "recall"    : round(recall_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "f1_score"  : round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "cv_mean"   : round(cv_scores.mean(), 4),
        "cv_std"    : round(cv_scores.std(), 4),
        "confusion_matrix"      : confusion_matrix(y_te, y_pred).tolist(),
        "classification_report" : classification_report(y_te, y_pred, target_names=class_names),
    }

    print(f"\n  --- {name} ---")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1 Score  : {metrics['f1_score']}")
    print(f"  CV Score  : {metrics['cv_mean']} (+/- {metrics['cv_std']})")
    print(f"\n  Classification Report:")
    print(metrics["classification_report"])

    return model, metrics


def save_model(filename, model, metrics):
    joblib.dump({
        "model"            : model,
        "scaler"           : scaler,
        "imputer"          : imputer,
        "label_encoders"   : label_encoders,
        "feature_names"    : feature_names,
        "class_names"      : class_names,
        "metrics"          : metrics,
    }, f"models/{filename}")
    print(f"  Saved -> models/{filename}")


# ----------------------------------------------------------
# 5. TRAIN ALL 5 MODELS
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("[3] Training Decision Tree Classifier")
print("=" * 60)
dt, dt_m = evaluate(
    "Decision Tree",
    DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
    X_train, X_test, y_train, y_test
)
top5 = sorted(zip(feature_names, dt.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
print("  Top 5 Features:")
for feat, score in top5:
    print(f"    {feat}: {score:.4f}")
save_model("liver_cancer_decision_tree.pkl", dt, dt_m)

print("\n" + "=" * 60)
print("[4] Training KNN Classifier")
print("=" * 60)
knn, knn_m = evaluate(
    "KNN",
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled, X_test_scaled, y_train, y_test
)
save_model("liver_cancer_knn.pkl", knn, knn_m)

print("\n" + "=" * 60)
print("[5] Training Logistic Regression Classifier")
print("=" * 60)
lr, lr_m = evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_scaled, X_test_scaled, y_train, y_test
)
save_model("liver_cancer_logistic_regression.pkl", lr, lr_m)

print("\n" + "=" * 60)
print("[6] Training Naive Bayes Classifier")
print("=" * 60)
nb, nb_m = evaluate(
    "Naive Bayes",
    GaussianNB(),
    X_train_scaled, X_test_scaled, y_train, y_test
)
save_model("liver_cancer_naive_bayes.pkl", nb, nb_m)

print("\n" + "=" * 60)
print("[7] Training SVM Classifier")
print("=" * 60)
svm, svm_m = evaluate(
    "SVM",
    SVC(kernel="rbf", probability=True, random_state=42),
    X_train_scaled, X_test_scaled, y_train, y_test
)
save_model("liver_cancer_svm.pkl", svm, svm_m)

# ----------------------------------------------------------
# 6. SUMMARY
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("   TRAINING COMPLETE — MODEL ACCURACY SUMMARY")
print("=" * 60)
results = {
    "Decision Tree"       : dt_m["accuracy"],
    "KNN"                 : knn_m["accuracy"],
    "Logistic Regression" : lr_m["accuracy"],
    "Naive Bayes"         : nb_m["accuracy"],
    "SVM"                 : svm_m["accuracy"],
}
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<25}: {acc:.4f} ({acc*100:.2f}%)")

print(f"\n  Best model: {max(results, key=results.get)} ({max(results.values())*100:.2f}%)")
print("\n  Files saved in models/ folder:")
print("    - liver_cancer_decision_tree.pkl")
print("    - liver_cancer_knn.pkl")
print("    - liver_cancer_logistic_regression.pkl")
print("    - liver_cancer_naive_bayes.pkl")
print("    - liver_cancer_svm.pkl")
print("\n  Each file contains: model + scaler + imputer + encoders + metrics")
