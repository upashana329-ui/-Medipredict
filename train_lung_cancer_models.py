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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs("models", exist_ok=True)
print("=" * 60)
print("   LUNG CANCER PREDICTION - MODEL TRAINING SYSTEM")
print("=" * 60)

data = pd.read_csv("data/lung_cancer.csv")
data = data.drop(columns=["index", "Patient Id"], errors="ignore")
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
data = data.dropna(axis=1, how="all")
print(f"\nRecords: {len(data)} | Columns: {len(data.columns)}")

le = LabelEncoder()
data["Level"] = le.fit_transform(data["Level"])
class_names = list(le.classes_)
X = data.drop(columns=["Level"])
y = data["Level"]
feature_names = list(X.columns)

imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    metrics = {
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "classification_report": classification_report(y_te, y_pred, target_names=class_names),
    }
    print(f"\n--- {name} ---")
    print(f"Accuracy: {metrics['accuracy']} | Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1: {metrics['f1_score']}")
    print(f"CV Score: {metrics['cv_mean']} (+/- {metrics['cv_std']})")
    print(metrics["classification_report"])
    return model, metrics

def save_model(filename, model, metrics):
    joblib.dump({"model": model, "scaler": scaler, "imputer": imputer,
                 "label_encoder": le, "feature_names": feature_names,
                 "class_names": class_names, "metrics": metrics}, f"models/{filename}")
    print(f"Saved -> models/{filename}")

print("\n[1] Decision Tree")
dt, dt_m = evaluate("Decision Tree", DecisionTreeClassifier(max_depth=10, random_state=42), X_train, X_test, y_train, y_test)
save_model("lung_cancer_decision_tree.pkl", dt, dt_m)

print("\n[2] KNN")
knn, knn_m = evaluate("KNN", KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled, y_train, y_test)
save_model("lung_cancer_knn.pkl", knn, knn_m)

print("\n[3] Logistic Regression")
lr, lr_m = evaluate("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled, y_train, y_test)
save_model("lung_cancer_logistic_regression.pkl", lr, lr_m)

print("\n[4] Naive Bayes")
nb, nb_m = evaluate("Naive Bayes", GaussianNB(), X_train_scaled, X_test_scaled, y_train, y_test)
save_model("lung_cancer_naive_bayes.pkl", nb, nb_m)

print("\n[5] SVM")
svm, svm_m = evaluate("SVM", SVC(kernel="rbf", probability=True, random_state=42), X_train_scaled, X_test_scaled, y_train, y_test)
save_model("lung_cancer_svm.pkl", svm, svm_m)

results = {"Decision Tree": dt_m["accuracy"], "KNN": knn_m["accuracy"],
           "Logistic Regression": lr_m["accuracy"], "Naive Bayes": nb_m["accuracy"], "SVM": svm_m["accuracy"]}
print("\n" + "=" * 60)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<25}: {acc*100:.2f}%")
print(f"\nBest model: {max(results, key=results.get)} | All 5 files saved in models/")