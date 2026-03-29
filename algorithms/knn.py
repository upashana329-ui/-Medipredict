
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import joblib
import os


def train_knn(
    X_train, X_test, y_train, y_test,
    class_names: list,
    feature_names: list,
    scaler=None,
    imputer=None,
    label_encoder=None,
    label_encoders: dict = None,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    random_state: int = 42,
    cv_folds: int = 5,
    hyperparameter_tuning: bool = False,
    verbose: bool = True,
) -> tuple[KNeighborsClassifier, dict]:
    """
    Train a KNN classifier and return the fitted model + metrics dict.

    KNN is distance-based — always pass SCALED feature splits.

    Parameters
    ----------
    X_train, X_test     : array-like — scaled train/test feature splits
    y_train, y_test     : array-like — train/test label splits
    class_names         : list of str — human-readable class labels
    feature_names       : list of str — feature column names
    scaler              : fitted StandardScaler (stored in bundle for prediction)
    imputer             : fitted SimpleImputer or None
    label_encoder       : fitted LabelEncoder for the target (lung/breast)
    label_encoders      : dict of fitted LabelEncoders for input cols (liver)
    n_neighbors         : int — number of neighbours (k)
    weights             : "uniform" or "distance"
    metric              : distance metric, default "minkowski" (= euclidean for p=2)
    random_state        : int — used only for GridSearchCV reproducibility
    cv_folds            : int — number of cross-validation folds
    hyperparameter_tuning : bool — run GridSearchCV to find best k
    verbose             : bool — print progress

    Returns
    -------
    model   : fitted KNeighborsClassifier
    metrics : dict with accuracy, precision, recall, f1, cv_mean, cv_std,
              confusion_matrix, classification_report
    """
    if verbose:
        print("\n── K-Nearest Neighbors ────────────────────────────────────")

    # ── Optional hyperparameter tuning ────────────────────────────────────────
    if hyperparameter_tuning:
        if verbose:
            print("  Running GridSearchCV (this may take a moment)...")

        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights":     ["uniform", "distance"],
            "metric":      ["minkowski", "manhattan"],
        }
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            KNeighborsClassifier(),
            param_grid, cv=skf, scoring="f1_weighted", n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_

        if verbose:
            print(f"  Best params: {gs.best_params_}")
            print(f"  Best CV F1 : {gs.best_score_:.4f}")
    else:
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
        )
        model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

    metrics = {
        "accuracy":              round(accuracy_score(y_test, y_pred), 4),
        "precision":             round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":                round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score":              round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "cv_mean":               round(cv_scores.mean(), 4),
        "cv_std":                round(cv_scores.std(), 4),
        "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
                                     y_test, y_pred,
                                     target_names=class_names,
                                     zero_division=0,
                                 ),
        "n_neighbors":           model.n_neighbors,
    }

    if verbose:
        print(f"  k (neighbors) : {model.n_neighbors}")
        print(f"  Accuracy      : {metrics['accuracy']}")
        print(f"  Precision     : {metrics['precision']}")
        print(f"  Recall        : {metrics['recall']}")
        print(f"  F1 Score      : {metrics['f1_score']}")
        print(f"  CV Score      : {metrics['cv_mean']} (+/- {metrics['cv_std']})")
        print(f"\n  Classification Report:")
        print(metrics["classification_report"])

    return model, metrics


def save_bundle(
    filepath: str,
    model: KNeighborsClassifier,
    metrics: dict,
    scaler=None,
    imputer=None,
    label_encoder=None,
    label_encoders: dict = None,
    feature_names: list = None,
    class_names: list = None,
) -> None:
    """
    Saves the model bundle as a dict — same format expected by the pages
    and utils/predictions.py.

    Parameters
    ----------
    filepath : str — e.g. "models/lung_cancer_knn.pkl"
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    bundle = {
        "model":          model,
        "scaler":         scaler,
        "imputer":        imputer,
        "label_encoder":  label_encoder,
        "label_encoders": label_encoders,
        "feature_names":  feature_names,
        "class_names":    class_names,
        "metrics":        metrics,
    }
    joblib.dump(bundle, filepath)
    print(f"  Saved → {filepath}")
