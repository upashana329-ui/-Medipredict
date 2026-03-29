# algorithms/logistic_regression.py
# Reusable Logistic Regression training wrapper for all three cancer datasets.
# Produces a model bundle dict compatible with the pages and utils/predictions.py.
#
# NOTE: Logistic Regression requires scaled data. Always pass scaled splits.

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import joblib
import os


def train_logistic_regression(
    X_train, X_test, y_train, y_test,
    class_names: list,
    feature_names: list,
    scaler=None,
    imputer=None,
    label_encoder=None,
    label_encoders: dict = None,
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    multi_class: str = "auto",
    random_state: int = 42,
    cv_folds: int = 5,
    hyperparameter_tuning: bool = False,
    verbose: bool = True,
) -> tuple[LogisticRegression, dict]:
    """
    Train a Logistic Regression classifier and return the fitted model + metrics dict.

    Logistic Regression is sensitive to feature scale — always pass SCALED splits.

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
    C                   : float — inverse of regularisation strength (smaller = stronger)
    max_iter            : int — max solver iterations
    solver              : solver algorithm ("lbfgs", "saga", "liblinear")
    multi_class         : "auto", "ovr", or "multinomial"
    random_state        : int
    cv_folds            : int — number of cross-validation folds
    hyperparameter_tuning : bool — run GridSearchCV to find best C
    verbose             : bool — print progress

    Returns
    -------
    model   : fitted LogisticRegression
    metrics : dict with accuracy, precision, recall, f1, cv_mean, cv_std,
              confusion_matrix, classification_report, top_coefficients
    """
    if verbose:
        print("\n── Logistic Regression ────────────────────────────────────")

    # ── Optional hyperparameter tuning ────────────────────────────────────────
    if hyperparameter_tuning:
        if verbose:
            print("  Running GridSearchCV (this may take a moment)...")

        param_grid = {
            "C":       [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver":  ["lbfgs", "saga"],
            "max_iter":[500, 1000, 2000],
        }
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            LogisticRegression(random_state=random_state, multi_class=multi_class),
            param_grid, cv=skf, scoring="f1_weighted", n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_

        if verbose:
            print(f"  Best params: {gs.best_params_}")
            print(f"  Best CV F1 : {gs.best_score_:.4f}")
    else:
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            multi_class=multi_class,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

    # Top feature coefficients (binary: one row; multiclass: mean abs across classes)
    coef = model.coef_
    if coef.shape[0] == 1:
        importance = np.abs(coef[0])
    else:
        importance = np.mean(np.abs(coef), axis=0)

    top_coef = dict(
        sorted(zip(feature_names, importance.tolist()), key=lambda x: x[1], reverse=True)[:10]
    )

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
        "top_coefficients":      top_coef,
    }

    if verbose:
        print(f"  C (regularisation) : {model.C}")
        print(f"  Solver             : {model.solver}")
        print(f"  Accuracy           : {metrics['accuracy']}")
        print(f"  Precision          : {metrics['precision']}")
        print(f"  Recall             : {metrics['recall']}")
        print(f"  F1 Score           : {metrics['f1_score']}")
        print(f"  CV Score           : {metrics['cv_mean']} (+/- {metrics['cv_std']})")
        print(f"\n  Classification Report:")
        print(metrics["classification_report"])
        print("  Top 5 Features by Coefficient Magnitude:")
        for feat, val in list(top_coef.items())[:5]:
            print(f"    {feat}: {val:.4f}")

    return model, metrics


def save_bundle(
    filepath: str,
    model: LogisticRegression,
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
    filepath : str — e.g. "models/lung_cancer_logistic_regression.pkl"
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
