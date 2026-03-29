# algorithms/svm.py
# Reusable Support Vector Machine training wrapper for all three cancer datasets.
# Produces a model bundle dict compatible with the pages and utils/predictions.py.
#
# NOTE: SVM is very sensitive to feature scale — always pass SCALED splits.
# Use probability=True so predict_proba() works in the pages.

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import joblib
import os


def train_svm(
    X_train, X_test, y_train, y_test,
    class_names: list,
    feature_names: list,
    scaler=None,
    imputer=None,
    label_encoder=None,
    label_encoders: dict = None,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    random_state: int = 42,
    cv_folds: int = 5,
    hyperparameter_tuning: bool = False,
    verbose: bool = True,
):
    """
    Train an SVM classifier and return the fitted model + metrics dict.

    SVM is the most powerful algorithm in this project but also the slowest
    to train. Always pass SCALED feature splits — unscaled data will give
    poor results.

    Parameters
    ----------
    X_train, X_test  : array-like  scaled train/test feature splits
    y_train, y_test  : array-like  train/test label splits
    class_names      : list of str  human-readable class labels
    feature_names    : list of str  feature column names
    scaler           : fitted StandardScaler stored in bundle for prediction time
    imputer          : fitted SimpleImputer or None
    label_encoder    : fitted LabelEncoder for the target (lung / breast)
    label_encoders   : dict of fitted LabelEncoders for categorical input cols (liver)
    kernel           : "rbf" (default), "linear", "poly", or "sigmoid"
                       rbf works best for most medical datasets
    C                : float  regularisation — higher C = less regularisation
    gamma            : "scale" (default), "auto", or a float
                       controls influence radius of each training sample
    random_state     : int
    cv_folds         : int  number of cross-validation folds
    hyperparameter_tuning : bool  run GridSearchCV to find best C and gamma
                            WARNING: slow on large datasets
    verbose          : bool  print progress to console

    Returns
    -------
    model   : fitted SVC (with probability=True)
    metrics : dict  accuracy, precision, recall, f1, cv_mean, cv_std,
                    confusion_matrix, classification_report
    """
    if verbose:
        print("\n-- Support Vector Machine -----------------------------------------")

    # ── Optional hyperparameter tuning ────────────────────────────────────────
    if hyperparameter_tuning:
        if verbose:
            print("  Running GridSearchCV — this can be slow on large datasets...")

        param_grid = {
            "C":     [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.001, 0.01],
            "kernel":["rbf", "linear"],
        }
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            SVC(probability=True, random_state=random_state),
            param_grid, cv=skf, scoring="f1_weighted", n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_

        if verbose:
            print(f"  Best params : {gs.best_params_}")
            print(f"  Best CV F1  : {gs.best_score_:.4f}")
    else:
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,       # required for predict_proba in the pages
            random_state=random_state,
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
        "kernel":                model.kernel,
        "n_support_vectors":     int(model.n_support_.sum()),
    }

    if verbose:
        print(f"  Kernel            : {model.kernel}")
        print(f"  C                 : {model.C}")
        print(f"  Gamma             : {model.gamma}")
        print(f"  Support Vectors   : {metrics['n_support_vectors']}")
        print(f"  Accuracy          : {metrics['accuracy']}")
        print(f"  Precision         : {metrics['precision']}")
        print(f"  Recall            : {metrics['recall']}")
        print(f"  F1 Score          : {metrics['f1_score']}")
        print(f"  CV Score          : {metrics['cv_mean']} (+/- {metrics['cv_std']})")
        print(metrics["classification_report"])

    return model, metrics


def save_bundle(
    filepath: str,
    model: SVC,
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
    filepath : str  e.g. "models/lung_cancer_svm.pkl"
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
    print(f"  Saved -> {filepath}")
