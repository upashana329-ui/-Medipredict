import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import joblib
import os


def train_decision_tree(
    X_train, X_test, y_train, y_test,
    class_names: list,
    feature_names: list,
    scaler=None,
    imputer=None,
    label_encoder=None,
    label_encoders: dict = None,
    max_depth: int = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    criterion: str = "gini",
    random_state: int = 42,
    cv_folds: int = 5,
    hyperparameter_tuning: bool = False,
    verbose: bool = True,
) -> tuple[DecisionTreeClassifier, dict]:
    """
    Train a Decision Tree classifier and return the fitted model + metrics dict.

    Decision Trees do NOT need scaled data — pass the raw (unscaled) splits.
    The scaler/imputer/encoders are stored in the bundle for prediction time.

    Parameters
    ----------
    X_train, X_test     : array-like — train/test feature splits
    y_train, y_test     : array-like — train/test label splits
    class_names         : list of str — human-readable class labels
    feature_names       : list of str — feature column names
    scaler              : fitted StandardScaler or None
    imputer             : fitted SimpleImputer or None
    label_encoder       : fitted LabelEncoder for the target (lung/breast)
    label_encoders      : dict of fitted LabelEncoders for input cols (liver)
    max_depth           : int — max tree depth
    min_samples_split   : int — min samples to split a node
    min_samples_leaf    : int — min samples in a leaf
    criterion           : "gini" or "entropy"
    random_state        : int
    cv_folds            : int — number of cross-validation folds
    hyperparameter_tuning : bool — run GridSearchCV to find best params
    verbose             : bool — print progress

    Returns
    -------
    model   : fitted DecisionTreeClassifier
    metrics : dict with accuracy, precision, recall, f1, cv_mean, cv_std,
              confusion_matrix, classification_report, feature_importances
    """
    if verbose:
        print("\n── Decision Tree ──────────────────────────────────────────")

    # ── Optional hyperparameter tuning ────────────────────────────────────────
    if hyperparameter_tuning:
        if verbose:
            print("  Running GridSearchCV (this may take a moment)...")

        param_grid = {
            "max_depth":        [5, 10, 15, 20],
            "min_samples_split":[2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion":        ["gini", "entropy"],
        }
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            DecisionTreeClassifier(random_state=random_state),
            param_grid, cv=skf, scoring="f1_weighted", n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_

        if verbose:
            print(f"  Best params: {gs.best_params_}")
            print(f"  Best CV F1 : {gs.best_score_:.4f}")
    else:
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

    metrics = {
        "accuracy":               round(accuracy_score(y_test, y_pred), 4),
        "precision":              round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":                 round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score":               round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "cv_mean":                round(cv_scores.mean(), 4),
        "cv_std":                 round(cv_scores.std(), 4),
        "confusion_matrix":       confusion_matrix(y_test, y_pred).tolist(),
        "classification_report":  classification_report(
                                      y_test, y_pred,
                                      target_names=class_names,
                                      zero_division=0,
                                  ),
        "feature_importances":    dict(zip(feature_names, model.feature_importances_.tolist())),
    }

    if verbose:
        print(f"  Accuracy  : {metrics['accuracy']}")
        print(f"  Precision : {metrics['precision']}")
        print(f"  Recall    : {metrics['recall']}")
        print(f"  F1 Score  : {metrics['f1_score']}")
        print(f"  CV Score  : {metrics['cv_mean']} (+/- {metrics['cv_std']})")
        print(f"\n  Classification Report:")
        print(metrics["classification_report"])

        # Top 5 features
        top5 = sorted(metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True)[:5]
        print("  Top 5 Features:")
        for feat, score in top5:
            print(f"    {feat}: {score:.4f}")

    return model, metrics


def save_bundle(
    filepath: str,
    model: DecisionTreeClassifier,
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
    filepath : str — e.g. "models/lung_cancer_decision_tree.pkl"
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
