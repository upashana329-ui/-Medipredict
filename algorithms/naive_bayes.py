# algorithms/naive_bayes.py
# Reusable Gaussian Naive Bayes training wrapper for all three cancer datasets.
# Produces a model bundle dict compatible with the pages and utils/predictions.py.
#
# Naive Bayes assumes features are conditionally independent and Gaussian-distributed.
# It works on raw or scaled data — both are fine.

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import joblib
import os


def train_naive_bayes(
    X_train, X_test, y_train, y_test,
    class_names: list,
    feature_names: list,
    scaler=None,
    imputer=None,
    label_encoder=None,
    label_encoders: dict = None,
    var_smoothing: float = 1e-9,
    random_state: int = 42,
    cv_folds: int = 5,
    verbose: bool = True,
):
    """
    Train a Gaussian Naive Bayes classifier and return the fitted model + metrics dict.

    Naive Bayes has no meaningful hyperparameters to tune via GridSearchCV.
    var_smoothing (default 1e-9) handles numerical stability and rarely needs changing.

    Parameters
    ----------
    X_train, X_test  : array-like  train/test feature splits
    y_train, y_test  : array-like  train/test label splits
    class_names      : list of str  human-readable class labels
    feature_names    : list of str  feature column names
    scaler           : fitted StandardScaler stored in bundle for prediction time
    imputer          : fitted SimpleImputer or None
    label_encoder    : fitted LabelEncoder for the target (lung / breast)
    label_encoders   : dict of fitted LabelEncoders for categorical input cols (liver)
    var_smoothing    : float  portion of largest variance added for numerical stability
    random_state     : int  used for StratifiedKFold reproducibility
    cv_folds         : int  number of cross-validation folds
    verbose          : bool  print progress to console

    Returns
    -------
    model   : fitted GaussianNB
    metrics : dict  accuracy, precision, recall, f1, cv_mean, cv_std,
                    confusion_matrix, classification_report, class_priors
    """
    if verbose:
        print("\n-- Gaussian Naive Bayes -------------------------------------------")

    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")

    # Class prior probabilities learned from training data
    class_priors = {
        class_names[i]: round(float(np.exp(model.class_log_prior_[i])), 4)
        for i in range(len(model.class_log_prior_))
    }

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
        "class_priors":          class_priors,
    }

    if verbose:
        print(f"  var_smoothing : {var_smoothing}")
        print(f"  Accuracy      : {metrics['accuracy']}")
        print(f"  Precision     : {metrics['precision']}")
        print(f"  Recall        : {metrics['recall']}")
        print(f"  F1 Score      : {metrics['f1_score']}")
        print(f"  CV Score      : {metrics['cv_mean']} (+/- {metrics['cv_std']})")
        print(metrics["classification_report"])
        print("  Class Prior Probabilities (learned from training data):")
        for cls, prior in class_priors.items():
            print(f"    {cls}: {prior:.4f}  ({prior*100:.1f}%)")

    return model, metrics



def save_bundle(
    filepath: str,
    model: GaussianNB,
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
    filepath : str  e.g. "models/lung_cancer_naive_bayes.pkl"
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
