import os
import numpy as np
import joblib
_model_cache: dict = {}


def load_model(model_path: str) -> dict:
    """
    Loads a saved model bundle from disk (or returns cached version).

    The bundle is a dict saved by the train_*.py scripts containing:
        - "model"          : fitted sklearn estimator
        - "scaler"         : fitted StandardScaler (may be absent)
        - "imputer"        : fitted SimpleImputer  (may be absent)
        - "label_encoders" : dict of fitted LabelEncoders (may be absent)
        - "feature_names"  : list of feature name strings
        - "class_names"    : list of class label strings
        - "metrics"        : dict of training evaluation metrics

    Parameters
    ----------
    model_path : str — relative path to the .pkl file, e.g. "models/liver_cancer_svm.pkl"

    Returns
    -------
    dict (the saved bundle)

    Raises
    ------
    FileNotFoundError if the .pkl file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run the corresponding train_*.py script first."
        )

    if model_path not in _model_cache:
        _model_cache[model_path] = joblib.load(model_path)

    return _model_cache[model_path]


def predict(model_path: str, input_array: np.ndarray) -> int | str:
    """
    Loads the model bundle and returns the predicted class label.

    Parameters
    ----------
    model_path  : str          — path to the .pkl file
    input_array : np.ndarray   — shape (1, n_features), raw numeric values

    Returns
    -------
    Predicted label (int for liver/breast binary, str for lung Low/Medium/High)
    """
    bundle = load_model(model_path)
    X = _apply_transforms(bundle, input_array)
    return bundle["model"].predict(X)[0]


def predict_proba(model_path: str, input_array: np.ndarray) -> np.ndarray | None:
    """
    Returns class probabilities if the model supports predict_proba,
    otherwise returns None.

    Parameters
    ----------
    model_path  : str
    input_array : np.ndarray — shape (1, n_features)

    Returns
    -------
    np.ndarray of shape (1, n_classes) or None
    """
    bundle = load_model(model_path)
    model = bundle["model"]

    if not hasattr(model, "predict_proba"):
        return None

    X = _apply_transforms(bundle, input_array)
    return model.predict_proba(X)[0]


def get_model_metrics(model_path: str) -> dict:
    """
    Returns the training metrics stored inside the model bundle.
    Useful for displaying accuracy/F1 in the sidebar.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, cv_mean, cv_std, etc.
    """
    bundle = load_model(model_path)
    return bundle.get("metrics", {})


def get_class_names(model_path: str) -> list:
    """Returns the list of class name strings from the model bundle."""
    bundle = load_model(model_path)
    return bundle.get("class_names", [])


# ── Internal helper ───────────────────────────────────────────────────────────

def _apply_transforms(bundle: dict, X: np.ndarray) -> np.ndarray:
    """
    Applies imputer and scaler transforms stored in the bundle,
    in the same order they were applied during training.
    """
    X = X.copy().astype(float)

    if "imputer" in bundle and bundle["imputer"] is not None:
        X = bundle["imputer"].transform(X)

    if "scaler" in bundle and bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)

    return X
