import numpy as np
import pandas as pd
LIVER_ENCODINGS = {
    "gender":                  {"Female": 0, "Male": 1},
    "alcohol_consumption":     {"Never": 0, "Occasional": 1, "Regular": 2},
    "smoking_status":          {"Current": 0, "Former": 1, "Never": 2},
    "physical_activity_level": {"High": 0, "Low": 1, "Moderate": 2},
}

LUNG_ENCODINGS = {
    "gender": {"Female": 2, "Male": 1},
}

BREAST_ENCODINGS = {}  # All features are numeric — no encoding needed


# ── Main preprocessing function ───────────────────────────────────────────────

def preprocess_input(raw_values: dict, disease: str) -> np.ndarray:
    """
    Takes a dict of raw field values from the UI and returns a
    numpy array ready to pass into model.predict().

    Parameters
    ----------
    raw_values : dict
        Key-value pairs matching the feature names used during training.
    disease : str
        One of "liver", "lung", or "breast".

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    encodings = {
        "liver":  LIVER_ENCODINGS,
        "lung":   LUNG_ENCODINGS,
        "breast": BREAST_ENCODINGS,
    }.get(disease.lower(), {})

    encoded = encode_categoricals(raw_values, encodings)
    values = list(encoded.values())
    return np.array([values], dtype=float)


def encode_categoricals(raw: dict, encodings: dict) -> dict:
    """
    Replaces string values with their integer-encoded equivalents.

    Parameters
    ----------
    raw : dict       — raw field values
    encodings : dict — mapping of field_name -> {string: int}

    Returns
    -------
    dict with categorical fields replaced by integers
    """
    result = dict(raw)
    for field, mapping in encodings.items():
        if field in result:
            original = result[field]
            result[field] = mapping.get(original, 0)
    return result


def validate_input(raw_values: dict, required_fields: list) -> tuple[bool, list]:
    """
    Checks that all required fields are present and non-null.

    Returns
    -------
    (is_valid: bool, missing_fields: list)
    """
    missing = [f for f in required_fields if f not in raw_values or raw_values[f] is None]
    return len(missing) == 0, missing


def clip_outliers(value: float, min_val: float, max_val: float) -> float:
    """Clips a value to the expected feature range."""
    return float(np.clip(value, min_val, max_val))
