import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   class_names: list = None, average: str = "weighted") -> dict:
    """
    Evaluates a trained sklearn model on test data.

    Parameters
    ----------
    model       : fitted sklearn estimator
    X_test      : test features
    y_test      : true labels
    class_names : list of class label strings (optional)
    average     : averaging strategy for precision/recall/f1

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                    classification_report
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":               round(accuracy_score(y_test, y_pred), 4),
        "precision":              round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
        "recall":                 round(recall_score(y_test, y_pred, average=average, zero_division=0), 4),
        "f1":                     round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
        "confusion_matrix":       confusion_matrix(y_test, y_pred).tolist(),
        "classification_report":  classification_report(
                                      y_test, y_pred,
                                      target_names=class_names,
                                      zero_division=0
                                  ),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        metrics["avg_confidence"] = round(float(np.max(proba, axis=1).mean()), 4)

    return metrics


def compare_models(results: dict) -> dict:
    """
    Given a dict of {model_name: metrics_dict}, returns a summary
    sorted by accuracy descending.

    Example
    -------
    results = {
        "SVM":              {"accuracy": 0.96, ...},
        "Decision Tree":    {"accuracy": 0.93, ...},
    }
    compare_models(results)
    -> {"SVM": 0.96, "Decision Tree": 0.93, ...}
    """
    return dict(
        sorted(
            {name: m["accuracy"] for name, m in results.items()}.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )


def format_metrics(metrics: dict) -> str:
    """
    Returns a human-readable string summary of model metrics.
    Useful for logging or displaying in the sidebar.
    """
    lines = [
        f"Accuracy  : {metrics['accuracy']*100:.2f}%",
        f"Precision : {metrics['precision']*100:.2f}%",
        f"Recall    : {metrics['recall']*100:.2f}%",
        f"F1 Score  : {metrics['f1']*100:.2f}%",
    ]
    if "avg_confidence" in metrics:
        lines.append(f"Avg Conf  : {metrics['avg_confidence']*100:.2f}%")
    return "\n".join(lines)


def get_risk_label(proba: float, thresholds: dict = None) -> tuple[str, str]:
    """
    Converts a cancer probability (0–1) into a risk label and colour.

    Parameters
    ----------
    proba      : float — probability of cancer (class 1)
    thresholds : dict  — custom thresholds, default: low<0.35, medium<0.65, high>=0.65

    Returns
    -------
    (label: str, colour: str)  e.g. ("High Risk", "#E53935")
    """
    if thresholds is None:
        thresholds = {"low": 0.35, "medium": 0.65}

    if proba < thresholds["low"]:
        return "Low Risk", "#43A047"
    elif proba < thresholds["medium"]:
        return "Medium Risk", "#FB8C00"
    else:
        return "High Risk", "#E53935"
