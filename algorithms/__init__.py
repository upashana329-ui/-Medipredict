# algorithms/__init__.py
# Exposes all five algorithm train functions for clean imports.
#
# Usage from a train script:
#   from algorithms import train_decision_tree, train_knn, train_svm
#   from algorithms import train_logistic_regression, train_naive_bayes
#
# Each function returns (model, metrics) and expects the same signature.
# Use save_bundle() from the individual module to persist the .pkl file.

from algorithms.decision_tree       import train_decision_tree
from algorithms.knn                 import train_knn
from algorithms.logistic_regression import train_logistic_regression
from algorithms.naive_bayes         import train_naive_bayes
from algorithms.svm                 import train_svm

# Scaling requirements — useful reference when writing train scripts:
#
#   Algorithm             Needs scaled data?
#   ─────────────────     ──────────────────
#   Decision Tree         NO  — tree splits are scale-invariant
#   KNN                   YES — distance-based, scale matters a lot
#   Logistic Regression   YES — gradient-based solver converges faster
#   Naive Bayes           NO  — but scaled data doesn't hurt
#   SVM                   YES — kernel distances are scale-sensitive

__all__ = [
    "train_decision_tree",
    "train_knn",
    "train_logistic_regression",
    "train_naive_bayes",
    "train_svm",
]
