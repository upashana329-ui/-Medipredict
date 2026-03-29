# config/__init__.py
# Exposes all settings for clean single-line imports anywhere in the project.
#
# Usage:
#   from config import MODEL_PATHS, TRAIN_CONFIG, APP_CONFIG
#   from config import DISEASE_CONFIG, RISK_THRESHOLDS, DATASETS

from config.settings import (
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
    DATASETS,
    MODEL_PATHS,
    TRAIN_CONFIG,
    RISK_THRESHOLDS,
    RISK_COLORS,
    APP_CONFIG,
    DISEASE_CONFIG,
    DEBUG_MODE,
    LOG_LEVEL,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "DATASETS",
    "MODEL_PATHS",
    "TRAIN_CONFIG",
    "RISK_THRESHOLDS",
    "RISK_COLORS",
    "APP_CONFIG",
    "DISEASE_CONFIG",
    "DEBUG_MODE",
    "LOG_LEVEL",
]
