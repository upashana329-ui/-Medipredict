# utils/feature_extraction.py
# Stores feature metadata for each disease — names, ranges, descriptions,
# and groupings. Used by the UI pages to render labels and tooltips.

# ── Liver Cancer Features ─────────────────────────────────────────────────────

LIVER_FEATURES = {
    "age":                     {"range": (18, 90),    "type": "numeric",      "desc": "Patient age in years"},
    "gender":                  {"range": None,        "type": "categorical",  "desc": "Biological sex (Male / Female)"},
    "bmi":                     {"range": (10.0, 45.0),"type": "numeric",      "desc": "Body Mass Index"},
    "alcohol_consumption":     {"range": None,        "type": "categorical",  "desc": "Never / Occasional / Regular"},
    "smoking_status":          {"range": None,        "type": "categorical",  "desc": "Never / Former / Current"},
    "hepatitis_b":             {"range": (0, 1),      "type": "binary",       "desc": "Hepatitis B diagnosis (Yes/No)"},
    "hepatitis_c":             {"range": (0, 1),      "type": "binary",       "desc": "Hepatitis C diagnosis (Yes/No)"},
    "liver_function_score":    {"range": (10.0, 120.0),"type": "numeric",     "desc": "Liver function test score"},
    "afp_level":               {"range": (0.0, 120.0),"type": "numeric",      "desc": "Alpha-Fetoprotein level (ng/mL)"},
    "cirrhosis":               {"range": (0, 1),      "type": "binary",       "desc": "History of cirrhosis (Yes/No)"},
    "family_history":          {"range": (0, 1),      "type": "binary",       "desc": "Family history of cancer (Yes/No)"},
    "physical_activity_level": {"range": None,        "type": "categorical",  "desc": "Low / Moderate / High"},
    "diabetes":                {"range": (0, 1),      "type": "binary",       "desc": "Diabetes diagnosis (Yes/No)"},
}

# ── Lung Cancer Features ──────────────────────────────────────────────────────

LUNG_FEATURES = {
    "age":                {"range": (1, 100),  "type": "numeric",     "desc": "Patient age in years"},
    "gender":             {"range": None,      "type": "categorical", "desc": "Male (1) / Female (2)"},
    "air_pollution":      {"range": (1, 8),    "type": "scale",       "desc": "Air pollution exposure level"},
    "alcohol_use":        {"range": (1, 8),    "type": "scale",       "desc": "Alcohol consumption level"},
    "dust_allergy":       {"range": (1, 8),    "type": "scale",       "desc": "Dust allergy severity"},
    "occupational_hazards":{"range": (1, 8),   "type": "scale",       "desc": "Workplace hazard exposure"},
    "genetic_risk":       {"range": (1, 8),    "type": "scale",       "desc": "Genetic predisposition level"},
    "chronic_lung_disease":{"range": (1, 8),   "type": "scale",       "desc": "Chronic lung disease severity"},
    "balanced_diet":      {"range": (1, 8),    "type": "scale",       "desc": "Diet quality (higher = healthier)"},
    "obesity":            {"range": (1, 8),    "type": "scale",       "desc": "Obesity level"},
    "smoking":            {"range": (1, 8),    "type": "scale",       "desc": "Smoking intensity"},
    "passive_smoker":     {"range": (1, 8),    "type": "scale",       "desc": "Secondhand smoke exposure"},
    "chest_pain":         {"range": (1, 9),    "type": "scale",       "desc": "Chest pain severity"},
    "coughing_of_blood":  {"range": (1, 9),    "type": "scale",       "desc": "Blood in cough frequency"},
    "fatigue":            {"range": (1, 9),    "type": "scale",       "desc": "Fatigue level"},
    "weight_loss":        {"range": (1, 8),    "type": "scale",       "desc": "Unexplained weight loss"},
    "shortness_of_breath":{"range": (1, 9),    "type": "scale",       "desc": "Breathing difficulty"},
    "wheezing":           {"range": (1, 8),    "type": "scale",       "desc": "Wheezing severity"},
    "swallowing_difficulty":{"range": (1, 8),  "type": "scale",       "desc": "Difficulty swallowing"},
    "clubbing_of_nails":  {"range": (1, 9),    "type": "scale",       "desc": "Nail clubbing severity"},
    "frequent_cold":      {"range": (1, 7),    "type": "scale",       "desc": "Cold frequency"},
    "dry_cough":          {"range": (1, 8),    "type": "scale",       "desc": "Dry cough frequency"},
    "snoring":            {"range": (1, 7),    "type": "scale",       "desc": "Snoring severity"},
}

# ── Breast Cancer Features (30 numeric measurements) ─────────────────────────

BREAST_FEATURES = {
    "radius_mean":            {"range": (6.0, 30.0),    "desc": "Mean radius of tumor nucleus"},
    "texture_mean":           {"range": (9.0, 40.0),    "desc": "Mean texture (gray-scale SD)"},
    "perimeter_mean":         {"range": (40.0, 190.0),  "desc": "Mean perimeter of nucleus"},
    "area_mean":              {"range": (140.0, 2500.0), "desc": "Mean area of nucleus"},
    "smoothness_mean":        {"range": (0.05, 0.17),   "desc": "Mean local radius variation"},
    "compactness_mean":       {"range": (0.02, 0.35),   "desc": "Mean compactness (perimeter²/area)"},
    "concavity_mean":         {"range": (0.0, 0.43),    "desc": "Mean concavity severity"},
    "concave_points_mean":    {"range": (0.0, 0.20),    "desc": "Mean concave portions count"},
    "symmetry_mean":          {"range": (0.10, 0.30),   "desc": "Mean symmetry of nucleus"},
    "fractal_dimension_mean": {"range": (0.05, 0.10),   "desc": "Mean fractal dimension"},
    # SE and Worst follow the same pattern — omitted for brevity
}


# ── Helper functions ──────────────────────────────────────────────────────────

def get_feature_info(disease: str) -> dict:
    """
    Returns the feature metadata dict for a given disease.

    Parameters
    ----------
    disease : str — "liver", "lung", or "breast"
    """
    return {
        "liver":  LIVER_FEATURES,
        "lung":   LUNG_FEATURES,
        "breast": BREAST_FEATURES,
    }.get(disease.lower(), {})


def get_feature_names(disease: str) -> list:
    """Returns an ordered list of feature names for a given disease."""
    return list(get_feature_info(disease).keys())


def get_feature_range(disease: str, feature: str) -> tuple | None:
    """Returns the (min, max) range for a numeric feature, or None if categorical."""
    info = get_feature_info(disease)
    return info.get(feature, {}).get("range", None)
