"""Random forest of 500 trees, with the SNPs per split and the minimum leaf size tuned."""

import math
from sklearn.ensemble import RandomForestClassifier

# SNPs per split, minimum leaf sizes and number of trees
MAX_FEATURES = ["sqrt", 0.1]
MIN_SAMPLES_LEAF = [1, 10, 50, 100]
N_ESTIMATORS = 500

# Tuning grid
GRID = [{"max_features": f, "min_samples_leaf": m} for f in MAX_FEATURES for m in MIN_SAMPLES_LEAF]

def build(setting, seed, pos_weight):
    """Build the estimator at one setting."""

    return RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features=setting["max_features"],
                                  min_samples_leaf=setting["min_samples_leaf"],
                                  class_weight="balanced", random_state=seed, n_jobs=-1)

def fit(estimator, x, y, x_val=None, y_val=None):
    """Fit the estimator, which always converges."""

    return estimator.fit(x, y), True, {}

def score(fitted, x):
    """Return each person's probability of being a case."""

    return fitted.predict_proba(x)[:, 1]

def importance(fitted):
    """Return each SNP's impurity reduction."""

    return fitted.feature_importances_

def regularisation(setting, n_features):
    """Rank settings from weak to strong: a larger leaf, then fewer SNPs per split."""

    per_split = math.sqrt(n_features) if setting["max_features"] == "sqrt" else 0.1 * n_features
    return (setting["min_samples_leaf"], -per_split)
