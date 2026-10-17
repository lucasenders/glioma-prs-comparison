"""XGBoost with 500 boosting rounds, with the maximum depth and learning rate tuned."""

from xgboost import XGBClassifier

# Maximum depths, learning rates and number of boosting rounds
MAX_DEPTH = [1, 2, 4, 6]
LEARNING_RATE = [0.01, 0.1, 0.3]
N_ESTIMATORS = 500

# Tuning grid
GRID = [{"max_depth": d, "learning_rate": r} for d in MAX_DEPTH for r in LEARNING_RATE]

def build(setting, seed, pos_weight):
    """Build the estimator at one setting, with cases weighted by the number of controls per case."""

    return XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=setting["max_depth"],
                         learning_rate=setting["learning_rate"], eval_metric="logloss",
                         scale_pos_weight=pos_weight, random_state=seed, n_jobs=-1)

def fit(estimator, x, y, x_val=None, y_val=None):
    """Fit the estimator, which runs a fixed number of rounds and always converges."""

    return estimator.fit(x, y), True, {}

def score(fitted, x):
    """Return each person's probability of being a case."""

    return fitted.predict_proba(x)[:, 1]

def importance(fitted):
    """Return each SNP's gain across the rounds."""

    return fitted.feature_importances_

def regularisation(setting, n_features):
    """Rank settings from weak to strong: shallower trees, then a smaller learning rate."""

    return (-setting["max_depth"], -setting["learning_rate"])
