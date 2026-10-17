"""Elastic net logistic regression, with the penalty strength and the mix between ridge and lasso tuned."""

import warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

# Penalty strengths, 20 values from 1e-6 to 1e2, and the mix between ridge and lasso
CS = np.logspace(-6, 2, 20)
L1_RATIOS = [0, 0.25, 0.5, 0.75, 1]

# Iteration limit
MAX_ITER = 10000

# Tuning grid
GRID = [{"C": float(c), "l1_ratio": r} for c in CS for r in L1_RATIOS]

# Number of penalty paths fitted at once in the inner search
N_JOBS = 8

def build(setting, seed, pos_weight):
    """Build the estimator at one setting."""

    return LogisticRegression(C=setting["C"], l1_ratio=setting["l1_ratio"], penalty="elasticnet",
                              solver="saga", max_iter=MAX_ITER, class_weight="balanced",
                              random_state=seed)

def fit(estimator, x, y, x_val=None, y_val=None):
    """Fit the estimator, and record whether the solver converged."""

    # Fit, catching any convergence warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x, y)
    return estimator, not any(issubclass(w.category, ConvergenceWarning) for w in caught), {}

def path(l1_ratio, x, y, x_val, seed):
    """Fit one mix's 20 penalties in turn, each starting from the last, and score the validation set."""

    # Build one warm-started estimator for the mix
    estimator = LogisticRegression(l1_ratio=l1_ratio, penalty="elasticnet", solver="saga",
                                   max_iter=MAX_ITER, class_weight="balanced", random_state=seed,
                                   warm_start=True)

    # Fit each penalty in turn and score the validation set
    out = []
    for c in CS:
        estimator.set_params(C=float(c))
        _, converged, _ = fit(estimator, x, y)
        out.append(({"C": float(c), "l1_ratio": l1_ratio}, score(estimator, x_val), converged, {}))
    return out

def fit_grid(splits, seed, pos_weight):
    """Fit every setting on every inner fold, as (setting, validation scores, converged, info) per fold."""

    # Fit each inner fold's paths in parallel
    tasks = [(j, r) for j in range(len(splits)) for r in L1_RATIOS]
    done = Parallel(n_jobs=N_JOBS)(
        delayed(path)(r, splits[j][0], splits[j][1], splits[j][2], seed) for j, r in tasks)

    # Group the results by inner fold
    per_split = [[] for _ in splits]
    for (j, _), items in zip(tasks, done):
        per_split[j] += items
    return per_split

def score(fitted, x):
    """Return each person's probability of being a case."""

    return fitted.predict_proba(x)[:, 1]

def importance(fitted):
    """Return the fitted coefficients."""

    return fitted.coef_[0]

def regularisation(setting, n_features):
    """Rank settings from weak to strong: smaller C, then a larger lasso share."""

    return (-setting["C"], setting["l1_ratio"])
