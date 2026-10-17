"""Unpenalised logistic regression, with nothing tuned."""

import warnings
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

# Penalty setting for an unpenalised fit, which sklearn renamed at version 1.2
NO_PENALTY = None if sklearn.__version__ >= "1.2" else "none"

# Iteration limit
MAX_ITER = 10000

# Tuning grid, a single setting
GRID = [{}]

def build(setting, seed, pos_weight):
    """Build the estimator."""

    return LogisticRegression(penalty=NO_PENALTY, max_iter=MAX_ITER, class_weight="balanced")

def fit(estimator, x, y, x_val=None, y_val=None):
    """Fit the estimator, and record whether the solver converged."""

    # Fit, catching any convergence warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x, y)
    return estimator, not any(issubclass(w.category, ConvergenceWarning) for w in caught), {}

def score(fitted, x):
    """Return each person's probability of being a case."""

    return fitted.predict_proba(x)[:, 1]

def importance(fitted):
    """Return the fitted coefficients."""

    return fitted.coef_[0]

def regularisation(setting, n_features):
    """Return an empty rank, since there is only one setting."""

    return ()
