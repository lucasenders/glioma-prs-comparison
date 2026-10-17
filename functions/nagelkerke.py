import numpy as np
from sklearn.linear_model import LogisticRegression

def r2(y, score):
    """Return Nagelkerke's R-squared for one score against case status."""

    # Fit case status on the score alone, and read off each person's fitted risk
    y = np.asarray(y, dtype=float)
    x = np.asarray(score, dtype=float).reshape(-1, 1)
    fitted = LogisticRegression(penalty=None, max_iter=5000).fit(x, y)
    p = np.clip(fitted.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)

    # Take the log-likelihood of that fit and of an intercept-only model
    prevalence = np.clip(y.mean(), 1e-12, 1 - 1e-12)
    full = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    null = len(y) * (prevalence * np.log(prevalence)
                     + (1 - prevalence) * np.log(1 - prevalence))

    # Scale Cox and Snell's R-squared by the most it could have reached
    cox = 1 - np.exp((2.0 / len(y)) * (null - full))
    return cox / (1 - np.exp((2.0 / len(y)) * null))
