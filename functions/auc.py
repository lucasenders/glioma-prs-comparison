import numpy as np
from scipy.stats import norm, rankdata

def auc_ci(y, score, alpha=0.05):
    """Return the AUC and its DeLong 95% CI."""

    # Rank the cases and the controls by score, separately and together
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    cases, controls = score[y == 1], score[y == 0]
    m, n = len(cases), len(controls)
    rank_case, rank_ctrl = rankdata(cases), rankdata(controls)
    rank_both = rankdata(np.concatenate([cases, controls]))

    # Compute the AUC
    auc = (rank_both[:m].sum() - m * (m + 1) / 2.0) / (m * n)

    # Compute the standard error of the AUC
    v_case = (rank_both[:m] - rank_case) / n
    v_ctrl = 1.0 - (rank_both[m:] - rank_ctrl) / m
    se = np.sqrt(np.var(v_case, ddof=1) / m + np.var(v_ctrl, ddof=1) / n)

    # Compute the 95% confidence interval
    a = min(max(auc, 1e-6), 1 - 1e-6)
    logit, se_logit = np.log(a / (1 - a)), se / (a * (1 - a))
    z = norm.ppf(1 - alpha / 2)
    lo = 1 / (1 + np.exp(-(logit - z * se_logit)))
    hi = 1 / (1 + np.exp(-(logit + z * se_logit)))
    return auc, lo, hi
