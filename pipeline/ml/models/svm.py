"""Support vector machine, with the kernel and C tuned."""

from sklearn.svm import SVC

# Kernels and C values
KERNEL = ["linear", "rbf"]
C = [0.1, 1, 10]

# Tuning grid
GRID = [{"kernel": k, "C": c} for k in KERNEL for c in C]

def build(setting, seed, pos_weight):
    """Build the estimator at one setting."""

    return SVC(kernel=setting["kernel"], C=setting["C"], class_weight="balanced", random_state=seed)

def fit(estimator, x, y, x_val=None, y_val=None):
    """Fit the estimator, which runs to its own tolerance with no iteration limit."""

    return estimator.fit(x, y), True, {}

def score(fitted, x):
    """Return each person's decision margin."""

    return fitted.decision_function(x)

def importance(fitted):
    """Return the coefficients of a linear kernel, or nothing for the radial basis function kernel."""

    return fitted.coef_[0] if fitted.kernel == "linear" else None

def regularisation(setting, n_features):
    """Rank settings from weak to strong: smaller C, then linear before radial basis function."""

    return (-setting["C"], 1 if setting["kernel"] == "linear" else 0)
