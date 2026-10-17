"""MLP settings and fitting shared by the four shapes, with the dropout after the first hidden layer tuned."""

import numpy as np

from models import mlp

def grid():
    """Return one setting per dropout."""

    return [{"dropout": d} for d in mlp.DROPOUT]

def build(setting, seed, pos_weight, hidden):
    """Return the settings one fit needs, since the network is built once the input width is known."""

    return dict(setting, seed=seed, pos_weight=pos_weight, hidden=hidden)

def fit(estimator, x, y, x_val=None, y_val=None):
    """Fit an inner network with early stopping on the validation set, or the outer network for a fixed number of epochs."""

    # Build the network
    net = mlp.MLP(x.shape[1], estimator["hidden"], estimator["dropout"],
                  estimator["pos_weight"], estimator["seed"])

    # Fit an inner network with early stopping, and return its best epoch
    if x_val is not None:
        epoch, value = net.fit(x, y, x_val=x_val, y_val=y_val)
        return net, True, {"epoch": epoch}

    # Fit the outer network for the given number of epochs
    net.fit(x, y, epochs=estimator["epochs"])
    return net, True, {}

def finalise(setting, row):
    """Add the outer number of epochs to the chosen setting: the mean best inner epoch, scaled by 2/3."""

    stops = [row[f"epoch_inner{j}"] for j in range(1, mlp_inner_count(row) + 1)]
    epochs = int(round(np.mean(stops)))
    epochs = max(1, int(round(epochs * (len(stops) - 1) / len(stops))))
    return dict(setting, epochs=epochs)

def mlp_inner_count(row):
    """Count the inner folds in a tuning row."""

    return sum(1 for c in row.index if str(c).startswith("epoch_inner"))

def score(fitted, x):
    """Return each person's logit."""

    return fitted.logits(x)

def importance(fitted):
    """Return the size of each SNP's first-layer weights."""

    return fitted.importance()

def regularisation(setting, n_features):
    """Rank settings from weak to strong: more dropout."""

    return (setting["dropout"],)
