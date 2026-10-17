"""Nested fit shared by the ML and DL models: inner search, outer refit, test scoring and output files.

Each model module provides: GRID (a list of settings, each a dict), build(setting, seed, pos_weight),
fit(estimator, x, y, x_val, y_val) -> (fitted, converged, info), score(fitted, x), importance(fitted),
regularisation(setting, n_features), and optionally finalise(setting, tuning_row).
"""

import os
import time
import numpy as np
import pandas as pd

import config as cfg
from functions import auc

class Fold:
    """Load one outer fold's matrix, training and test sets, inner folds and feature sets."""

    def __init__(self, repeat, dataset, k):
        # Read the matrix, the outer training and test sets and the inner folds
        self.repeat, self.dataset, self.k = repeat, dataset, k
        self.dir = cfg.fold_dir(repeat, dataset, k)
        self.table = pd.read_csv(f"{self.dir}/matrix.csv", dtype={"IID": str}).set_index("IID")
        self.train_ids = cfg.read_ids(f"{cfg.outer_dir(repeat, dataset, k)}/train.txt")
        self.test_ids = cfg.read_ids(f"{cfg.outer_dir(repeat, dataset, k)}/test.txt")
        self.inner = [(cfg.read_ids(f"{cfg.inner_dir(repeat, dataset, k, j)}/train.txt"),
                       cfg.read_ids(f"{cfg.inner_dir(repeat, dataset, k, j)}/val.txt"), j)
                      for j in range(1, cfg.INNER + 1)]
        self.check_ids()

    def check_ids(self):
        """Check no test person is in the training set, and no inner validation person in its inner training set."""

        # Check the outer training and test sets
        train, test = set(self.train_ids), set(self.test_ids)
        if train & test:
            raise ValueError(f"{len(train & test)} people are in both train.txt and test.txt")
        if (self.table["is_train"] != self.table.index.isin(train)).any():
            raise ValueError("matrix is_train disagrees with train.txt")

        # Check each inner fold
        for fit, val, j in self.inner:
            if set(fit) & set(val):
                raise ValueError(f"inner{j}: {len(set(fit) & set(val))} people in both halves")
            if not set(fit) <= train or not set(val) <= train:
                raise ValueError(f"inner{j}: a person outside the outer training set")

    def snps(self, name, level="outer"):
        """Read one feature set at one level: 'outer' or 'inner<j>'."""

        folder = f"{self.dir}/outer_sets" if level == "outer" else f"{self.dir}/{level}/sets"
        return cfg.read_list(f"{folder}/{name}.txt")

    def labels(self, ids):
        """Return the case status of the people named, in that order."""

        return self.table.loc[ids, "PHENO"].values

def prepare(table, snps, fit_ids, other_ids):
    """Impute, drop constant SNPs and standardise on the fitting set, and apply the same to the others."""

    # Impute missing genotypes with the fitting set's mean and drop constant SNPs
    fit = table.loc[fit_ids, snps]
    mean = fit.mean()
    fit = fit.fillna(mean)
    sd = fit.std()
    keep = sd[sd > 0].index

    # Standardise both sets with the fitting set's mean and SD
    x_fit = (fit[keep] - mean[keep]) / sd[keep]
    other = table.loc[other_ids, keep].fillna(mean[keep])
    x_other = (other - mean[keep]) / sd[keep]
    return x_fit, x_other

def pos_weight(y):
    """Return the number of controls per case."""

    return float((y == 0).sum() / (y == 1).sum())

def search(module, fold, name, seed, weight):
    """Fit every setting on every inner fold, and return the tuning table and the chosen setting."""

    # Fit the whole grid at once if the model can, otherwise one setting at a time
    if hasattr(module, "fit_grid"):
        table = search_grid(module, fold, name, seed, weight)
    else:
        table = search_each(module, fold, name, seed, weight)

    # Choose the highest mean inner AUC, breaking ties towards the more strongly regularised setting
    n_outer = len(fold.snps(name))
    best = max(range(len(module.GRID)),
               key=lambda i: (table.at[i, "mean_auc"], module.regularisation(module.GRID[i], n_outer)))
    return table, module.GRID[best], best

def search_grid(module, fold, name, seed, weight):
    """Build the tuning table from a model that fits its whole grid on every inner fold at once."""

    # Prepare each inner fold
    splits, labels = [], []
    for fit_ids, val_ids, j in fold.inner:
        x_fit, x_val = prepare(fold.table, fold.snps(name, f"inner{j}"), fit_ids, val_ids)
        splits.append((x_fit, fold.labels(fit_ids), x_val))
        labels.append((fold.labels(val_ids), j, x_fit.shape[1]))

    # Fit the grid and record each setting's validation AUC on each inner fold
    rows = {tuple(sorted(s.items())): dict(s) for s in module.GRID}
    for items, (y_val, j, width) in zip(module.fit_grid(splits, seed, weight), labels):
        for setting, scores, converged, info in items:
            row = rows[tuple(sorted(setting.items()))]
            row[f"auc_inner{j}"], _, _ = auc.auc_ci(y_val, scores)
            row[f"converged_inner{j}"] = converged
            row[f"n_snps_inner{j}"] = width
            row.update({f"{key}_inner{j}": v for key, v in info.items()})

    # Average the AUC over the inner folds
    table = pd.DataFrame([rows[tuple(sorted(s.items()))] for s in module.GRID])
    table["mean_auc"] = table[[f"auc_inner{j}" for _, _, j in fold.inner]].mean(axis=1)
    return table

def search_each(module, fold, name, seed, weight):
    """Build the tuning table by fitting one setting at a time."""

    # Fit each setting on each inner fold and record its validation AUC
    rows = []
    for setting in module.GRID:
        row = dict(setting)
        for fit_ids, val_ids, j in fold.inner:
            x_fit, x_val = prepare(fold.table, fold.snps(name, f"inner{j}"), fit_ids, val_ids)
            y_val = fold.labels(val_ids)
            fitted, converged, info = module.fit(module.build(setting, seed, weight),
                                                 x_fit, fold.labels(fit_ids), x_val, y_val)
            value, _, _ = auc.auc_ci(y_val, module.score(fitted, x_val))
            row[f"auc_inner{j}"] = value
            row[f"converged_inner{j}"] = converged
            row[f"n_snps_inner{j}"] = x_fit.shape[1]
            row.update({f"{key}_inner{j}": v for key, v in info.items()})
        row["mean_auc"] = np.mean([row[f"auc_inner{j}"] for _, _, j in fold.inner])
        rows.append(row)
    return pd.DataFrame(rows)

def run_set(module, fold, name, seed, out_dir):
    """Tune, refit on the outer training set, score everyone, and write the output files."""

    # Read the outer labels and the class ratio
    started = time.time()
    os.makedirs(out_dir, exist_ok=True)
    y_train, y_test = fold.labels(fold.train_ids), fold.labels(fold.test_ids)
    weight = pos_weight(y_train)

    # Run the inner search, or take the only setting
    if len(module.GRID) > 1:
        tuning, setting, best = search(module, fold, name, seed, weight)
        tuning["chosen"] = [i == best for i in range(len(tuning))]
        tuning.to_csv(f"{out_dir}/tuning.csv", index=False)
        if hasattr(module, "finalise"):
            setting = module.finalise(setting, tuning.iloc[best])
    else:
        tuning, setting = None, module.GRID[0]

    # Refit on the outer training set with the outer feature set and score everyone
    x_train, x_test = prepare(fold.table, fold.snps(name), fold.train_ids, fold.test_ids)
    fitted, converged, _ = module.fit(module.build(setting, seed, weight), x_train, y_train)
    train_score, test_score = module.score(fitted, x_train), module.score(fitted, x_test)

    # Write the scores and the chosen setting
    pd.DataFrame({"IID": fold.train_ids + fold.test_ids,
                  "is_train": [True] * len(fold.train_ids) + [False] * len(fold.test_ids),
                  "case": np.concatenate([y_train, y_test]),
                  "score": np.concatenate([train_score, test_score])}
                 ).to_csv(f"{out_dir}/scores.csv", index=False)
    pd.DataFrame([dict(setting, converged=converged, n_snps=x_train.shape[1])]
                 ).to_csv(f"{out_dir}/chosen.csv", index=False)

    # Write every fit's convergence flag, inner and final
    flags = [{"level": "outer", **setting, "converged": converged}]
    if tuning is not None:
        for _, row in tuning.iterrows():
            for _, _, j in fold.inner:
                flags.append({"level": f"inner{j}", **{k: row[k] for k in setting if k in row},
                              "converged": row[f"converged_inner{j}"]})
    pd.DataFrame(flags).to_csv(f"{out_dir}/converged.csv", index=False)

    # Write each SNP's importance, largest first
    weights = module.importance(fitted)
    used = pd.DataFrame({"SNP": x_train.columns,
                         "importance": weights if weights is not None else np.nan}).dropna()
    used = used[used["importance"] != 0]
    used.iloc[used["importance"].abs().argsort()[::-1]].to_csv(f"{out_dir}/importance.csv", index=False)

    # Return the training and test AUC of the final fit
    t_auc = auc.auc_ci(y_train, train_score)
    v_auc = auc.auc_ci(y_test, test_score)
    return {"repeat": fold.repeat, "dataset": fold.dataset, "fold": fold.k, "features": name,
            "n_snps": x_train.shape[1],
            "train_auc": t_auc[0], "train_lo": t_auc[1], "train_hi": t_auc[2],
            "test_auc": v_auc[0], "test_lo": v_auc[1], "test_hi": v_auc[2],
            "chosen": " ".join(f"{k}={v}" for k, v in setting.items()),
            "converged": converged,
            "n_not_converged": int(sum(not f["converged"] for f in flags)),
            "seconds": round(time.time() - started)}
