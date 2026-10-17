import os
import pandas as pd
from functions import auc as auc_mod
from functions import plots

# Columns of every summary table, per model and combined
SUMMARY_COLS = ["analysis", "dataset", "stratum", "n_case", "n_ctrl",
                "auc", "ci_low", "ci_high"]

def evaluate(people, scores, outdir, analysis, dataset, stratum, label, tag=None):
    """AUC, CI and two figures for one subgroup; returns one summary row."""

    # Get this subgroup's AUC and confidence interval
    n_case = int(people["case"].sum())
    n_ctrl = len(people) - n_case
    scores = scores.loc[people.index]
    value, lo, hi = auc_mod.auc_ci(people["case"], scores)

    # Build the folder, filename and title for this subgroup's figures
    folder = f"{outdir}/{analysis}"
    os.makedirs(folder, exist_ok=True)
    name = dataset if stratum == "all" else f"{dataset}_{tag or stratum}"
    head = dataset if stratum == "all" else f"{dataset} — {stratum}"

    # Save the distribution plot and the confusion table as PNGs
    plots.distributions(scores, people["case"],
                        f"{head}   ·   {label}\n"
                        f"AUC = {value:.3f}   (95% CI {lo:.3f} – {hi:.3f})",
                        f"{folder}/{name}_dist.png")
    plots.confusion_table(people["case"], scores, f"{head}   ·   {label}",
                          f"{folder}/{name}_confusion.png")

    # Report the result as it runs, and hand back the row for the summary table
    print(f"    {analysis:15}{stratum:22} AUC={value:.3f} ({lo:.3f}–{hi:.3f})  "
          f"(cases={n_case}, controls={n_ctrl})")
    return [analysis, dataset, stratum, n_case, n_ctrl,
            round(value, 4), round(lo, 4), round(hi, 4)]

def combine(results_dir, models, analysis_order):
    """Stack per-model summaries into master_results.csv and master_auc.csv."""

    # Gather the summary from every model that has been run
    frames = [pd.read_csv(f"{results_dir}/{m}/auc_summary.csv").assign(model=m)
              for m in models
              if os.path.exists(f"{results_dir}/{m}/auc_summary.csv")]
    if not frames:
        return

    # Stack them into one row per model, analysis, dataset and subgroup
    long = pd.concat(frames, ignore_index=True)
    long.insert(0, "model", long.pop("model"))
    long["analysis"] = pd.Categorical(long["analysis"], categories=analysis_order,
                                      ordered=True)
    long = long.sort_values(["analysis", "dataset", "stratum"], kind="stable")
    long.to_csv(f"{results_dir}/master_results.csv", index=False)

    # Turn that into the comparison table: one column of AUCs per model
    long["auc_ci"] = long.apply(
        lambda r: "" if pd.isna(r["auc"])
        else f"{r['auc']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})", axis=1)
    wide = long.pivot_table(index=["analysis", "dataset", "stratum"], columns="model",
                            values="auc_ci", aggfunc="first", observed=True)
    wide = wide.reindex(columns=[m for m in models if m in wide.columns])
    wide.to_csv(f"{results_dir}/master_auc.csv")
    print(f"\n{wide.to_string()}")
