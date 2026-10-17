"""One MLP shape on each feature set of one outer fold, tuned on the inner folds. Output goes to out/repeat<r>/<dataset>/fold<k>/dl/<model>/.

Run with: python run.py                                  submits one job per shape and fold
          python run.py <dataset> <k> <model>            fits one shape on one fold here
          python run.py <dataset> <k> <model> submit     submits that one job
          python run.py <dataset> <k> all submit         submits every shape of that fold
          add --repeat r for a later repeat
          <model> is mlp (256-64-16), mlp_narrow (64-16-4), mlp_shallow (256-64) or mlp_small (64-16)
"""

import importlib
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as cfg
import fitting

# Job resources
RESOURCES = ["--time", "12:00:00", "--mem", "32G"]

# Name of each shape, and the module under models/ that builds it
MODELS = {"mlp": "mlp_main", "mlp_narrow": "mlp_narrow", "mlp_shallow": "mlp_shallow",
          "mlp_small": "mlp_small"}

def run(repeat, dataset, k, model):
    """Fit one shape on every feature set of one fold."""

    # Load the shape and the fold
    module = importlib.import_module(f"models.{MODELS[model]}")
    fold = fitting.Fold(repeat, dataset, k)
    out_dir = os.path.join(fold.dir, "dl", model)

    # Fit each feature set and write the results after each one
    rows = []
    for name in cfg.SETS:
        row = fitting.run_set(module, fold, name, cfg.seed(repeat), os.path.join(out_dir, name))
        row["model"] = model
        rows.append(row)
        print(f"  {name:>16} {model:>12}  train {row['train_auc']:.3f}  test {row['test_auc']:.3f}  "
              f"{row['seconds']:>6}s  {row['chosen']}", flush=True)
        pd.DataFrame(rows).to_csv(f"{out_dir}/results.csv", index=False)

def submit(repeat, dataset, k, model):
    """Submit one shape on one fold as its own job."""

    # Submit the job, with its log in the shape's folder
    log_dir = os.path.join(cfg.fold_dir(repeat, dataset, k), "dl", model)
    cfg.submit(f"dl_{model}_{dataset}_fold{k}", f"{log_dir}/job.log", RESOURCES,
               f"{cfg.ANACONDA} && python {__file__} {dataset} {k} {model} --repeat {repeat}", HERE)

if __name__ == "__main__":
    # Fit one shape on one fold here, or submit one job per shape and fold
    repeat, dataset, k, rest, do_submit = cfg.parse(sys.argv)
    model = rest[0] if rest else "all"
    if dataset and k and model != "all" and not do_submit:
        run(repeat, dataset, k, model)
        print(f"\n>>> DONE: {model} {dataset} fold{k} repeat{repeat}")
    else:
        for m in (list(MODELS) if model == "all" else [model]):
            for d, f in cfg.jobs(dataset, k):
                print(f"  {m} / {d} / fold{f}")
                submit(repeat, d, f, m)
        print(f"\n>>> SUBMITTED: check with squeue -u $USER")
