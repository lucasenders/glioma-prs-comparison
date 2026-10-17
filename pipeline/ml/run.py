"""One ML model on each feature set of one outer fold, tuned on the inner folds. Output goes to out/repeat<r>/<dataset>/fold<k>/ml/<model>/.

Run with: python run.py                                  submits one job per model and fold
          python run.py <dataset> <k> <model>            fits one model on one fold here
          python run.py <dataset> <k> <model> submit     submits that one job
          python run.py <dataset> <k> all submit         submits every model of that fold
          add --repeat r for a later repeat
"""

import importlib
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as cfg
import fitting

# Job resources and the models
RESOURCES = ["--time", "24:00:00", "--mem", "64G"]
MODELS = ["logistic", "elastic_net", "random_forest", "xgboost_model", "svm"]

def run(repeat, dataset, k, model):
    """Fit one model on every feature set of one fold."""

    # Load the model and the fold
    module = importlib.import_module(f"models.{model}")
    fold = fitting.Fold(repeat, dataset, k)
    out_dir = os.path.join(fold.dir, "ml", model)
    sets = cfg.LOGISTIC_SETS if model == "logistic" else cfg.SETS

    # Fit each feature set and write the results after each one
    rows = []
    for name in sets:
        row = fitting.run_set(module, fold, name, cfg.seed(repeat), os.path.join(out_dir, name))
        row["model"] = model
        rows.append(row)
        print(f"  {name:>16} {model:>14}  train {row['train_auc']:.3f}  test {row['test_auc']:.3f}  "
              f"{row['seconds']:>6}s  {row['chosen']}", flush=True)
        pd.DataFrame(rows).to_csv(f"{out_dir}/results.csv", index=False)

def submit(repeat, dataset, k, model):
    """Submit one model on one fold as its own job."""

    # Submit the job, with its log in the model's folder
    log_dir = os.path.join(cfg.fold_dir(repeat, dataset, k), "ml", model)
    cfg.submit(f"ml_{model}_{dataset}_fold{k}", f"{log_dir}/job.log", RESOURCES,
               f"{cfg.ANACONDA} && python {__file__} {dataset} {k} {model} --repeat {repeat}", HERE)

if __name__ == "__main__":
    # Fit one model on one fold here, or submit one job per model and fold
    repeat, dataset, k, rest, do_submit = cfg.parse(sys.argv)
    model = rest[0] if rest else "all"
    if dataset and k and model != "all" and not do_submit:
        run(repeat, dataset, k, model)
        print(f"\n>>> DONE: {model} {dataset} fold{k} repeat{repeat}")
    else:
        for m in (MODELS if model == "all" else [model]):
            for d, f in cfg.jobs(dataset, k):
                print(f"  {m} / {d} / fold{f}")
                submit(repeat, d, f, m)
        print(f"\n>>> SUBMITTED: check with squeue -u $USER")
