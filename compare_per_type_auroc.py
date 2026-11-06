"""
Per-anomaly-type image AUROC from run_anomalydino.py outputs, compared across runs.

Each run directory must contain measurements_seed=<s>.csv
(columns: Object, Sample="<type>/<file>", Anomaly_Score, ...).
For every object and anomaly type, AUROC = good vs. that type only.
"ALL" = good vs. all anomalies (the usual image-level AUROC).
Seeds are averaged (mean ± std).

Usage (label=path, first run is the baseline for the deltas):
  python compare_per_type_auroc.py \
      --runs no_mask=results_MVTec/dinov2_vits14_448/1-shot_preprocess=agnostic_no_mask \
             mask=results_MVTec/dinov2_vits14_448/1-shot_preprocess=agnostic \
      --objects transistor
Outputs: per_type_auroc.csv (long format) and a printed comparison table.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_run(path):
    files = sorted(glob.glob(os.path.join(path, "measurements_seed=*.csv")))
    if not files:
        raise FileNotFoundError(f"no measurements_seed=*.csv in {path}")
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["seed"] = int(re.search(r"seed=(\d+)", f).group(1))
        dfs.append(d)
    d = pd.concat(dfs, ignore_index=True)
    d["type"] = d["Sample"].str.split("/").str[0]
    return d


def per_type_auroc(d):
    rows = []
    for (obj, seed), g in d.groupby(["Object", "seed"]):
        good = g.loc[g["type"] == "good", "Anomaly_Score"].values
        if len(good) == 0:
            continue
        types = sorted(t for t in g["type"].unique() if t != "good")
        for t in types + ["ALL"]:
            bad = g.loc[(g["type"] != "good") if t == "ALL" else (g["type"] == t), "Anomaly_Score"].values
            y = np.r_[np.zeros(len(good)), np.ones(len(bad))]
            rows.append(dict(object=obj, type=t, seed=seed, n_bad=len(bad),
                             auroc=roc_auc_score(y, np.r_[good, bad])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=path (first = baseline)")
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--out", default="per_type_auroc.csv")
    a = ap.parse_args()

    res = []
    labels = []
    for r in a.runs:
        label, path = r.split("=", 1) if "=" in r.split("/")[0] else (os.path.basename(r.rstrip("/")), r)
        labels.append(label)
        d = load_run(path)
        if a.objects:
            d = d[d["Object"].isin(a.objects)]
        p = per_type_auroc(d); p["run"] = label
        print(f"[{label}] {path}: seeds={sorted(d['seed'].unique().tolist())}, objects={d['Object'].nunique()}")
        res.append(p)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(a.out, index=False)

    agg = res.groupby(["object", "type", "run"]).agg(mean=("auroc", "mean"), std=("auroc", "std"),
                                                     n_seeds=("auroc", "size"), n_bad=("n_bad", "first")).reset_index()
    mean = agg.pivot_table(index=["object", "type"], columns="run", values="mean")[labels]
    std = agg.pivot_table(index=["object", "type"], columns="run", values="std")[labels]
    nbad = agg.groupby(["object", "type"])["n_bad"].first()

    table = pd.DataFrame(index=mean.index)
    table["n_bad"] = nbad
    for l in labels:
        table[l] = [f"{m:.3f}" + (f" ±{s:.3f}" if pd.notna(s) else "") for m, s in zip(mean[l], std[l])]
    base = labels[0]
    for l in labels[1:]:
        table[f"Δ {l}-{base}"] = (mean[l] - mean[base]).map(lambda x: f"{x:+.3f}")
        # flag deltas larger than the seed noise (pooled std), when >1 seed
        noise = np.sqrt((std[l].fillna(0) ** 2 + std[base].fillna(0) ** 2) / 2)
        table[f"sig {l}"] = np.where(noise > 0, np.where((mean[l] - mean[base]).abs() > 2 * noise, "*", ""), "")

    # put ALL last within each object
    order = sorted(table.index, key=lambda k: (k[0], k[1] == "ALL", k[1]))
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    print(table.loc[order].to_string())
    print(f"\nsaved {a.out}   (* = |Δ| > 2× pooled seed std; needs --num_seeds > 1)")
