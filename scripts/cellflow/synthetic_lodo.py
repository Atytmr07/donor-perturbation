from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import anndata as ad
import flax.linen as nn
import numpy as np
import optax
import pandas as pd
from cellflow.model import CellFlow

from pertbench.data import pseudobulk
from pertbench.metrics import condition_metrics, distribution_metrics
from pertbench.models import ContextRidge, NoChange, PerturbationMean
from pertbench.synthetic import make_synthetic
from pertbench.splits import Split

MODS = ["rna", "protein", "atac"]


def fit_space(X, train, n_pcs):
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Z = (X - mu) / sd
    _, _, vt = np.linalg.svd(Z[train] - Z[train].mean(0), full_matrices=False)
    c, V = Z[train].mean(0), vt[:n_pcs].T
    encode = lambda A: ((A - mu) / sd - c) @ V
    decode = lambda P: (P @ V.T + c) * sd + mu
    return encode, decode


def run_fold(data, donor, a):
    obs = data.obs.reset_index(drop=True)
    X = np.hstack([data.X[m] for m in MODS])
    cuts = np.cumsum([0] + [data.X[m].shape[1] for m in MODS])
    is_ctrl = (obs["perturbation"] == data.control).to_numpy()
    hidden = (obs["donor"] == donor).to_numpy() & ~is_ctrl
    train = ~hidden
    train_donors = (obs["donor"] != donor).to_numpy()
    encode, decode = fit_space(X, train_donors, a.n_pcs)

    obs = obs.assign(context=obs["cell_type"] + "_" + obs["donor"], is_control=is_ctrl)
    adata = ad.AnnData(X=np.zeros((len(obs), 1), np.float32), obs=obs)
    adata.obsm["X_pca"] = encode(X).astype(np.float32)
    perts = sorted(set(obs["perturbation"]) - {data.control})
    adata.uns["pert_emb"] = {p: np.eye(len(perts), dtype=np.float32)[i] for i, p in enumerate(perts)}
    adata.uns["pert_emb"][data.control] = np.zeros(len(perts), np.float32)
    adata.uns["ctx_emb"] = {c: adata.obsm["X_pca"][(obs["context"] == c).to_numpy() & is_ctrl].mean(0)
                            for c in obs["context"].unique()}

    tr = adata[train_donors].copy()
    cf = CellFlow(tr, solver="otfm")
    cf.prepare_data(sample_rep="X_pca", control_key="is_control",
                    perturbation_covariates={"pert": ("perturbation",)},
                    perturbation_covariate_reps={"pert": "pert_emb"},
                    sample_covariates=["context"], sample_covariate_reps={"context": "ctx_emb"},
                    split_covariates=["context"], max_combination_length=1)
    w = a.width
    cf.prepare_model(condition_embedding_dim=64, cond_output_dropout=0.1, pooling="mean",
                     layers_before_pool={"pert": {"layer_type": "mlp", "dims": [w, w], "dropout_rate": 0.0},
                                         "context": {"layer_type": "mlp", "dims": [w, w], "dropout_rate": 0.0}},
                     layers_after_pool={"layer_type": "mlp", "dims": [w], "dropout_rate": 0.0},
                     time_freqs=64, time_encoder_dims=[w, w], hidden_dims=[w, w, w], decoder_dims=[w, w, w],
                     vf_act_fn=nn.silu, probability_path={"constant_noise": 0.5},
                     optimizer=optax.adam(a.lr), seed=a.seed)
    t0 = time.time()
    cf.train(num_iterations=a.iterations, batch_size=a.batch, valid_freq=a.iterations + 1)
    t_train = time.time() - t0

    test_ctrl = adata[(obs["donor"] == donor).to_numpy() & is_ctrl].copy()
    cov = obs[hidden].drop_duplicates(["perturbation", "context"]).copy()
    cov["cond"] = cov["perturbation"] + "|" + cov["context"]
    preds = cf.predict(adata=test_ctrl, covariate_data=cov, sample_rep="X_pca", condition_id_key="cond")

    pb = pseudobulk(data)
    test_mask = (pb.obs["donor"] == donor).to_numpy() & ~pb.is_control()
    split = Split(f"lodo_{donor}", train=~test_mask, test=test_mask)
    base = {m.name: m.fit(pb.subset(split.train)).predict(pb.obs[split.test])
            for m in (NoChange(), PerturbationMean(), ContextRidge())}
    q = pb.obs[split.test].reset_index(drop=True)
    pca_true = adata.obsm["X_pca"]

    rows = []
    for i, (p, ct, dn) in enumerate(q[["perturbation", "cell_type", "donor"]].itertuples(index=False, name=None)):
        ctx = f"{ct}_{dn}"
        cells_pred = np.squeeze(preds[f"{p}|{ctx}"])
        sel_true = ((obs["perturbation"] == p) & (obs["context"] == ctx)).to_numpy()
        sel_ctrl = ((obs["context"] == ctx).to_numpy() & is_ctrl)
        dec = decode(cells_pred).mean(0)
        oracle = decode(pca_true[sel_true]).mean(0)
        dist = distribution_metrics(cells_pred, pca_true[sel_true])
        ctrl_dist = distribution_metrics(pca_true[sel_ctrl], pca_true[sel_true])
        for k, m in enumerate(MODS):
            ctrl_mean, true_mean = X[sel_ctrl, cuts[k]:cuts[k + 1]].mean(0), X[sel_true, cuts[k]:cuts[k + 1]].mean(0)
            cand = {"CellFlow": dec[cuts[k]:cuts[k + 1]], "PCAOracle": oracle[cuts[k]:cuts[k + 1]],
                    **{n: b[m][i] for n, b in base.items()}}
            for name, pred in cand.items():
                r = condition_metrics(pred, true_mean, ctrl_mean)
                r.update(model=name, modality=m, donor=dn, cell_type=ct, perturbation=p)
                if name == "CellFlow":
                    r.update(dist)
                elif name == "NoChange":
                    r.update(ctrl_dist)
                rows.append(r)
    print(f"[{donor}] trained {a.iterations} it in {t_train / 60:.1f} min", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", nargs="*", default=None, help="default: all donors")
    ap.add_argument("--n-donors", type=int, default=8)
    ap.add_argument("--n-perts", type=int, default=30)
    ap.add_argument("--nonlinear", type=float, default=0.0, help="donor_nonlinear of make_synthetic")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--n-pcs", type=int, default=30)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cellflow_synthetic_lodo")
    ap.add_argument("--cores", type=int, default=0, help="if > 0: pin to this many CPUs at low priority")
    a = ap.parse_args()
    if a.cores > 0:
        import psutil
        p = psutil.Process()
        p.cpu_affinity(list(range(a.cores)))
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)

    data = make_synthetic(n_donors=a.n_donors, n_perts=a.n_perts, donor_nonlinear=a.nonlinear, seed=a.data_seed)
    donors = a.donors or sorted(data.obs["donor"].unique())
    out = ROOT / a.out
    (out / "folds").mkdir(parents=True, exist_ok=True)
    deadline, last = float(os.environ.get("SWEEP_DEADLINE", "inf")), 0.0
    for d in donors:
        f = out / "folds" / f"{d}.csv"
        if f.exists():
            print(f"[{d}] done earlier, skipped", flush=True)
            continue
        if time.time() + last > deadline:
            print(f"[{d}] deadline reached, stopping", flush=True)
            break
        t0 = time.time()
        pd.DataFrame(run_fold(data, d, a)).to_csv(f, index=False)
        last = time.time() - t0
    res = pd.concat([pd.read_csv(f) for f in sorted((out / "folds").glob("*.csv"))], ignore_index=True)
    res.to_csv(out / "per_condition.csv", index=False)
    cols = ["pearson_delta", "mse_top20", "direction_acc_top20", "mmd", "energy"]
    cols = [c for c in cols if c in res]
    summ = res.groupby(["modality", "model"])[cols].mean().round(3)
    summ.to_csv(out / "summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
