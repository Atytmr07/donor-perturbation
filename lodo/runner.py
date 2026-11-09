from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from lodo.data import KEYS, ConditionTable
from lodo.metrics import condition_metrics, delta_norm
from lodo.models import Model
from lodo.splits import SPLITS, Split


def evaluate_split(model: Model, ct: ConditionTable, split: Split, top_k: int = 20) -> pd.DataFrame:
    train = ct.subset(split.train)
    ctrl_ctx = set(map(tuple, train.obs.loc[train.is_control(), ["cell_type", "donor"]].itertuples(index=False, name=None)))
    has_ctrl = np.array([(c, d) in ctrl_ctx for c, d in ct.obs[["cell_type", "donor"]].itertuples(index=False, name=None)])
    test_mask = split.test & has_ctrl
    if (split.test & ~has_ctrl).any():
        print(f"[lodo] {split.name}: {int((split.test & ~has_ctrl).sum())} test conditions skipped (no control group)")
    test = ct.subset(test_mask)
    m = copy.deepcopy(model)
    t0 = time.perf_counter()
    m.fit(train)
    pred = m.predict(test.obs)
    elapsed = time.perf_counter() - t0
    ctrl = train.control_means()
    rows = []
    for i, (p, c, d) in enumerate(test.obs[KEYS].itertuples(index=False, name=None)):
        for mod in ct.modalities:
            base = ctrl[(c, d)][mod]
            r = condition_metrics(pred[mod][i], test.means[mod][i], base, top_k=top_k)
            r.update(model=m.name, split=split.name, modality=mod, perturbation=p, cell_type=c, donor=d,
                     effect_size=delta_norm(test.means[mod][i], base), seconds=elapsed)
            rows.append(r)
    return pd.DataFrame(rows)


def run(models: list[Model], ct: ConditionTable, split_names: list[str], top_k: int = 20,
        split_kwargs: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    frames = []
    for sname in split_names:
        splits = SPLITS[sname](ct, **(split_kwargs or {}).get(sname, {}))
        for split in splits:
            if not split.test.any():
                continue
            for model in models:
                df = evaluate_split(model, ct, split, top_k=top_k)
                df["split_family"] = sname
                frames.append(df)
        if verbose:
            print(f"[lodo] finished split family {sname} ({len(splits)} folds)")
    return pd.concat(frames, ignore_index=True)


def summarise(results: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or ["pearson_delta", "pearson_delta_top20", "mse_top20", "direction_acc_top20"]
    metrics = [m for m in metrics if m in results.columns]
    return (results.groupby(["split_family", "modality", "model"])[metrics]
            .mean().round(3).reset_index())


def skill_vs_baseline(results: pd.DataFrame, metric: str = "mse_top20", baseline: str = "PerturbationMean") -> pd.DataFrame:
    key = ["split", "modality", "perturbation", "cell_type", "donor"]
    b = results[results["model"] == baseline].set_index(key)[metric]
    r = results.set_index(key)
    r = r.assign(skill=1 - r[metric] / b.reindex(r.index).to_numpy()).reset_index()
    return (r.groupby(["split_family", "modality", "model"])["skill"]
            .agg(lambda s: np.nanmean(s.replace([np.inf, -np.inf], np.nan))).round(3).unstack("model"))
