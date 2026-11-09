import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pertbench import make_synthetic, pseudobulk
from pertbench.metrics import condition_metrics
from pertbench.models import (ContextRidge, NoChange, PerturbationMean,
                              default_models)
from pertbench.runner import evaluate_split, run, summarise
from pertbench.splits import SPLITS, unseen_context, unseen_perturbation


@pytest.fixture(scope="module")
def ct():
    return pseudobulk(make_synthetic(n_genes=60, n_proteins=10, n_peaks=20, n_perts=10,
                                     n_cell_types=2, n_donors=5, cells_per_condition=15, seed=1))


def test_pseudobulk_shapes(ct):
    assert len(ct.obs) == 11 * 2 * 5
    assert ct.means["rna"].shape == (110, 60)
    assert ct.is_control().sum() == 10


def test_splits_keep_controls_in_train(ct):
    for name, f in SPLITS.items():
        for s in f(ct):
            assert not (s.test & s.train).any()
            assert s.train[ct.is_control()].all(), name
            assert s.test.any(), name


def test_unseen_donor_hides_whole_donor(ct):
    s = unseen_context(ct, "donor", "donor0")
    test_donors = set(ct.obs.loc[s.test, "donor"])
    assert test_donors == {"donor0"}
    train_nc = s.train & ~ct.is_control()
    assert "donor0" not in set(ct.obs.loc[train_nc, "donor"])


def test_unseen_perturbation_disjoint(ct):
    s = unseen_perturbation(ct, frac=0.3)
    tr = set(ct.obs.loc[s.train, "perturbation"]) - {"control"}
    te = set(ct.obs.loc[s.test, "perturbation"])
    assert tr.isdisjoint(te)


def test_metrics_perfect_prediction():
    rng = np.random.default_rng(0)
    c, t = rng.normal(size=50), rng.normal(size=50)
    m = condition_metrics(t, t, c)
    assert m["mse"] == 0 and m["pearson_delta"] == pytest.approx(1) and m["direction_acc_top20"] == 1


def test_nochange_has_zero_delta(ct):
    df = evaluate_split(NoChange(), ct, unseen_context(ct, "donor", "donor1"))
    assert np.isnan(df["pearson_delta"]).all()


def test_models_beat_nochange_on_unseen_donor(ct):
    s = unseen_context(ct, "donor", "donor2")
    base = evaluate_split(NoChange(), ct, s)["mse_top20"].mean()
    for m in [PerturbationMean(), ContextRidge()]:
        assert evaluate_split(m, ct, s)["mse_top20"].mean() < base


def test_donor_aware_model_helps_when_donor_interaction_exists():
    ct = pseudobulk(make_synthetic(n_donors=10, donor_interaction=1.0, seed=3))
    res = run([PerturbationMean(), ContextRidge()], ct, ["unseen_donor"], verbose=False)
    s = res[res.modality == "rna"].groupby("model")["mse_top20"].mean()
    assert s["ContextRidge"] < s["PerturbationMean"]


def test_full_run_smoke(ct):
    res = run(default_models(ct.modalities), ct, list(SPLITS), verbose=False)
    summ = summarise(res)
    assert set(summ["modality"]) == {"rna", "protein", "atac"}
    assert not res["mse"].isna().any()


def _fake_covid(tmp_path, seed=0):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    donors, days, cts = [f"D{i}" for i in range(6)], ["0", "2", "10", "28"], ["CD4", "CD8", "Mono"]
    paths = {}
    for tname, mods in {"cite": {"rna": 300, "adt": 20}, "asap": {"atac": 500, "adt": 20}}.items():
        rows = [(d, day, c) for d in donors for day in days for c in cts for _ in range(rng.integers(15, 30))]
        obs = pd.DataFrame(rows, columns=["donor_id", "day", "celltype"])
        obs.index = [f"{tname}_{i}" for i in range(len(obs))]
        paths[tname] = {}
        for m, nf in mods.items():
            lam = rng.gamma(1.0, 1.0, nf)
            eff = 1 + (obs["day"].astype(int).to_numpy()[:, None] / 28) * rng.normal(0, 0.5, nf)
            X = rng.poisson(np.clip(lam * eff, 0.01, None))
            a = ad.AnnData(sp.csr_matrix(X.astype(np.float32)), obs=obs.copy())
            a.var_names = [f"{m}{j}" for j in range(nf)]
            p = tmp_path / f"{tname}_{m}.h5ad"
            a.write_h5ad(p)
            paths[tname][m] = str(p)
    return {"files": paths, "donor_key": "donor_id", "day_key": "day", "cell_type_key": "celltype",
            "control_day": "0", "n_top": {"rna": 100, "atac": 200}}


def test_covid_loader_merges_assays(tmp_path):
    from pertbench.datasets import covid_vaccine
    cfg = _fake_covid(tmp_path)
    ct = covid_vaccine(**cfg)
    assert set(ct.modalities) == {"rna", "cite_adt", "atac", "asap_adt"}
    assert ct.means["rna"].shape[1] == 100 and ct.means["atac"].shape[1] == 200
    assert len(ct.obs) == 6 * 4 * 3 and ct.is_control().sum() == 18
    strict = covid_vaccine(train_donors=[f"D{i}" for i in range(1, 6)], **cfg)
    assert strict.means["rna"].shape == ct.means["rna"].shape


def test_covid_lodo_runs(tmp_path):
    from pertbench.datasets import covid_vaccine
    from pertbench.models import default_models
    ct = covid_vaccine(**_fake_covid(tmp_path))
    df = evaluate_split(default_models(ct.modalities)[2], ct, unseen_context(ct, "donor", "D3"))
    assert set(df["donor"]) == {"D3"} and not df["mse"].isna().any()


def test_distribution_metrics_detect_shift():
    from pertbench.metrics import distribution_metrics
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(400, 10)), rng.normal(size=(400, 10))
    near = distribution_metrics(a, b)
    small = distribution_metrics(a, b + 0.3)
    big = distribution_metrics(a, b + 1.0)
    for k in ("mmd", "energy"):
        assert abs(near[k]) < small[k] < big[k], k
    wide = distribution_metrics(a, b * 2.0)
    assert wide["energy"] > small["energy"] * 0.5 and wide["mmd"] > near["mmd"]


def test_covid_loader_per_table_keys_and_norm(tmp_path):
    import anndata as ad
    from pertbench.datasets import covid_vaccine
    cfg = _fake_covid(tmp_path)
    for m, p in cfg["files"]["asap"].items():
        a = ad.read_h5ad(p)
        a.obs = a.obs.rename(columns={"celltype": "predicted_l1"})
        if m == "adt":
            x = a.X.toarray()
            a.X = (x - x.mean(0)) / (x.std(0) + 1e-6)
        a.write_h5ad(p)
    cfg["cell_type_key"] = {"cite": "celltype", "asap": "predicted_l1"}
    ct = covid_vaccine(norm={"asap_adt": "none"}, **cfg)
    assert set(ct.modalities) == {"rna", "cite_adt", "atac", "asap_adt"}
    assert len(ct.obs) == 6 * 4 * 3
    assert abs(ct.means["asap_adt"].mean()) < 0.5


def test_evaluate_skips_conditions_without_control(ct):
    from pertbench.models import PerturbationMean
    drop = ct.is_control() & (ct.obs["cell_type"] == "ct0").to_numpy() & (ct.obs["donor"] == "donor1").to_numpy()
    sub = ct.subset(~drop)
    df = evaluate_split(PerturbationMean(), sub, unseen_context(sub, "donor", "donor1"))
    assert len(df) and "ct0" not in set(df["cell_type"])


def test_covid_loader_fixed_feature_list(tmp_path):
    from pertbench.datasets import covid_vaccine
    cfg = _fake_covid(tmp_path)
    wanted = ["atac3", "atac7", "atac42", "not_a_peak"]
    f = tmp_path / "peaks.txt"
    f.write_text("\n".join(wanted) + "\n", encoding="utf-8")
    for spec in (str(f), wanted):
        ct = covid_vaccine(features={"atac": spec}, **cfg)
        assert ct.var["atac"] == ["atac3", "atac7", "atac42"]
        strict = covid_vaccine(features={"atac": spec}, train_donors=["D1", "D2", "D3", "D4", "D5"], **cfg)
        assert strict.var["atac"] == ["atac3", "atac7", "atac42"]
