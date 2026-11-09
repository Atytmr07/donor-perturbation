from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sparse_norm_select(X, kind: str, n_top: int | None):
    import scipy.sparse as sp
    X = sp.csr_matrix(X, dtype=np.float32)
    if kind == "binary_log1p_cp10k":
        X.data[:] = 1.0
        kind = "log1p_cp10k"
    if kind == "log1p_cp10k":
        lib = np.asarray(X.sum(axis=1)).ravel()
        lib[lib == 0] = 1
        X = sp.diags(1e4 / lib) @ X
        X.data = np.log1p(X.data)
    elif kind == "clr":
        Xd = np.log1p(X.toarray())
        X = sp.csr_matrix(Xd - Xd.mean(axis=1, keepdims=True))
    elif kind != "none":
        raise ValueError(kind)
    keep = None
    if n_top and n_top < X.shape[1]:
        mean = np.asarray(X.mean(axis=0)).ravel()
        var = np.asarray(X.multiply(X).mean(axis=0)).ravel() - mean ** 2
        keep = np.sort(np.argsort(-var)[:n_top])
        X = X[:, keep]
    return X, keep


def _feature_list(spec) -> set[str]:
    if isinstance(spec, (str, Path)):
        return {ln.strip() for ln in Path(spec).read_text(encoding="utf-8").splitlines() if ln.strip()}
    return set(map(str, spec))


def covid_vaccine(files: dict, donor_key: str, day_key: str, cell_type_key: str | dict, control_day: str,
                  norm: dict | None = None, n_top: dict | None = None, min_cells: int = 10,
                  cell_type_map: dict | None = None, train_donors: list[str] | None = None,
                  features: dict | None = None):
    import anndata as ad
    from pertbench.data import ConditionTable, merge_tables, pseudobulk_sparse
    norm = {"rna": "log1p_cp10k", "adt": "clr", "atac": "binary_log1p_cp10k", **(norm or {})}
    n_top = {"rna": 3000, "atac": 20000, **(n_top or {})}
    tables = {}
    for tname, mods in files.items():
        means, var, obs0 = {}, {}, None
        ct_key = cell_type_key[tname] if isinstance(cell_type_key, dict) else cell_type_key
        for m, path in mods.items():
            kind = norm.get(f"{tname}_{m}", norm[m])
            top = n_top.get(f"{tname}_{m}", n_top.get(m))
            a = ad.read_h5ad(path)
            o = a.obs
            ct = o[ct_key].astype(str)
            if cell_type_map:
                ct = ct.map(lambda s: cell_type_map.get(s, s))
            day = o[day_key].astype(str)
            label = day.where(~day.str.fullmatch(r"\d+"), "day" + day)
            obs = pd.DataFrame({"perturbation": np.where(day == str(control_day), "control", label),
                                "cell_type": ct.to_numpy(), "donor": o[donor_key].astype(str).to_numpy()})
            names = np.asarray(a.var_names.astype(str))
            fixed = (features or {}).get(f"{tname}_{m}", (features or {}).get(m))
            if fixed is not None:
                wanted = _feature_list(fixed)
                X, _ = _sparse_norm_select(a.X, kind, None)
                keep = np.array([i for i, n in enumerate(names) if n in wanted])
                if len(keep) == 0:
                    raise ValueError(f"{tname}_{m}: none of the given features are in {path}")
                X = X[:, keep]
            elif train_donors is None:
                X, keep = _sparse_norm_select(a.X, kind, top)
            else:
                in_train = obs["donor"].isin(train_donors).to_numpy()
                _, keep = _sparse_norm_select(a.X[in_train], kind, top)
                X, _ = _sparse_norm_select(a.X, kind, None)
                if keep is not None:
                    X = X[:, keep]
            names = names[keep] if keep is not None else names
            pb_obs, M = pseudobulk_sparse(X, obs, min_cells=min_cells)
            if obs0 is None:
                obs0 = pb_obs
            elif not obs0[["perturbation", "cell_type", "donor"]].equals(pb_obs[["perturbation", "cell_type", "donor"]]):
                raise ValueError(f"{tname}: modalities do not share cells/metadata")
            means[m], var[m] = M, list(names)
        tables[tname] = ConditionTable(obs=obs0, means=means, var=var)
    return merge_tables(tables) if len(tables) > 1 else next(iter(tables.values()))


