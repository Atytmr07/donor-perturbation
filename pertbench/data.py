from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KEYS = ["perturbation", "cell_type", "donor"]
CONTEXT = ["cell_type", "donor"]


@dataclass
class PerturbData:
    X: dict[str, np.ndarray]
    obs: pd.DataFrame
    var: dict[str, list[str]]
    control: str = "control"
    pert_meta: pd.DataFrame | None = None

    def __post_init__(self):
        missing = [k for k in KEYS if k not in self.obs.columns]
        if missing:
            raise ValueError(f"obs is missing columns {missing}")
        n = len(self.obs)
        for m, x in self.X.items():
            if x.shape[0] != n:
                raise ValueError(f"modality {m} has {x.shape[0]} rows, obs has {n}")
            if len(self.var[m]) != x.shape[1]:
                raise ValueError(f"modality {m}: var length != n_features")
        self.obs = self.obs.reset_index(drop=True)
        for k in KEYS:
            self.obs[k] = self.obs[k].astype(str)

    @property
    def modalities(self) -> list[str]:
        return list(self.X)

    def summary(self) -> str:
        o = self.obs
        parts = [f"{len(o)} cells"]
        parts += [f"{o[k].nunique()} {k}s" for k in KEYS]
        parts += [f"{m}: {x.shape[1]} features" for m, x in self.X.items()]
        return ", ".join(parts)


@dataclass
class ConditionTable:
    obs: pd.DataFrame
    means: dict[str, np.ndarray]
    var: dict[str, list[str]]
    control: str = "control"
    pert_meta: pd.DataFrame | None = None
    _index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.obs = self.obs.reset_index(drop=True)
        self._index = {tuple(r): i for i, r in enumerate(self.obs[KEYS].itertuples(index=False, name=None))}

    @property
    def modalities(self) -> list[str]:
        return list(self.means)

    def row(self, pert: str, cell_type: str, donor: str) -> int | None:
        return self._index.get((pert, cell_type, donor))

    def is_control(self) -> np.ndarray:
        return (self.obs["perturbation"] == self.control).to_numpy()

    def subset(self, mask: np.ndarray) -> "ConditionTable":
        mask = np.asarray(mask, dtype=bool)
        return ConditionTable(
            obs=self.obs[mask].copy(),
            means={m: v[mask] for m, v in self.means.items()},
            var=self.var,
            control=self.control,
            pert_meta=self.pert_meta,
        )

    def control_means(self) -> dict[tuple[str, str], dict[str, np.ndarray]]:
        out = {}
        ctrl = self.obs[self.is_control()]
        for i, r in ctrl.iterrows():
            out[(r["cell_type"], r["donor"])] = {m: v[i] for m, v in self.means.items()}
        return out

    def deltas(self) -> dict[str, np.ndarray]:
        cm = self.control_means()
        out = {}
        for m, v in self.means.items():
            d = np.full_like(v, np.nan)
            for i, (ct, dn) in enumerate(self.obs[CONTEXT].itertuples(index=False, name=None)):
                c = cm.get((ct, dn))
                if c is not None:
                    d[i] = v[i] - c[m]
            out[m] = d
        return out


def pseudobulk(data: PerturbData, min_cells: int = 5) -> ConditionTable:
    g = data.obs.groupby(KEYS, sort=True, observed=True).indices
    keys, rows = [], []
    for k, idx in g.items():
        if len(idx) >= min_cells:
            keys.append((*k, len(idx)))
            rows.append(idx)
    obs = pd.DataFrame(keys, columns=KEYS + ["n_cells"])
    means = {m: np.stack([x[idx].mean(axis=0) for idx in rows]) for m, x in data.X.items()}
    return ConditionTable(obs=obs, means=means, var=data.var, control=data.control, pert_meta=data.pert_meta)


def merge_tables(tables: dict[str, ConditionTable]) -> ConditionTable:
    keysets = [set(map(tuple, t.obs[KEYS].itertuples(index=False, name=None))) for t in tables.values()]
    common = sorted(set.intersection(*keysets))
    if not common:
        raise ValueError("no (perturbation, cell_type, donor) condition is shared by all tables")
    counts = {}
    for t in tables.values():
        for m in t.modalities:
            counts[m] = counts.get(m, 0) + 1
    obs = pd.DataFrame(common, columns=KEYS)
    means, var, ncells = {}, {}, []
    for name, t in tables.items():
        rows = [t.row(*k) for k in common]
        ncells.append(t.obs["n_cells"].to_numpy()[rows])
        for m in t.modalities:
            new = f"{name}_{m}" if counts[m] > 1 else m
            means[new] = t.means[m][rows]
            var[new] = t.var[m]
    obs["n_cells"] = np.minimum.reduce(ncells)
    first = next(iter(tables.values()))
    return ConditionTable(obs=obs, means=means, var=var, control=first.control, pert_meta=first.pert_meta)


def pseudobulk_sparse(X, obs: pd.DataFrame, min_cells: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    import scipy.sparse as sp
    g = obs.groupby(KEYS, sort=True, observed=True).indices
    keys, idx = [], []
    for k, ii in g.items():
        if len(ii) >= min_cells:
            keys.append((*k, len(ii)))
            idx.append(ii)
    rows = np.concatenate(idx)
    cols = np.repeat(np.arange(len(idx)), [len(i) for i in idx])
    w = np.concatenate([np.full(len(i), 1.0 / len(i)) for i in idx])
    A = sp.csr_matrix((w, (cols, rows)), shape=(len(idx), X.shape[0]))
    M = A @ X
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    return pd.DataFrame(keys, columns=KEYS + ["n_cells"]), M.astype(np.float32)


def _dense(x) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def _normalise_counts(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log1p_cp10k":
        lib = x.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1
        return np.log1p(x / lib * 1e4)
    if kind == "clr":
        lx = np.log1p(x)
        return lx - lx.mean(axis=1, keepdims=True)
    if kind == "tfidf":
        tf = x / np.maximum(x.sum(axis=1, keepdims=True), 1)
        idf = np.log1p(x.shape[0] / np.maximum((x > 0).sum(axis=0, keepdims=True), 1))
        return np.log1p(tf * idf * 1e4)
    if kind == "none":
        return x
    raise ValueError(kind)


def from_anndata(
    modalities: dict,
    perturbation_key: str,
    control_value: str,
    cell_type_key: str | None = None,
    donor_key: str | None = None,
    normalise: dict[str, str] | None = None,
    n_top_features: dict[str, int] | None = None,
) -> PerturbData:
    first = next(iter(modalities.values()))
    o = first.obs
    obs = pd.DataFrame({
        "perturbation": o[perturbation_key].astype(str).replace({control_value: "control"}).to_numpy(),
        "cell_type": o[cell_type_key].astype(str).to_numpy() if cell_type_key else "all",
        "donor": o[donor_key].astype(str).to_numpy() if donor_key else "all",
    })
    X, var = {}, {}
    for name, ad in modalities.items():
        x = _dense(ad.X)
        kind = (normalise or {}).get(name, "none")
        x = _normalise_counts(x, kind)
        k = (n_top_features or {}).get(name)
        names = list(map(str, ad.var_names))
        if k and k < x.shape[1]:
            keep = np.argsort(-x.var(axis=0))[:k]
            keep.sort()
            x, names = x[:, keep], [names[i] for i in keep]
        X[name], var[name] = x.astype(np.float32), names
    return PerturbData(X=X, obs=obs, var=var)
