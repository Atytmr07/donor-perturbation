from __future__ import annotations

import numpy as np
import pandas as pd

from pertbench.data import CONTEXT, ConditionTable


class Model:
    name = "base"

    def fit(self, train: ConditionTable) -> "Model":
        self.train = train
        self.mods = train.modalities
        self.ctrl = train.control_means()
        nc = ~train.is_control()
        self.tobs = train.obs[nc].reset_index(drop=True)
        d = train.deltas()
        self.tdelta = {m: d[m][nc] for m in self.mods}
        ok = ~np.isnan(self.tdelta[self.mods[0]]).any(axis=1)
        self.tobs = self.tobs[ok].reset_index(drop=True)
        self.tdelta = {m: v[ok] for m, v in self.tdelta.items()}
        self._fit()
        return self

    def _fit(self):
        pass

    def predict(self, query: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {m: [] for m in self.mods}
        for p, ct, dn in query[["perturbation", "cell_type", "donor"]].itertuples(index=False, name=None):
            c = self.ctrl[(ct, dn)]
            d = self._delta(p, ct, dn)
            for m in self.mods:
                out[m].append(c[m] + d[m])
        return {m: np.stack(v) for m, v in out.items()}

    def _delta(self, p: str, ct: str, dn: str) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _zeros(self):
        return {m: np.zeros(self.tdelta[m].shape[1]) for m in self.mods}

    def _mean(self, mask: np.ndarray, w: np.ndarray | None = None):
        if not mask.any():
            return None
        if w is None:
            return {m: self.tdelta[m][mask].mean(axis=0) for m in self.mods}
        w = w / w.sum()
        return {m: w @ self.tdelta[m][mask] for m in self.mods}


class NoChange(Model):
    name = "NoChange"

    def _delta(self, p, ct, dn):
        return self._zeros()


class GlobalMean(Model):
    name = "GlobalMean"

    def _fit(self):
        self.g = self._mean(np.ones(len(self.tobs), bool)) or self._zeros()

    def _delta(self, p, ct, dn):
        return self.g


class PerturbationMean(GlobalMean):
    name = "PerturbationMean"

    def __init__(self, prefer_same_cell_type: bool = True):
        self.prefer = prefer_same_cell_type

    def _delta(self, p, ct, dn):
        isp = (self.tobs["perturbation"] == p).to_numpy()
        if self.prefer:
            r = self._mean(isp & (self.tobs["cell_type"] == ct).to_numpy())
            if r is not None:
                return r
        return self._mean(isp) or self.g


class Additive(GlobalMean):
    name = "Additive"

    def __init__(self, n_iter: int = 10):
        self.n_iter = n_iter

    def _fit(self):
        super()._fit()
        o = self.tobs
        keys = {"perturbation": o["perturbation"].to_numpy(),
                "cell_type": o["cell_type"].to_numpy(),
                "donor": o["donor"].to_numpy()}
        self.eff = {}
        for m in self.mods:
            Y = self.tdelta[m]
            eff = {k: {v: np.zeros(Y.shape[1]) for v in np.unique(a)} for k, a in keys.items()}
            for _ in range(self.n_iter):
                for k, a in keys.items():
                    others = sum(np.stack([eff[j][x] for x in keys[j]]) for j in keys if j != k)
                    R = Y - others
                    for v in eff[k]:
                        eff[k][v] = R[a == v].mean(axis=0)
                    if k != "perturbation":
                        mu = np.mean(list(eff[k].values()), axis=0)
                        for v in eff[k]:
                            eff[k][v] -= mu
                        for v in eff["perturbation"]:
                            eff["perturbation"][v] += mu
            self.eff[m] = eff

    def _delta(self, p, ct, dn):
        out = {}
        for m in self.mods:
            e = self.eff[m]
            a = e["perturbation"].get(p, self.g[m])
            out[m] = a + e["cell_type"].get(ct, 0) + e["donor"].get(dn, 0)
        return out


def _context_features(ctrl: dict, mods: list[str], n_pcs: int):
    keys = list(ctrl)
    X = np.hstack([np.stack([ctrl[k][m] for k in keys]) for m in mods])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    r = min(n_pcs, len(S))
    emb = (Z - Z.mean(0)) @ Vt[:r].T / (S[:r] + 1e-6) * np.sqrt(len(keys))
    return {k: emb[i] for i, k in enumerate(keys)}


class ContextKNN(PerturbationMean):
    name = "ContextKNN"

    def __init__(self, n_pcs: int = 10, bandwidth: float = 1.0, same_cell_type: bool = True):
        super().__init__(prefer_same_cell_type=same_cell_type)
        self.n_pcs, self.bw = n_pcs, bandwidth

    def _fit(self):
        super()._fit()
        self.phi = _context_features(self.ctrl, self.mods, self.n_pcs)
        self.tphi = np.stack([self.phi[k] for k in self.tobs[CONTEXT].itertuples(index=False, name=None)])

    def _delta(self, p, ct, dn):
        isp = (self.tobs["perturbation"] == p).to_numpy()
        same = isp & (self.tobs["cell_type"] == ct).to_numpy()
        if self.prefer and same.any():
            isp = same
        if not isp.any():
            return self.g
        d2 = ((self.tphi[isp] - self.phi[(ct, dn)]) ** 2).sum(1)
        h = self.bw * (np.median(d2) + 1e-6)
        return self._mean(isp, np.exp(-d2 / h))


class ContextRidge(GlobalMean):
    name = "ContextRidge"

    def __init__(self, n_pcs: int = 5, alpha: float = 1.0):
        self.n_pcs, self.alpha = n_pcs, alpha

    def _fit(self):
        super()._fit()
        self.phi = _context_features(self.ctrl, self.mods, self.n_pcs)
        self.cts = sorted({k[0] for k in self.ctrl})
        self.coef = {}
        o = self.tobs
        feats = np.stack([self._feat(ct, dn) for ct, dn in o[CONTEXT].itertuples(index=False, name=None)])
        groups = list(o.groupby("perturbation").indices.items()) + [("__global__", np.arange(len(o)))]
        for p, idx in groups:
            F = feats[idx]
            Fm = F.mean(0)
            Fc = F - Fm
            A = Fc.T @ Fc + self.alpha * len(idx) * np.eye(F.shape[1])
            self.coef[p] = {}
            for m in self.mods:
                Y = self.tdelta[m][idx]
                Ym = Y.mean(0)
                Bm = np.linalg.solve(A, Fc.T @ (Y - Ym))
                self.coef[p][m] = (Ym - Fm @ Bm, Bm)

    def _feat(self, ct, dn):
        oh = np.array([ct == c for c in self.cts], float)
        return np.concatenate([self.phi[(ct, dn)], oh])

    def _delta(self, p, ct, dn):
        f = self._feat(ct, dn)
        c = self.coef.get(p, self.coef["__global__"])
        return {m: c[m][0] + f @ c[m][1] for m in self.mods}


class LinearEmbedding(Model):
    name = "LinearEmbedding"

    def __init__(self, n_pcs: int = 10, alpha: float = 0.1, target_col: str = "target_gene"):
        self.n_pcs, self.alpha, self.target_col = n_pcs, alpha, target_col

    def _fit(self):
        meta = self.train.pert_meta
        self.target = dict(zip(meta["perturbation"], meta[self.target_col])) if meta is not None else {}
        perts = sorted(self.tobs["perturbation"].unique())
        pm = {m: np.stack([self.tdelta[m][(self.tobs["perturbation"] == p).to_numpy()].mean(0) for p in perts])
              for m in self.mods}
        Yr = pm["rna"].T
        U, S, _ = np.linalg.svd(Yr - Yr.mean(1, keepdims=True), full_matrices=False)
        r = min(self.n_pcs, len(S))
        self.gene_emb = U[:, :r] * S[:r]
        self.gidx = {g: i for i, g in enumerate(self.train.var["rna"])}
        P = np.stack([self._emb(p) for p in perts])
        self.mean = {m: pm[m].mean(0) for m in self.mods}
        Pm = P.mean(0)
        A = (P - Pm).T @ (P - Pm) + self.alpha * np.eye(r) * (S[0] ** 2 if len(S) else 1)
        self.Pm = Pm
        self.W = {m: np.linalg.solve(A, (P - Pm).T @ (pm[m] - self.mean[m])) for m in self.mods}

    def _emb(self, p):
        g = self.target.get(p)
        i = self.gidx.get(g) if g is not None else None
        return self.gene_emb[i] if i is not None else np.zeros(self.gene_emb.shape[1])

    def _delta(self, p, ct, dn):
        e = self._emb(p) - self.Pm
        return {m: self.mean[m] + e @ self.W[m] for m in self.mods}


class CrossModal(Model):
    def __init__(self, base: Model, alpha: float = 10.0):
        self.base, self.alpha = base, alpha
        self.name = f"CrossModal[{base.name}]"

    def _fit(self):
        self.base.fit(self.train)
        X = self.tdelta["rna"]
        Xm = X.mean(0)
        A = (X - Xm).T @ (X - Xm) + self.alpha * np.eye(X.shape[1])
        self.Xm = Xm
        self.maps = {}
        for m in self.mods:
            if m == "rna":
                continue
            Y = self.tdelta[m]
            self.maps[m] = (Y.mean(0), np.linalg.solve(A, (X - Xm).T @ (Y - Y.mean(0))))

    def _delta(self, p, ct, dn):
        d = self.base._delta(p, ct, dn)
        out = {"rna": d["rna"]}
        for m, (b, W) in self.maps.items():
            out[m] = b + (d["rna"] - self.Xm) @ W
        return out


def default_models(modalities: list[str] = ("rna",)) -> list[Model]:
    ms = [NoChange(), GlobalMean(), PerturbationMean(), Additive(), ContextKNN(), ContextRidge()]
    if "rna" in modalities:
        ms.append(LinearEmbedding())
        if len(modalities) > 1:
            ms.append(CrossModal(PerturbationMean()))
    return ms
