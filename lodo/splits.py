from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from lodo.data import ConditionTable


@dataclass
class Split:
    name: str
    train: np.ndarray
    test: np.ndarray


def _make(ct: ConditionTable, name: str, test_nonctrl: np.ndarray) -> Split:
    ctrl = ct.is_control()
    test = test_nonctrl & ~ctrl
    return Split(name, train=~test, test=test)


def random_split(ct: ConditionTable, frac: float = 0.2, seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    return _make(ct, f"random_{frac}", rng.random(len(ct.obs)) < frac)


def unseen_perturbation(ct: ConditionTable, frac: float = 0.2, seed: int = 0) -> Split:
    perts = np.array(sorted(set(ct.obs["perturbation"]) - {ct.control}))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(perts, max(1, int(round(frac * len(perts)))), replace=False))
    return _make(ct, "unseen_perturbation", ct.obs["perturbation"].isin(held).to_numpy())


def unseen_context(ct: ConditionTable, column: str, value: str, k_shot: int = 0, seed: int = 0) -> Split:
    in_ctx = (ct.obs[column] == value).to_numpy()
    test = in_ctx.copy()
    if k_shot:
        perts = sorted(set(ct.obs.loc[in_ctx, "perturbation"]) - {ct.control})
        rng = np.random.default_rng(seed)
        shown = set(rng.choice(perts, min(k_shot, len(perts)), replace=False))
        test &= ~ct.obs["perturbation"].isin(shown).to_numpy()
    return _make(ct, f"unseen_{column}={value}" + (f"_k{k_shot}" if k_shot else ""), test)


def leave_one_out(ct: ConditionTable, column: str, k_shot: int = 0, seed: int = 0) -> Iterator[Split]:
    for v in sorted(ct.obs[column].unique()):
        yield unseen_context(ct, column, v, k_shot=k_shot, seed=seed)


SPLITS = {
    "random": lambda ct, **kw: [random_split(ct, **kw)],
    "unseen_perturbation": lambda ct, **kw: [unseen_perturbation(ct, **kw)],
    "unseen_cell_type": lambda ct, **kw: list(leave_one_out(ct, "cell_type", **kw)),
    "unseen_donor": lambda ct, **kw: list(leave_one_out(ct, "donor", **kw)),
}
