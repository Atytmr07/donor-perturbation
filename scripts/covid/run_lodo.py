from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from pertbench.datasets import covid_vaccine
from pertbench.models import default_models
from pertbench.runner import evaluate_split, skill_vs_baseline, summarise
from pertbench.splits import unseen_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strict", action="store_true", help="feature selection on training donors only (slower)")
    ap.add_argument("--out", default=str(ROOT / "results" / "covid_lodo"))
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    load = lambda train=None: covid_vaccine(train_donors=train, **cfg)

    ct = load()
    donors = sorted(ct.obs["donor"].unique())
    print(f"[covid] {len(ct.obs)} conditions | donors {donors} | modalities "
          + ", ".join(f"{m}:{v.shape[1]}" for m, v in ct.means.items()))
    print(ct.obs.groupby(["perturbation"])["n_cells"].describe()[["count", "min", "50%"]])

    frames = []
    for d in donors:
        fold_ct = load([x for x in donors if x != d]) if a.strict else ct
        split = unseen_context(fold_ct, "donor", d)
        for model in default_models(fold_ct.modalities):
            df = evaluate_split(model, fold_ct, split)
            df["split_family"] = "unseen_donor"
            frames.append(df)
        print(f"[covid] fold donor={d} done")
    res = pd.concat(frames, ignore_index=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "per_condition.csv", index=False)
    summ = summarise(res)
    summ.to_csv(out / "summary.csv", index=False)
    by_day = res.groupby(["modality", "perturbation", "model"])["mse_top20"].mean().unstack("model").round(4)
    by_day.to_csv(out / "mse_top20_by_day.csv")
    skill = skill_vs_baseline(res)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summ.to_string(index=False))
        print("\nTop-20 MSE by day:\n", by_day.to_string())
        print("\nSkill vs PerturbationMean (>0 is better):\n", skill.to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
