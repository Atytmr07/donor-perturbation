from __future__ import annotations

import argparse
import logging
from pathlib import Path

import py8rds


def assay_to_adata(robj, assay: str, layer: str):
    try:
        return py8rds.seurat2adata(robj, assay=assay, layer=layer)
    except Exception as e:
        logging.warning("seurat2adata(%s, %s) failed (%s); reading matrix + meta.data directly", assay, layer, e)
    names = [str(v) for v in robj.get(["assays", "names"]).value]
    mat = robj.get(["assays", names.index(assay), layer])
    if mat is None:
        raise ValueError(f"assay {assay!r} has no layer {layer!r}")
    ad = py8rds.as_anndata(mat)
    obs = py8rds.as_data_frame(robj.get("meta.data"))
    if py8rds.is_default_index(obs) and obs.shape[0] == ad.shape[0]:
        obs.index = ad.obs_names
    ad.obs = obs.loc[ad.obs_names]
    return ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rds")
    ap.add_argument("--assays", nargs="+", required=True, help="ASSAY or ASSAY:layer")
    ap.add_argument("--layer", default="counts", help="for assays given without a layer")
    ap.add_argument("--no-compress", action="store_true", help="faster writing for the large ATAC assay")
    a = ap.parse_args()

    robj = py8rds.parse_rds(a.rds)
    stem = Path(a.rds).with_suffix("")
    for spec in a.assays:
        assay, _, layer = spec.partition(":")
        layer = layer or a.layer
        ad = assay_to_adata(robj, assay, layer)
        for c in ad.obs.columns:
            if ad.obs[c].dtype == object:
                ad.obs[c] = ad.obs[c].astype(str)
        ad.uns["seurat_layer"] = layer
        out = Path(f"{stem}__{assay}.h5ad")
        ad.write_h5ad(out, compression=None if a.no_compress else "gzip")
        print(f"{assay} [{layer}]: {ad.shape[0]} cells x {ad.shape[1]} features -> {out}", flush=True)


if __name__ == "__main__":
    main()
