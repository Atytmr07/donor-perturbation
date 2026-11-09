from __future__ import annotations

import json
import sys
from pathlib import Path

import py8rds


def names(robj):
    n = robj.get("names") if robj is not None else None
    return [str(v) for v in n.value] if n is not None and n.value is not None else []


def slot_names(robj):
    return [a[0] for a in getattr(robj, "attributes", []) if isinstance(a, (list, tuple)) and isinstance(a[0], str)]


def main(path: str):
    r = py8rds.parse_rds(path)
    out = {"file": path, "object_slots": slot_names(r), "assays": {}, "reductions": names(r.get("reductions"))}
    for i, a in enumerate(names(r.get("assays"))):
        assay = r.get(["assays", i])
        info = {"slots": slot_names(assay)}
        for layer in ("counts", "data", "scale.data"):
            m = assay.get(layer) if assay is not None else None
            dim = m.get("Dim") if m is not None else None
            if dim is not None and dim.value is not None:
                info[layer] = [int(x) for x in dim.value]
        layers = assay.get("layers") if assay is not None else None
        if layers is not None:
            info["layers(Assay5)"] = names(layers)
        out["assays"][a] = info
    meta = py8rds.as_data_frame(r.get("meta.data"))
    out["n_cells"] = int(meta.shape[0])
    out["meta"] = {}
    for c in meta.columns:
        s = meta[c]
        nun = int(s.nunique())
        out["meta"][c] = {"dtype": str(s.dtype), "n_unique": nun,
                          "examples": [str(v) for v in s.astype(str).value_counts().index[:12]] if nun <= 300 else []}
    dest = Path(path).with_suffix(".inspect.json")
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False)[:12000])
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main(sys.argv[1])
