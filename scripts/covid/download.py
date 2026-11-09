from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "covid"
API = "https://zenodo.org/api/records/7555405"
CORE = ["PBMC_vaccine_CITE.rds", "PBMC_vaccine_ASAP.rds", "antigen_module_genes.rds", "antigen_module_peaks.rds"]
FRAG = ["PBMC_vaccine_ASAP_fragments.tsv.gz", "PBMC_vaccine_ASAP_fragments.tsv.gz.tbi"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, size: int):
    have = dest.stat().st_size if dest.exists() else 0
    if have == size:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r, open(dest, "ab" if have else "wb") as f:
        done = have
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {dest.name}: {done / 1e9:6.2f} / {size / 1e9:.2f} GB", end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = {f["key"]: f for f in json.load(urllib.request.urlopen(API))["files"]}
    for name in CORE + (FRAG if a.fragments else []):
        f = files[name]
        dest = OUT / name
        print(f"{name} ({f['size'] / 1e9:.2f} GB)")
        fetch(f["links"]["self"], dest, f["size"])
        want = f["checksum"].split(":", 1)[1]
        got = md5(dest)
        print("  md5", "OK" if got == want else f"MISMATCH ({got} != {want}); delete the file and rerun")


if __name__ == "__main__":
    main()
