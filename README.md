# donor-perturbation

Data: COVID vaccine dataset from Zhang et al. 2023 (Nat Immunol), https://zenodo.org/records/7555405.
CITE-seq (RNA + ADT) and ASAP-seq (ATAC + ADT), 6 donors, days 0 / 2 / 11 / 28.

So far:
- The COVID data is converted to h5ad and compared against simple baselines (e.g. the mean response of the training donors, as in Ahlmann-Eltze et al. 2025).
- CellFlow (https://github.com/theislab/cellflow) has only been run on a small synthetic dataset, not on the COVID data yet.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install git+https://github.com/cellgeni/py8rds.git
```

## COVID data

```bash
python scripts/covid/download.py
python scripts/covid/convert.py data/covid/PBMC_vaccine_CITE.rds --assays RNA ADT
python scripts/covid/convert.py data/covid/PBMC_vaccine_ASAP.rds --assays ATAC ADT:scale.data
python scripts/covid/run_lodo.py --config scripts/covid/config.json --strict
```

`inspect_rds.py` prints the assays and metadata columns of an .rds file.

## Other datasets

`lodo.data.from_anndata` builds the same input from an AnnData object with perturbation, cell type and donor columns. Baselines are in `lodo/models.py`, metrics in `lodo/metrics.py`.

## CellFlow

`scripts/cellflow/synthetic_lodo.py` runs CellFlow leave-one-donor-out on synthetic data. It needs Python 3.11 and a separate environment with CellFlow installed.
