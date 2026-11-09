# donor-perturbation

Predicting the perturbation response of an unseen donor from its untreated cells, evaluated leave-one-donor-out.

Data: COVID vaccine dataset from Zhang et al. 2023 (Nat Immunol), https://zenodo.org/records/7555405.
CITE-seq (RNA + ADT) and ASAP-seq (ATAC + ADT), 6 donors, days 0 / 2 / 11 / 28.

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

`pertbench.data.from_anndata` builds the same input from an AnnData object with perturbation, cell type and donor columns. Models are in `pertbench/models.py`, metrics in `pertbench/metrics.py`.

## CellFlow

`scripts/cellflow/synthetic_lodo.py` runs CellFlow (https://github.com/theislab/cellflow) leave-one-donor-out on synthetic data. It needs Python 3.11 and a separate environment with CellFlow installed.
