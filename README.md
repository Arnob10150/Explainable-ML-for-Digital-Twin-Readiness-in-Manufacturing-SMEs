# GitHub repository title

**Explainable ML for Digital Twin Readiness in Manufacturing SMEs**

## Repository description

Reproducible 12-criterion Fuzzy-AHP/TOPSIS and machine-learning pipeline for assessing Digital Twin readiness in manufacturing SMEs. Includes a SHAP importance audit, outcome-supervised weight learning, robustness and clustering analyses, cost-aware intervention ranking, and figure/table generation.

## Project structure

```text
scripts/
  run_pipeline.py              # analysis from source surveys to results
  run_all.py                   # analysis plus figures and LaTeX tables
  run_external_validation.py   # optional independent-country replication
src/dtreadiness/
  config.py                     # 12-criterion taxonomy and data locations
  data.py                       # survey loading and construct scores
  fuzzy.py                      # Fuzzy-AHP, TOPSIS, VIKOR, and BWM methods
  ml.py                         # classifiers and SHAP importance
  hybrid_weights.py             # nested-CV outcome-supervised weights
  analysis.py                   # sensitivity, clustering, gaps, cost analysis
  validity.py                   # validity and common-method checks
  psychometrics.py              # reliability and factorability measures
  figures.py, tables.py         # research figures and result tables
data/raw/                       # source survey workbooks (not included)
data/processed/                 # generated analysis outputs
reports/figures/                # generated figures
reports/tables/                 # generated LaTeX table fragments
```

## Data

The source workbooks are not bundled. Obtain the datasets cited as [20] and [21] in the conference paper and place them at:

```text
data/raw/d1/Digital-Twin Model Adoption in Malaysian Semicondusctor Industry.xlsx
data/raw/d2/Raw data.xlsx
```

Alternatively, set `DT_READINESS_DATA_DIR` to a directory containing `d1/` and `d2/` with those filenames. Keep downloaded source data out of Git unless its terms explicitly permit redistribution.

The optional external replication uses the dynamic-capabilities survey cited as [24]. Place its workbook at `data/raw/external/Survey data for DT study.xlsx`, or set `DT_READINESS_EXTERNAL_DATA` to its location, then run `python scripts/run_external_validation.py`.

## Install and reproduce

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
```

The pipeline writes machine-readable outputs to `data/processed/`, figures to `reports/figures/`, and LaTeX table fragments to `reports/tables/`. The project fixes random seeds at 42 where applicable.

## Citation and authorship

Please cite the conference paper and the original datasets when reusing this code or its results. Repository ownership, author list, and license should be set by the paper authors before publication.
