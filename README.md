# GCIA

Does concept decodability imply functional reliance in histopathology graph neural networks?

The canonical study is **Probe -> TCAV -> GCIA**, with three-layer DeepSets and GraphSAGE,
seeds 11/23/37, and one configuration: `configs/study.yaml`.

From this directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe experiments/synthetic/run.py --verify-only
.\.venv\Scripts\python.exe experiments/synthetic/run.py
.\.venv\Scripts\python.exe experiments/bracs/run.py
.\.venv\Scripts\python.exe experiments/synthetic/run.py --verify-only
```

`src/graph.py` constructs and validates data; `models.py` defines forward passes and
layer representations; `probes.py` fits train-only logistic/Ridge probes; `tcav.py`
computes final-layer directional sensitivity; `interventions.py` implements feature,
trusted-population and relation interventions; `evaluation.py` contains metrics,
patient/replicate uncertainty, validity interpretation, provenance and reporting.
Training and experiment orchestration live in the two canonical experiment runners.

All canonical numeric results are in `results/summary.csv` (aggregate and observation
rows distinguished by `record_type`). Read `results/report.md` for the scientific
findings and `results/run_manifest.json` for configuration, data/source hashes,
selected epochs, weight hashes, software versions and test results. Rerunning a stage
replaces that stage's canonical rows. No checkpoints or figures are emitted.

Synthetic ground truth uses noisy binary OR of C_used and C_relational, with visible
C_unused absent from the label rule. Generator truth is distinct from measured
encoding and learned dependence. Controls match feature or edge edit budgets.

BRACS reuses checksummed graphs in `../GCIA_archive/legacy/reports/cell_cohort`.
The only available concept is mean segmented nuclear area in pixels squared, an
explicit-feature technical positive control. Unvalidated segmentation, unknown MPP,
missing independent concept/cell-type annotations and previously explored test
patients limit this to development/exploratory analysis.

Never claim biological causality. Interpret dependence as **model-level functional
reliance under the tested intervention**. Small excess alone does not establish
negligible use: the target-effect interval must also meet the fixed margin.

Historical files remain in `../GCIA_archive/legacy`, `results/synthetic_v1`, and
`experiments/experiments`. Those runners/configurations are noncanonical. The small
`experiments/synthetic/data.py` shim and old helper defaults retain historical test
compatibility; the new study explicitly selects binary outputs and its own generator.
