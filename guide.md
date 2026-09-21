# Reading the canonical study

1. Read `configs/study.yaml` for all experimental choices fixed before evaluation.
2. Read `src/graph.py` for the binary synthetic label rule and BRACS provenance checks.
3. Follow `experiments/synthetic/run.py`, then `experiments/bracs/run.py`.
4. Read `results/report.md`; inspect aggregate rows in `results/summary.csv` before
   patient/replicate observation rows. Probe, TCAV and GCIA use the same frozen model.
5. Check `results/run_manifest.json` for source/data hashes and complete test logs.

Historical results are preserved and are not the canonical full study. There is no
confirmatory histology claim and no biological causality claim. Missing annotations
are recorded as blockers rather than replaced by generated pathology labels.
