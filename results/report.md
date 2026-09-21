# Canonical GCIA study

Question: does concept decodability imply functional reliance in histopathology GNNs?

**Scope: development/exploratory; no biological causality.** The only supported dependence wording is model-level functional reliance under the tested intervention. No test-based tuning was performed.

Generator ground truth: binary OR of C_used and C_relational with independent label flips. C_unused is visible but absent from the label function. Eight factorial variants share a replicate; paired relations have identical features and node counts. This construction is not evidence of learned use.

## Answers to the ten study questions

1. **Encoded and used:** 5/6 model-seed runs meet both criteria.
2. **Encoded but unused:** 1/6 model-seed runs support negligible effect under the tested intervention; other runs are inconclusive.
3. **Relational use:** 3/3 model-seed runs GraphSAGE runs show excess effects above the prespecified margin.
4. **No-edge control:** 3/3 model-seed runs DeepSets runs have exactly zero edge-only response.
5. **Valid BRACS concept:** measured mean segmented nuclear area in pixel squared units, a technical positive control already explicit in node features. No independent histological concept is validated.
6. **BRACS encoding:** 6/6 model-seed runs pass held-out R2 and patient-group permutation criteria.
7. **BRACS TCAV:** see final-layer values below. Values are descriptive directional sensitivities, not significance or reliance tests.
8. **BRACS GCIA:** 0/6 model-seed runs show supported excess area-coordinate response; see validity and confidence intervals below.
9. **Unsupported:** Concept 2 and Concept 3 are BLOCKED: requires independent concept annotations; relational histology also requires trustworthy cell/compartment labels. MPP and independent BRACS segmentation validation are missing.
10. **Exploratory:** historical BRACS test patients were already explored; there is no untouched formally locked confirmatory test set.

## Frozen-model results

| Stage | Model | Seed | Concept | Probe | Encoded | TCAV | Target TV | Control TV | Excess [CI] | Valid | Interpretation |
|---|---|---:|---|---:|---|---:|---:|---:|---|---|---|
| synthetic | DeepSets | 11 | C_used | 1.000 | True | 1.000 | 0.3791 | 0.3727 | 0.0064 [-0.0015, 0.0136] | True | inconclusive |
| synthetic | DeepSets | 11 | C_unused | 1.000 | True | 0.000 | 0.0195 | 0.0581 | -0.0386 [-0.0428, -0.0340] | True | inconclusive |
| synthetic | DeepSets | 11 | C_relational | 0.500 | False | nan | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | True | negligible effect under the tested intervention |
| synthetic | DeepSets | 23 | C_used | 1.000 | True | 1.000 | 0.4098 | 0.1201 | 0.2896 [0.2877, 0.2915] | True | model-level functional reliance under the tested intervention |
| synthetic | DeepSets | 23 | C_unused | 1.000 | True | 1.000 | 0.0060 | 0.0059 | 0.0001 [-0.0005, 0.0007] | True | negligible effect under the tested intervention |
| synthetic | DeepSets | 23 | C_relational | 0.500 | False | nan | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | True | negligible effect under the tested intervention |
| synthetic | DeepSets | 37 | C_used | 1.000 | True | 1.000 | 0.4547 | 0.1293 | 0.3254 [0.3229, 0.3278] | True | model-level functional reliance under the tested intervention |
| synthetic | DeepSets | 37 | C_unused | 1.000 | True | 1.000 | 0.0288 | 0.0338 | -0.0051 [-0.0064, -0.0038] | True | inconclusive |
| synthetic | DeepSets | 37 | C_relational | 0.500 | False | nan | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | True | negligible effect under the tested intervention |
| synthetic | GraphSAGE | 11 | C_used | 1.000 | True | 1.000 | 0.3939 | 0.0892 | 0.3047 [0.2870, 0.3211] | True | model-level functional reliance under the tested intervention |
| synthetic | GraphSAGE | 11 | C_unused | 1.000 | True | 0.000 | 0.0189 | 0.1702 | -0.1513 [-0.1589, -0.1437] | True | inconclusive |
| synthetic | GraphSAGE | 11 | C_relational | 1.000 | True | 0.000 | 0.3962 | 0.0124 | 0.3838 [0.3674, 0.3982] | True | model-level functional reliance under the tested intervention |
| synthetic | GraphSAGE | 23 | C_used | 1.000 | True | 0.000 | 0.3858 | 0.1651 | 0.2207 [0.2054, 0.2368] | True | model-level functional reliance under the tested intervention |
| synthetic | GraphSAGE | 23 | C_unused | 1.000 | True | 1.000 | 0.0174 | 0.1444 | -0.1271 [-0.1348, -0.1197] | True | inconclusive |
| synthetic | GraphSAGE | 23 | C_relational | 1.000 | True | 1.000 | 0.3879 | 0.0110 | 0.3769 [0.3644, 0.3887] | True | model-level functional reliance under the tested intervention |
| synthetic | GraphSAGE | 37 | C_used | 1.000 | True | 1.000 | 0.3697 | 0.0855 | 0.2842 [0.2770, 0.2911] | True | model-level functional reliance under the tested intervention |
| synthetic | GraphSAGE | 37 | C_unused | 1.000 | True | 1.000 | 0.0130 | 0.0474 | -0.0343 [-0.0368, -0.0320] | True | inconclusive |
| synthetic | GraphSAGE | 37 | C_relational | 1.000 | True | 1.000 | 0.4275 | 0.0078 | 0.4196 [0.4100, 0.4287] | True | model-level functional reliance under the tested intervention |
| bracs | DeepSets | 11 | nuclear_area_pixels | 0.918 | True | 0.818 | 0.0080 | 0.0010 | 0.0070 [0.0006, 0.0189] | False | inconclusive: intervention validity failed |
| bracs | DeepSets | 23 | nuclear_area_pixels | 0.936 | True | 0.000 | 0.0146 | 0.0014 | 0.0132 [0.0046, 0.0258] | False | inconclusive: intervention validity failed |
| bracs | DeepSets | 37 | nuclear_area_pixels | 0.936 | True | 0.955 | 0.0134 | 0.0014 | 0.0120 [0.0037, 0.0240] | False | inconclusive: intervention validity failed |
| bracs | GraphSAGE | 11 | nuclear_area_pixels | 0.929 | True | 0.909 | 0.0112 | 0.0008 | 0.0104 [0.0034, 0.0209] | False | inconclusive: intervention validity failed |
| bracs | GraphSAGE | 23 | nuclear_area_pixels | 0.930 | True | 0.818 | 0.0081 | 0.0008 | 0.0072 [0.0022, 0.0132] | False | inconclusive: intervention validity failed |
| bracs | GraphSAGE | 37 | nuclear_area_pixels | 0.957 | True | 0.000 | 0.0065 | 0.0004 | 0.0062 [0.0005, 0.0140] | False | inconclusive: intervention validity failed |

Probe is AUROC for synthetic binary concepts and R2 for continuous BRACS area. Balanced accuracy and MAE, all layerwise probes, permutation nulls, randomized-model checks and patient/replicate observations are in summary.csv.

## Classification

| Stage | Model | Seed | Accuracy | Balanced accuracy | AUROC |
|---|---|---:|---:|---:|---:|
| synthetic | DeepSets | 11 | 0.703 | 0.576 | 0.721 |
| synthetic | DeepSets | 23 | 0.703 | 0.743 | 0.744 |
| synthetic | DeepSets | 37 | 0.703 | 0.739 | 0.736 |
| synthetic | GraphSAGE | 11 | 0.906 | 0.865 | 0.813 |
| synthetic | GraphSAGE | 23 | 0.906 | 0.865 | 0.810 |
| synthetic | GraphSAGE | 37 | 0.906 | 0.865 | 0.848 |
| bracs | DeepSets | 11 | 0.773 | 0.795 | 0.795 |
| bracs | DeepSets | 23 | 0.773 | 0.795 | 0.821 |
| bracs | DeepSets | 37 | 0.773 | 0.795 | 0.786 |
| bracs | GraphSAGE | 11 | 0.773 | 0.795 | 0.839 |
| bracs | GraphSAGE | 23 | 0.727 | 0.759 | 0.848 |
| bracs | GraphSAGE | 37 | 0.773 | 0.795 | 0.830 |

## Interpretation and limits

Randomized-model sanity checks exceeding the excess-effect margin: synthetic GraphSAGE seed 23 C_used: 0.0372 [0.0361, 0.0381]; synthetic GraphSAGE seed 23 C_unused: 0.0310 [0.0301, 0.0318]. These are failures of interpreting positive GCIA alone as learned task reliance; the model responds even before training.

Intervention validity failures (all retained in the analysis): bracs patient/replicate 22: nonpositive nuclear area introduced by matched control

The synthetic learned-model results demonstrate decodable != necessarily functionally used for at least one run.
Recovery must be judged per seed; the table exposes all negative and inconclusive findings. A small excess by itself is insufficient for negligible use: the target-effect upper interval must also lie below the fixed margin.

GCIA intervals resample patients (BRACS) or independent paired replicates (synthetic), after averaging controls within graph. They are pointwise exploratory intervals, not multiplicity-adjusted confirmatory evidence. Probe null fits use permuted training labels and are scored against independently permuted held-out labels; both permutations respect patient blocks. Synthetic binary labels are flipped jointly within replicate. No held-out labels enter fitting.

Matched feature controls preserve the edited column, node count and L2 budget but redistribute signs; they change within-graph variability. Synthetic relation controls preserve mixing fraction, exact replaced-edge budget, degrees, edge count, no loops/duplicates, and connectivity. Synthetic graphs have no physical coordinates, so edge-length matching is unavailable.

BRACS uses existing graphs without resegmentation. Measured area is not an independent pathology annotation. Area-only edits leave correlated axes and pixels unchanged; biological plausibility is unvalidated. Any supported result is conditional on this numerical feature intervention. Population and histological relation interventions were not run.

TCAV estimates local class-1 probability sensitivity along a train-fitted direction, only at the final pooled layer. An increasing concept can be associated with the negative class; a low positive fraction is not proof of insensitivity. NaN means a zero-length CAV, not a measured zero response.

Randomized networks may respond to interventions and encode input features. Their checks are reported without requiring zero sensitivity. Shuffled-concept null probe values and permutation p-values must be considered with the encoding threshold.

## Missing evidence and next step

- BLOCKED: requires independent concept annotations (Concept 2)
- BLOCKED: requires independent concept annotations and cell/compartment labels (Concept 3)
- Validated physical scale / MPP
- Independent BRACS segmentation validation
- Untouched formally locked confirmatory test set
- Biological plausibility of edited feature vectors

Next scientific step: audit the synthetic feature controls for departure from generator support and preregister a revised control protocol for a new benchmark run; do not retune this run. For BRACS, independently validate nuclear segmentation and obtain patient-linked expert concept annotations before locking an untouched patient-disjoint confirmatory cohort.

## Software verification

The final run corrects the permutation null to shuffle labels independently by group in both train and held-out partitions. This is an implementation correction, with unchanged training, interventions and decision thresholds; a regression test covers the error.

```json
{
  "checks": [
    {
      "command": [
        "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA\\.venv\\Scripts\\python.exe",
        "-m",
        "ruff",
        "check",
        "src",
        "experiments/synthetic",
        "experiments/bracs",
        "tests"
      ],
      "cwd": "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA",
      "returncode": 0,
      "output": "All checks passed!\n"
    },
    {
      "command": [
        "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA\\.venv\\Scripts\\python.exe",
        "-m",
        "ruff",
        "format",
        "--check",
        "src",
        "experiments/synthetic",
        "experiments/bracs",
        "tests"
      ],
      "cwd": "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA",
      "returncode": 0,
      "output": "13 files already formatted\n"
    },
    {
      "command": [
        "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA\\.venv\\Scripts\\python.exe",
        "-m",
        "pytest",
        "-q"
      ],
      "cwd": "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA",
      "returncode": 0,
      "output": "..................                                                       [100%]\n============================== warnings summary ===============================\n.venv\\Lib\\site-packages\\torch\\jit\\_script.py:1488\n.venv\\Lib\\site-packages\\torch\\jit\\_script.py:1488\n  C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA\\.venv\\Lib\\site-packages\\torch\\jit\\_script.py:1488: DeprecationWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.\n    warnings.warn(\n\n-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n18 passed, 2 warnings in 12.72s\n"
    },
    {
      "command": [
        "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA\\.venv\\Scripts\\python.exe",
        "-m",
        "pytest",
        "-q",
        "tests/test_gcia.py"
      ],
      "cwd": "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA",
      "returncode": 0,
      "output": "...........                                                              [100%]\n============================== warnings summary ===============================\n.venv\\Lib\\site-packages\\torch\\jit\\_script.py:1488\n.venv\\Lib\\site-packages\\torch\\jit\\_script.py:1488\n  C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA\\.venv\\Lib\\site-packages\\torch\\jit\\_script.py:1488: DeprecationWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.\n    warnings.warn(\n\n-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n11 passed, 2 warnings in 10.95s\n"
    },
    {
      "command": [
        "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA\\.venv\\Scripts\\python.exe",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider"
      ],
      "cwd": "C:\\Users\\Mahugnon\\Documents\\CB Biomedecine\\GCIA_archive\\legacy",
      "returncode": 0,
      "output": "......................................................................   [100%]\n70 passed in 27.42s\n"
    }
  ],
  "passed": true,
  "timestamp_utc": "2026-09-21T16:35:38.852187+00:00",
  "source_hashes": {
    "src\\evaluation.py": "0ec3c6ed43215a9170db7eab0ade332274cb76a44ec0d76b1189de50c76dc5cc",
    "src\\graph.py": "0ef0dc734610088f5f711c49d84cc9e683026dcf0600076e73c5a73f25507779",
    "src\\interventions.py": "d9081fed4c16663f6ecedb7cf6dd12b44ba92ee5b8e5ca70e662680b4181b752",
    "src\\models.py": "97ed23e1b2e2b07754516a5afc11fbcd35165bdcfb659e8b48667ed7a33b8fd9",
    "src\\probes.py": "7befad15be048b36719bf7389947b5cc7f50510a8610ad49d0849b6bc9917a4c",
    "src\\tcav.py": "b7096a5baac2b836d7e73b768cc1f2c956019be3bc4df32f27e0bc49e784e52f",
    "experiments\\synthetic\\run.py": "e5fb2d17af43a66062b1dc3920d9cda6c37ec434917ed514e0f2d17a41e6205f",
    "experiments\\bracs\\run.py": "f335bc74280d8b75689e5de5c9a0250a7bae5e6bfab172972fb23cd3be3fb2a0",
    "configs\\study.yaml": "a1fd2b62dc2bc2760068d46de3a62b52e48852df45162c2b9cda8e5be2472a55",
    "tests\\test_gcia.py": "47b3c68ee2b1b9efc0750b8a5781678569213e2689649d37ad6098b751e36fa4"
  }
}
```

Historical pilot artifacts remain unchanged in GCIA_archive/legacy and results/synthetic_v1. Only summary.csv, report.md and run_manifest.json are canonical full-study outputs.
