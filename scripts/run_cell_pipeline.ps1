param([string]$OutputDirectory = "reports/cell_pipeline")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv/Scripts/python.exe"
Push-Location $projectRoot
try {
    if (Test-Path -LiteralPath $OutputDirectory) {
        throw "Output directory already exists. Preserve results and choose a new directory."
    }
    & $pythonPath scripts/acquire_hovernet.py
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint acquisition failed" }
    & $pythonPath scripts/prepare_cell_cohort.py --weights data/pretrained/hovernet_fast-pannuke.pth --output "$OutputDirectory/cohort"
    if ($LASTEXITCODE -ne 0) { throw "Cohort construction failed; training not started" }
    & $pythonPath scripts/train_cell_gnns.py --manifest "$OutputDirectory/cohort/manifest.csv" --output "$OutputDirectory/training"
    if ($LASTEXITCODE -ne 0) { throw "Training failed; inspect saved outputs" }
} finally {
    Pop-Location
}
