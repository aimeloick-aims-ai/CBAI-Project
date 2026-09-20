$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
    foreach ($scriptName in @('bracs_data.py', 'audit_bcss.py', 'check_acquisitions.py')) {
        Write-Output "Running $scriptName"
        & $pythonExe -u (Join-Path 'scripts' $scriptName)
        if ($LASTEXITCODE -ne 0) { throw "Failed: $scriptName" }
    }
    Write-Output 'Data smoke checks finished. No model was trained.'
} finally {
    Pop-Location
}
