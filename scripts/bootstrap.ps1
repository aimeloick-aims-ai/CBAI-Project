$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.python'
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        uv venv --python 3.11.15 .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
    }
    & ./.venv/Scripts/python.exe -c 'import sys; assert sys.version_info[:3] == (3, 11, 15)'
    if ($LASTEXITCODE -ne 0) { throw 'Expected isolated Python 3.11.15' }
    uv pip sync --python .venv/Scripts/python.exe --require-hashes environment/requirements-win-py311.lock
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    & ./.venv/Scripts/python.exe scripts/check_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment smoke test failed' }
    & ./.venv/Scripts/python.exe -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Integration tests failed' }
} finally {
    Pop-Location
}
