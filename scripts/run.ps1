# PowerShell equivalents of the Makefile targets, for Windows users who
# do not have GNU Make installed.
#
# Each function runs what the Makefile runs, through Poetry, so the two
# cannot disagree about what a target means. They had disagreed: this
# script linted a deleted examples/ directory, checked formatting with
# black — which this project does not use, and never has in the
# Makefile — skipped the guards and Mypy entirely, and served the web UI
# through a turkic_tools.py that no longer exists.

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Show-Help {
    Write-Host "Available commands:"
    Write-Host "  ./scripts/run.ps1 install    - Install dependencies with the dev extras"
    Write-Host "  ./scripts/run.ps1 guard      - Run the guard rules"
    Write-Host "  ./scripts/run.ps1 lint       - Guards, then Ruff, then strict Mypy"
    Write-Host "  ./scripts/run.ps1 test       - Run the suite with 100% coverage enforced"
    Write-Host "  ./scripts/run.ps1 check      - lint + test, the full gate"
    Write-Host "  ./scripts/run.ps1 clean      - Remove build artifacts"
    Write-Host "  ./scripts/run.ps1 build      - Build the distribution package"
    Write-Host "  ./scripts/run.ps1 web        - Serve the web demo"
}

function Invoke-Install {
    Write-Host "Installing dependencies..."
    poetry install --extras dev --no-ansi
}

function Invoke-Guard {
    Write-Host "Running guard rules..."
    poetry run python -m scripts.guard
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Lint {
    Invoke-Install
    Invoke-Guard

    Write-Host "Running Ruff..."
    poetry run ruff check . --fix
    poetry run ruff format .

    Write-Host "Running Mypy..."
    poetry run mypy src tests scripts
}

function Invoke-Test {
    Invoke-Install
    Write-Host "Running tests..."
    $env:COVERAGE_PROCESS_START = "$ProjectRoot/pyproject.toml"
    poetry run pytest -n auto --cov=src --cov=scripts --cov-branch --cov-report=term-missing
}

function Invoke-Check {
    Invoke-Lint
    Invoke-Test
}

function Invoke-Clean {
    Write-Host "Cleaning build artifacts..."
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\dist
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\build
    Get-ChildItem -Path $ProjectRoot -Filter "*.egg-info" -Recurse -Directory | Remove-Item -Recurse -Force
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\.coverage
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\htmlcov
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\.pytest_cache
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\.ruff_cache
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\.mypy_cache

    Get-ChildItem -Path $ProjectRoot -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Path $ProjectRoot -Filter "*.pyc" -Recurse -File | Remove-Item -Force
}

function Invoke-Build {
    Invoke-Clean
    Invoke-Install
    Write-Host "Building distribution package..."
    poetry run python -m build
}

function Invoke-Web {
    Invoke-Install
    Write-Host "Starting web UI..."
    poetry run turkic-translit web
}

switch ($Command) {
    "install" { Invoke-Install }
    "guard" { Invoke-Guard }
    "lint" { Invoke-Lint }
    "test" { Invoke-Test }
    "check" { Invoke-Check }
    "clean" { Invoke-Clean }
    "build" { Invoke-Build }
    "web" { Invoke-Web }
    default { Show-Help }
}
