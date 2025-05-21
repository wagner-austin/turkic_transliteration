# PowerShell script to run equivalent commands to the Makefile targets
# for Windows users who don't have GNU Make installed

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Show-Help {
    Write-Host "Available commands:"
    Write-Host "  ./scripts/run.ps1 lint       - Run linting checks"
    Write-Host "  ./scripts/run.ps1 format     - Auto-format code" 
    Write-Host "  ./scripts/run.ps1 test       - Run tests"
    Write-Host "  ./scripts/run.ps1 clean      - Remove build artifacts"
    Write-Host "  ./scripts/run.ps1 build      - Build distribution package"
    Write-Host "  ./scripts/run.ps1 web        - Run the web UI example"
    Write-Host "  ./scripts/run.ps1 demo       - Run the simple demo"
    Write-Host "  ./scripts/run.ps1 full-demo  - Run the comprehensive demo"
}

function Invoke-Lint {
    Write-Host "Running ruff linter..."
    ruff check $ProjectRoot\src $ProjectRoot\tests $ProjectRoot\examples
    
    Write-Host "Checking formatting with black..."
    black --check $ProjectRoot\src $ProjectRoot\tests $ProjectRoot\examples
    
    Write-Host "Running type checking with mypy..."
    mypy --strict $ProjectRoot
}

function Invoke-Format {
    Write-Host "Formatting code with black..."
    black $ProjectRoot\src $ProjectRoot\tests $ProjectRoot\examples
    
    Write-Host "Auto-fixing issues with ruff..."
    ruff check --fix $ProjectRoot\src $ProjectRoot\tests $ProjectRoot\examples
}

function Invoke-Test {
    Write-Host "Running tests..."
    python -m pytest
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
    Write-Host "Building distribution package..."
    python -m build
}

function Invoke-Web {
    Write-Host "Starting web UI..."
    python $ProjectRoot\turkic_tools.py web
}

function Invoke-Demo {
    Write-Host "Running simple demo..."
    python $ProjectRoot\turkic_tools.py demo
}

function Invoke-FullDemo {
    Write-Host "Running comprehensive demo..."
    python $ProjectRoot\turkic_tools.py full-demo
}

# Execute the requested command
switch ($Command) {
    "lint" { Invoke-Lint }
    "format" { Invoke-Format }
    "test" { Invoke-Test }
    "clean" { Invoke-Clean }
    "build" { Invoke-Build }
    "web" { Invoke-Web }
    "demo" { Invoke-Demo }
    "full-demo" { Invoke-FullDemo }
    default { Show-Help }
}
