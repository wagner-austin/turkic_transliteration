<#
release.ps1 - Clean build and publish to PyPI (setuptools/PEP 621 project)
Usage: powershell -ExecutionPolicy Bypass -File release.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "Removing previous build artifacts..."
Remove-Item dist,build -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Installing/updating build tools (build, twine)..."
pip install --upgrade build twine

Write-Host "Building package..."
python -m build

Write-Host "Uploading package to PyPI..."
python -m twine upload dist/*

Write-Host "`nAll done! View your release at https://pypi.org/project/turkic-translit/"
