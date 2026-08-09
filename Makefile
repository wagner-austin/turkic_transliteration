SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: install lint guard test check clean build web

# Install dependencies, including the Windows PyICU wheel which PyPI rules
# prevent pip from resolving automatically.
install:
	poetry install --extras corpus --extras dev --no-ansi
	if ($$IsWindows -or $$env:OS -eq 'Windows_NT') { poetry run python -m turkic_translit.pyicu_install }

# Guards: no Any/cast/object/type-ignore/TypeAlias/suppress, no silent excepts,
# no print in src, no weak or fake tests. Covers src, tests and scripts.
guard:
	poetry run python -m scripts.guard
	if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }

# Lint: guards first (cheapest signal), then Ruff, then strict Mypy over
# src, tests and scripts.
lint: install guard
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Test: statement and branch coverage over src and scripts, 100% enforced by
# the fail_under in pyproject.
#
# COVERAGE_PROCESS_START is what makes sitecustomize.py start coverage in
# child processes. Several tests exercise the real console scripts by
# spawning them; without this their execution is invisible and the report
# calls that code untested.
test: install
	$$env:COVERAGE_PROCESS_START = "$(CURDIR)/pyproject.toml"; poetry run pytest -n auto --cov=src --cov=scripts --cov-branch --cov-report=term-missing

check: lint test

clean:
	poetry run python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('dist') + glob.glob('build') + glob.glob('*.egg-info') + glob.glob('htmlcov') + glob.glob('.pytest_cache') + glob.glob('.ruff_cache') + glob.glob('.mypy_cache') + glob.glob('**/__pycache__', recursive=True) if os.path.exists(p)]"

build: clean install
	poetry run python -m build

web: install
	poetry run python turkic_tools.py web
