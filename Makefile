# Rotor-fault recovery for a 5-inch quadrotor -- simulation study.
# All targets use the local virtual environment (.venv); `make setup` creates it.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup test lint format experiments figures report reproduce smoke clean-results

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests experiments
	$(PY) -m ruff format --check src tests experiments

format:
	$(PY) -m ruff format src tests experiments
	$(PY) -m ruff check --fix src tests experiments

# Calibration + E1-E6 (results/*.json, results/*.csv, results/replays/*.json)
experiments:
	$(PY) -m rotorfault experiments

# Figures from results/ (re-simulates representative runs if the cache is missing)
figures:
	$(PY) -m rotorfault figures

# Tables, headline numbers, generated blocks of README.md / paper/report.md, LaTeX tables
report:
	$(PY) -m rotorfault report

# Everything end to end: experiments -> figures -> report
reproduce:
	$(PY) -m rotorfault reproduce

# Small version of the whole pipeline in a scratch directory (used by CI)
smoke:
	mkdir -p build/smoke
	ROTORFAULT_ROOT=build/smoke $(PY) -m rotorfault experiments --quick

clean-results:
	rm -rf results/cache
