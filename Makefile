# Rotor-fault recovery for a 5-inch quadrotor -- simulation study.
# All targets use the local virtual environment (.venv); `make setup` creates it.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup test lint format experiments figures report reproduce smoke clean-results paper

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

# The IEEE-style paper, paper/main.pdf: pdflatex + bibtex when TeX Live is installed,
# otherwise Tectonic (a single binary that fetches the packages it needs).
paper:
	cd paper && if command -v pdflatex >/dev/null 2>&1; then \
	  pdflatex -interaction=nonstopmode main && bibtex main && \
	  pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main; \
	elif command -v tectonic >/dev/null 2>&1; then tectonic main.tex; \
	else echo "install TeX Live (pdflatex, bibtex) or Tectonic to build the paper"; exit 1; fi
