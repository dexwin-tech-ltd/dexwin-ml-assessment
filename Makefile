PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: help setup baseline notebook data verify

help:
	@echo "Dexwin live ML assessment"
	@echo "  make setup     Install pinned dependencies"
	@echo "  make baseline  Run the weak baseline script"
	@echo "  make notebook  Open the candidate notebook"
	@echo "  make data      Regenerate data/transactions.csv (interviewers only)"
	@echo "  make verify    Alias for baseline (sanity-check the room)"

setup:
	$(PIP) install -r requirements.txt

baseline:
	$(PYTHON) scripts/baseline.py

notebook:
	$(PYTHON) -m jupyter notebook notebooks/assessment.ipynb

data:
	$(PYTHON) scripts/generate_data.py

verify: baseline
