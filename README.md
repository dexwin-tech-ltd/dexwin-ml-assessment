# Dexwin live ML engineer assessment

60–75 minute pairing session. You get messy payment data and a weak baseline. Improve the model **modestly and honestly**, then talk through how you would serve and monitor it.

This is not a Kaggle contest. We care about judgment: data leaks, metrics that match the product, and how the thing would live in production.

## Before you start

You need Python 3.11+ and a terminal. Cursor is enough — you do not need a Dev Container. A virtualenv is recommended (`python3 -m venv .venv`); on Debian/Ubuntu install `python3-venv` first if that command fails.

1. Open this folder in Cursor (`File → Open Folder`).
2. In the Cursor terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or: `make setup`

3. Start from either:

```bash
make baseline                      # scripts/baseline.py
# or
make notebook                      # notebooks/assessment.ipynb
```

If `make notebook` does not open a browser, open `notebooks/assessment.ipynb` in Cursor and use the Jupyter kernel from your venv.

`data/transactions.csv` is already in the repo. You do **not** need to generate data.

## Read these (10 minutes max)

1. This README
2. [`data/DATA_CARD.md`](data/DATA_CARD.md) — business goal, columns, constraints
3. The baseline in [`scripts/baseline.py`](scripts/baseline.py) or the first cells of the notebook

Ignore [`INTERVIEWER.md`](INTERVIEWER.md). That file is for the interviewer.

## Product goal

Dexwin Pay currently declines too many good customers (**false declines**). Fraud is real, but blocking honest payments loses GMV and merchant trust.

Label: `is_fraud` (1 = payment later confirmed bad).  
Optimize for the product, not for accuracy.

## Timebox

| When | What |
| --- | --- |
| 0–10 min | Setup, skim the data card, peek at the table |
| 10–45 min | Clean, choose features, pick a metric, beat the *honest* baseline |
| 45–70 min | How you would serve this, and how you would know it is breaking |
| last 5 min | Recap: what you would do with another day |

If you only do modeling, we will still spend the last stretch on serving and monitoring — that part is required.

## What to deliver in the session

1. A model you would actually consider putting behind a shadow or canary — not a leaderboard score.
2. A metric (and threshold story) tied to **false declines vs missed fraud**.
3. A short verbal design for serving + monitoring.

You may edit the notebook, the script, or both. Keep it runnable.

## Ground rules

- Offline only. No extra datasets, no copying a full solution from the internet.
- All customer identifiers in the CSV are fake (`C-10482`, not names or phones).
- Ask questions. Talking through a tradeoff is better than silent tuning.
- A simple, correct approach beats a complex, leaky one.

## Layout

```
data/DATA_CARD.md              business + column notes
data/transactions.csv          synthetic payments
scripts/baseline.py            weak baseline (run this first)
notebooks/assessment.ipynb     same baseline + space to work
requirements.txt               pinned pandas / scikit-learn / jupyter
Makefile                       setup / baseline / notebook
```

Good luck. Think out loud.
