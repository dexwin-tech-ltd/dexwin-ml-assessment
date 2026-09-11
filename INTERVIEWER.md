# Interviewer only — do not walk the candidate through this file

If a candidate opens it anyway, that is useful signal. Do not punish curiosity in the first minute; do mark it if they copy the “hidden issues” list instead of finding problems in the data.

This session is a **live ML screen** for an ML engineer (or ML-leaning backend) who will own tabular models in production. It is not a research interview.

---

## 5-minute interviewer setup (before the candidate joins)

```bash
git clone <this-repo> && cd dexwin-ml-assessment
python3 -m venv .venv && source .venv/bin/activate
make setup
make baseline
```

You should see something like (seed 42, committed CSV):

- `rows loaded: 3218` then `rows after dropna: 2368`
- **accuracy around 0.93**
- confusion matrix that **never predicts fraud**, e.g. `[[439, 0], [35, 0]]`

That accuracy is the majority class. A `LogisticRegression` convergence warning is expected (unscaled features). If the script errors, the room is not ready.

The candidate can use Cursor’s terminal the same way.

Optional: open `notebooks/assessment.ipynb` so you can share the screen on the notebook rather than the script.

## What the candidate is told

They have messy Dexwin Pay authorization data. The product goal is **fewer false declines** while still catching fraud. A weak baseline is provided. They must improve it modestly **and** explain serving + monitoring.

They are told to ignore this file.

---

## Session clock (60–75 min)

| Minutes | You | Them |
| --- | --- | --- |
| 0–10 | Stay quiet unless they are stuck on install | Setup, DATA_CARD, `head` of the CSV |
| 10–45 | Nudge only if they vanish into hyperparams | Cleaning, leakage, metric, simple model |
| 45–70 | Drive this if they do not start it | Serve + monitor conversation |
| 70–75 | Ask for a recap | What they would do with another day |

Do **not** let modeling eat the serving discussion. At minute 45, switch: “Let’s freeze the model. How would this run in production?”

---

## Hidden dataset facts

All numbers below are from seed `42` in `scripts/generate_data.py`. If someone regenerates the CSV with a different seed, update this section.

### Label and policy

- `is_fraud` rate is about **8.5%** (272 / 3218).
- `rules_engine_decline` is a blunt live policy: high amount, new device, recent declines, gambling/crypto, brand-new customer + non-tiny amount. Decline rate is **~46%**, precision **~0.14**, recall **~0.75**. About 1,265 good payments are blocked vs 67 missed frauds. That is the false-decline story.
- Using `rules_engine_decline` as a feature to *replace* the rules engine is circular. Using it as a **baseline policy** to compare against is correct.

### Leakage temptation: `ops_review_code`

Assigned **after** ops sees downstream evidence.

| Code | Rough meaning | Reality in this file |
| --- | --- | --- |
| `CNF` | confirmed fraud | mostly `is_fraud = 1` |
| `CLR` | cleared | mostly `is_fraud = 0` |
| `ESC` | escalated / messy | mixed |

The weak baseline **keeps this column** (and IDs, fingerprints, etc.). On the committed CSV it still behaves like a majority classifier — high-cardinality encodings drown the leak. A candidate who *isolates* `ops_review_code` will see ROC ~0.97. That is the temptation: the column is almost a delayed label.

A candidate who ships `ops_review_code` has not built an authorization model. They have rebuilt the post-hoc ops queue.

### Other traps

- `device_fingerprint` is high cardinality noise. Encoding it is a waste; it will overfit.
- `txn_id` / `customer_id` as numeric encodings are not features.
- `dropna()` on the whole frame throws away a lot of usable traffic, often non-randomly.
- Amount / country / category / timestamps are dirty on purpose.
- A few duplicate `txn_id`s and impossible tenures (`-1`, `99999`).
- Target is imbalanced; **accuracy is the wrong headline**. The baseline prints accuracy on purpose.
- LabelEncoder fitted on the **full** frame before the split (information leak from test into encoding).

### Honest signal (so they can actually improve)

Fraud is more likely when:

- amount is a spike vs `avg_amount_30d`
- new customer **and** new device
- `gambling` / `crypto`
- late-night hours, `ussd`
- several recent declines

Established customers with many recent payments are safer.

A plain `LogisticRegression(class_weight="balanced")` on cleaned, non-leaky features (seed-42 stratified 80/20) lands around:

| | Average precision | ROC AUC | Notes |
| --- | --- | --- | --- |
| Prevalence / dummy | ~0.08 | 0.50 | accuracy ~0.92 |
| Weak `scripts/baseline.py` | n/a (all-negative) | n/a | accuracy ~0.93, recall 0 |
| Honest logistic | **~0.19** | **~0.70** | accuracy *drops* to ~0.72 at threshold 0.5; fewer false declines than rules, slightly less fraud recall |
| `ops_review_code` only | **~0.72** | **~0.97** | this is cheating |
| Small `HistGradientBoostingClassifier` | ~0.24 | ~0.69 | optional extra lift, not required |

Beating the rules engine on the same recall is **not** required to pass. Picking an operating point (fewer false declines vs missed fraud) is. Accuracy will usually **lose** to the leaky/majority baseline. That is the point.

---

## How to score the model work

**Do not pass someone for beating baseline accuracy.** The leaky baseline is supposed to look strong on accuracy.

Score the *honest* work:

| Bar | What you should see |
| --- | --- |
| Below | Random forest on raw dump, accuracy celebrated, leakage used, no metric discussion |
| Pass | Drops or questions `ops_review_code`; stops using accuracy as the goal; stratified split or equivalent; handles missing amount/category without dropping the world; simple model (logistic / tree); talks precision-recall or cost |
| Strong | Threshold chosen for false-decline vs missed-fraud; compares to the rules engine, not just to `scripts/baseline.py`; calibration or expected cost; clear “what I would not put in the online request” |

Modest improvement means: **honest PR-AUC / fraud F1 / precision@k clearly better than an honest naive model**, not “I added XGBoost.” sklearn `LogisticRegression(class_weight="balanced")` or a small `HistGradientBoostingClassifier` is plenty.

If they need a north star metric, prefer:

1. PR-AUC or average precision on `is_fraud`
2. Precision at a recall target (e.g. catch ~60% of fraud)
3. A cost: false decline ≈ 1× amount, missed fraud ≈ 3× amount (see DATA_CARD)

Rules-engine comparison is gold: “same fraud recall, fewer good customers blocked.”

---

## Serving & monitoring — answer key

This is half the interview. A backend-strong candidate can still pass if modeling was only OK and this part is crisp.

### Serving (good answers)

- **Online, low latency** (authorization is a few hundred ms budget). Batch scoring is the wrong default; batch is fine for backfills and training.
- Features must be **available at request time**. `ops_review_code` is not. `is_fraud` is not. Aggregates like `txns_30d` need a point-in-time store or pre-computed profile, not a warehouse join that includes the current payment.
- Shadow the rules engine first: log model score, do not change the decision. Then canary a merchant slice.
- Version the model artifact + the feature schema together.
- Fail open vs fail closed is a product call; they should *name* it (e.g. fail toward approve for low amount, fail toward rules engine if the model is down).

### Monitoring (good answers)

- **Decision metrics:** decline rate, false-decline proxy (approved customers who would have been declined; complaints; conversion), fraud caught vs missed (lagged).
- **Label delay:** fraud labels arrive days later. You cannot wait for them to know today’s model is broken. Use leading indicators: score distribution, decline rate by merchant/channel, feature drift (PSI / KS on amount, category mix, new-device rate).
- **Data quality:** missing amount spike, new `merchant_category` values, timestamp parse failures.
- **Ops loop:** how CNF/CLR get back into training without leaking into serving features.
- Retrain cadence: weekly/monthly batch is enough at this size; triggered retrain on drift.

Weak answers: “we’ll monitor accuracy in prod” or “retrain every hour.”

---

## Pass / fail guidance

Use this as a hiring recommendation, not a points quiz.

**Pass** (hire / onsite) when they:

1. Treat leakage as a first-class bug, not a feature, **and**
2. Pick a metric that matches false declines vs fraud, **and**
3. Can describe an online (or shadow) serving path and lagged-label monitoring.

They do not need a beautiful notebook. They do need a runnable honest model and a coherent production story.

**Strong pass:** cost/threshold, rules-engine comparison, point-in-time features, fail-open/closed, concrete drift alerts.

**Fail** when two or more of these are true:

- Never questions `ops_review_code` even after a hint
- Reports accuracy as success and resists changing metric
- No idea how the model would be called at payment time
- Silent 40-minute hyperparameter crawl
- Invents real PII or external data to “fix” the task

**Hint ladder** (only if stuck):

1. “Which of these columns exist *before* we approve the payment?”
2. “If accuracy is 94%, what is the majority-class accuracy?”
3. “How would you compare this to the current rules engine?”
4. (Late, if still lost) “What do you think `ops_review_code` is?”

---

## If they finish early

Ask:

- How would you A/B this without lighting money on fire?
- What is in the JSON the checkout service sends the model?
- How do you stop training on the model’s own yesterday’s declines (feedback loops)?
- Walk through a merchant who suddenly sells electronics instead of groceries.

## If install fails

`pip install pandas scikit-learn jupyter` unpinned is an acceptable fallback. The notebook is optional; `python scripts/baseline.py` is the source of truth.

## Regenerating data

```bash
make data
```

Keep `SEED = 42` unless you intentionally want a new draw. After a new draw, re-run the baseline and refresh the metric ranges in this file.
