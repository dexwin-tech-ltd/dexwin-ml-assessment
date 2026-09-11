# Data card — Dexwin Pay authorization sample

**This file is for candidates.** All rows are synthetic. There is no real customer PII.

## Business context

Dexwin Pay is a checkout product for merchants across Ghana, Nigeria, and Kenya. When a customer pays, we have to decide in a few hundred milliseconds: **approve** or **decline**.

The current **rules engine** declines a large share of traffic. That catches some fraud, but it also blocks a lot of good customers. Merchants feel that as lost sales; we call those **false declines**.

Fraud labels (`is_fraud`) arrive later — chargebacks, mule reports, and ops confirmations, often days after the payment. Treat `is_fraud = 1` as "this payment was bad," not as "we declined it."

## What we want from you

Build something **better than the rules engine and the weak baseline**, with the product goal:

> Catch fraud, but **cut false declines**. Do not optimize raw accuracy.

A model that declines almost nobody, or almost everybody, is not useful. Be ready to say which metric you would actually ship on, and why.

## File

`data/transactions.csv` — 3,218 rows (including a handful of duplicate `txn_id`s), one row per payment attempt. Fraud rate is about **8.5%**.

## Columns

| Column | Meaning | Available at decision time? |
| --- | --- | --- |
| `txn_id` | Payment id | yes |
| `event_ts` | Payment timestamp (warehouse export; formats vary) | yes, as event time |
| `amount` | Payment amount (messy strings) | yes |
| `currency` | `GHS` / `NGN` / `USD` | yes |
| `merchant_id` | Merchant | yes |
| `merchant_category` | Merchant category (inconsistent labels) | yes |
| `channel` | `web`, `android`, `ios`, `ussd`, `pos` | yes |
| `country` | Customer country (inconsistent labels) | yes |
| `customer_id` | Synthetic customer token (`C-xxxxx`) | yes |
| `customer_tenure_days` | Days since first seen | usually |
| `txns_30d` | Payments by this customer in last 30 days | yes |
| `avg_amount_30d` | Average amount in last 30 days | yes |
| `prior_declines_30d` | How often we declined them recently | yes |
| `hour` | Hour of day | yes |
| `is_new_device` | 1 if device not seen before | yes |
| `payment_method` | `card`, `momo`, `bank` | yes |
| `device_fingerprint` | Opaque device token | yes |
| `ops_review_code` | Ops warehouse field | **you should decide** |
| `rules_engine_decline` | What the live rules engine did (`Y`/`N`) | known, but it is the thing we want to replace |
| `is_fraud` | **Label.** 1 = later confirmed bad payment | **no** (future information) |

You may assume feature values in this file are a warehouse dump, not a carefully designed training table. Some fields might not belong in an online model.

## Label and imbalance

Fraud is the minority class (single-digit percent). Most payments are good. That is why accuracy is a poor headline metric here.

The rules engine currently declines on the order of **~45%** of payments. Only about **1 in 10** of those declines is actually fraud. That is the false-decline problem.

## Known mess (not exhaustive)

- Missing values
- Inconsistent category and country strings
- Amount stored as text (`$1,200.00`, `NA`, blanks, extra units)
- Mixed timestamp formats
- A few duplicated `txn_id`s
- A few impossible tenure values

## Constraints for this session

- Offline only. Do not download extra datasets.
- You may use pandas / scikit-learn / the standard library.
- Do not spend the whole session on hyperparameter search.
- A simple model with honest features and the right metric beats a fancy model that cheats.

## Suggested cost sketch (optional)

If you want a single number to argue about:

- False decline (block a good payment): lose **~1.0 × amount** in GMV, plus merchant trust.
- Missed fraud (approve a bad payment): lose **~1.0 × amount** plus ops cost — treat as **~3 × amount** if you want a simple asymmetric cost.

Use this only if it helps you choose a threshold. It is not a trick.
