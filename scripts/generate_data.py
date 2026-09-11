#!/usr/bin/env python3
"""Generate the synthetic Dexwin Pay assessment dataset.

Re-run with:  python scripts/generate_data.py
The committed CSV is what candidates should use; this script is here so
interviewers can regenerate it from a fixed seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 3200
N_CUSTOMERS = 740
N_MERCHANTS = 96

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"

CATEGORIES = [
    "grocery",
    "electronics",
    "travel",
    "gambling",
    "crypto",
    "restaurants",
    "telecom",
    "fuel",
    "other",
]
CHANNELS = ["web", "android", "ios", "ussd", "pos"]
PAYMENT_METHODS = ["card", "momo", "bank"]
CURRENCIES = ["GHS", "NGN", "USD"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def mess_amount(value: float, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.07:
        return ""
    if r < 0.10:
        return "NA"
    if r < 0.14:
        return f"${value:,.2f}"
    if r < 0.17:
        return f"{value:.2f} GHS"
    if r < 0.20:
        return f" {value:.2f} "
    if r < 0.22:
        return f"{int(round(value))}"
    return f"{value:.2f}"


def mess_category(cat: str, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.09:
        return ""
    if r < 0.12:
        return "unk"
    if r < 0.22:
        return cat.upper()
    if r < 0.32:
        return cat.title()
    if r < 0.36:
        return f" {cat} "
    return cat


def mess_country(country: str, rng: np.random.Generator) -> str:
    aliases = {
        "GH": ["GH", "Ghana", "gh", "GHA"],
        "NG": ["NG", "Nigeria", "ng", "NGA"],
        "KE": ["KE", "Kenya", "ke"],
    }
    choices = aliases[country]
    return str(rng.choice(choices))


def mess_timestamp(ts: pd.Timestamp, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.45:
        return ts.strftime("%Y-%m-%dT%H:%M:%S")
    if r < 0.75:
        return ts.strftime("%d/%m/%Y %H:%M")
    if r < 0.90:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts.strftime("%m-%d-%Y")


def main() -> None:
    rng = np.random.default_rng(SEED)

    customer_ids = np.array([f"C-{10000 + i}" for i in range(N_CUSTOMERS)])
    merchant_ids = np.array([f"M-{2000 + i}" for i in range(N_MERCHANTS)])
    merchant_cats = rng.choice(
        CATEGORIES,
        size=N_MERCHANTS,
        p=[0.22, 0.12, 0.08, 0.05, 0.04, 0.18, 0.12, 0.11, 0.08],
    )
    merchant_cat_map = dict(zip(merchant_ids, merchant_cats))

    customer_country = rng.choice(["GH", "NG", "KE"], size=N_CUSTOMERS, p=[0.62, 0.28, 0.10])
    customer_tenure = rng.integers(0, 1800, size=N_CUSTOMERS)
    # A slice of brand-new customers.
    new_idx = rng.choice(N_CUSTOMERS, size=90, replace=False)
    customer_tenure[new_idx] = rng.integers(0, 12, size=90)

    txn_customer = rng.choice(customer_ids, size=N_ROWS)
    txn_merchant = rng.choice(merchant_ids, size=N_ROWS)
    cust_index = pd.Index(customer_ids)
    cust_pos = cust_index.get_indexer(txn_customer)

    country = customer_country[cust_pos]
    tenure = customer_tenure[cust_pos].astype(float)
    category = np.array([merchant_cat_map[m] for m in txn_merchant])
    channel = rng.choice(CHANNELS, size=N_ROWS, p=[0.28, 0.30, 0.22, 0.08, 0.12])
    payment_method = rng.choice(PAYMENT_METHODS, size=N_ROWS, p=[0.42, 0.48, 0.10])
    currency = rng.choice(CURRENCIES, size=N_ROWS, p=[0.70, 0.22, 0.08])

    base_amount = rng.lognormal(mean=3.4, sigma=0.85, size=N_ROWS)
    base_amount = np.clip(base_amount, 3.0, 2500.0)
    # A few high-ticket outliers.
    outlier_idx = rng.choice(N_ROWS, size=25, replace=False)
    base_amount[outlier_idx] = rng.uniform(1800, 4800, size=25)

    txns_30d = rng.integers(0, 40, size=N_ROWS)
    avg_amount_30d = base_amount * rng.uniform(0.45, 1.55, size=N_ROWS)
    prior_declines_30d = rng.integers(0, 8, size=N_ROWS)
    # Most customers have few recent declines.
    prior_declines_30d = np.where(rng.random(N_ROWS) < 0.72, 0, prior_declines_30d)

    hour = rng.integers(0, 24, size=N_ROWS)
    is_new_device = rng.binomial(1, 0.18, size=N_ROWS)

    high_risk_cat = np.isin(category, ["gambling", "crypto"]).astype(float)
    amount_spike = (base_amount > 3.0 * np.maximum(avg_amount_30d, 1.0)).astype(float)
    new_and_new_device = ((tenure < 21) & (is_new_device == 1)).astype(float)
    night = ((hour <= 4) | (hour >= 23)).astype(float)
    ussd = (channel == "ussd").astype(float)
    lots_of_declines = (prior_declines_30d >= 3).astype(float)
    established = (txns_30d >= 12).astype(float)

    logit = (
        -3.20
        + 1.90 * amount_spike
        + 1.70 * new_and_new_device
        + 2.10 * high_risk_cat
        + 0.90 * night
        + 0.70 * ussd
        + 1.00 * lots_of_declines
        + 0.00045 * base_amount
        - 0.70 * established
    )
    logit += rng.normal(0, 0.22, size=N_ROWS)
    fraud_p = sigmoid(logit)
    is_fraud = (rng.random(N_ROWS) < fraud_p).astype(int)

    # Old rules engine: high recall, lots of false declines.
    rules_engine_decline = (
        (base_amount > 420)
        | (is_new_device == 1)
        | (prior_declines_30d >= 2)
        | (high_risk_cat == 1)
        | ((tenure < 7) & (base_amount > 80))
    ).astype(int)

    # LEAKAGE: assigned by ops after seeing the outcome / downstream evidence.
    ops_review_code = np.empty(N_ROWS, dtype=object)
    for i in range(N_ROWS):
        r = rng.random()
        if is_fraud[i] == 1:
            if r < 0.84:
                ops_review_code[i] = "CNF"  # confirmed fraud, after the fact
            elif r < 0.95:
                ops_review_code[i] = "ESC"
            else:
                ops_review_code[i] = "CLR"
        else:
            if r < 0.88:
                ops_review_code[i] = "CLR"
            elif r < 0.97:
                ops_review_code[i] = "ESC"
            else:
                ops_review_code[i] = "CNF"

    # Point-in-time timestamps over ~8 weeks ending 11 Sep 2026.
    end = pd.Timestamp("2026-09-11 18:00:00")
    offsets_min = rng.integers(0, 60 * 24 * 56, size=N_ROWS)
    event_ts = pd.to_datetime(end) - pd.to_timedelta(offsets_min, unit="m")

    # Inject missingness into tenure (more often on new customers).
    tenure_missing = rng.random(N_ROWS) < np.where(tenure < 30, 0.22, 0.06)
    tenure_out = tenure.copy()
    tenure_out[tenure_missing] = np.nan
    # A few impossible values.
    bad_tenure = rng.choice(np.where(~tenure_missing)[0], size=12, replace=False)
    tenure_out[bad_tenure] = rng.choice([-3, -1, 99999], size=12)

    hour_out = hour.astype(object)
    hour_missing = rng.random(N_ROWS) < 0.04
    hour_out[hour_missing] = ""

    amount_out = [mess_amount(float(v), rng) for v in base_amount]
    category_out = [mess_category(c, rng) for c in category]
    country_out = [mess_country(c, rng) for c in country]
    ts_out = [mess_timestamp(t, rng) for t in event_ts]

    # High-cardinality trap. Not PII — opaque synthetic tokens.
    device_fingerprint = np.array(
        [f"DEV-{rng.integers(10_000, 99_999)}" for _ in range(N_ROWS)]
    )

    df = pd.DataFrame(
        {
            "txn_id": [f"TXN-{i:06d}" for i in range(1, N_ROWS + 1)],
            "event_ts": ts_out,
            "amount": amount_out,
            "currency": currency,
            "merchant_id": txn_merchant,
            "merchant_category": category_out,
            "channel": channel,
            "country": country_out,
            "customer_id": txn_customer,
            "customer_tenure_days": tenure_out,
            "txns_30d": txns_30d,
            "avg_amount_30d": np.round(avg_amount_30d, 2),
            "prior_declines_30d": prior_declines_30d,
            "hour": hour_out,
            "is_new_device": is_new_device,
            "payment_method": payment_method,
            "device_fingerprint": device_fingerprint,
            "ops_review_code": ops_review_code,
            "rules_engine_decline": np.where(rules_engine_decline == 1, "Y", "N"),
            "is_fraud": is_fraud,
        }
    )

    # Duplicate a handful of rows (warehouse glitch).
    dup_idx = rng.choice(df.index, size=18, replace=False)
    df = pd.concat([df, df.loc[dup_idx].copy()], ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # Stats for DATA_CARD / INTERVIEWER.md (clean numeric view).
    amount_num = pd.to_numeric(
        df["amount"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )
    fraud = df["is_fraud"].to_numpy()
    rules = (df["rules_engine_decline"] == "Y").to_numpy()
    declined = int(rules.sum())
    caught = int(((rules) & (fraud == 1)).sum())
    false_declines = int(((rules) & (fraud == 0)).sum())
    missed = int(((~rules) & (fraud == 1)).sum())

    print(f"Wrote {OUT_PATH}  ({len(df)} rows, {df.shape[1]} cols)")
    print(f"fraud rate:           {fraud.mean():.3f}  ({int(fraud.sum())} / {len(df)})")
    print(f"rules decline rate:   {rules.mean():.3f}  ({declined})")
    print(f"rules precision:      {caught / max(declined, 1):.3f}")
    print(f"rules recall:         {caught / max(int(fraud.sum()), 1):.3f}")
    print(f"false declines:       {false_declines}")
    print(f"missed fraud:         {missed}")
    print(f"amount missing/NA:    {(amount_num.isna() | (df['amount'].astype(str).str.strip() == '')).mean():.3f}")
    print(f"category blank/unk:   {df['merchant_category'].astype(str).str.strip().isin(['', 'unk']).mean():.3f}")
    print(f"tenure missing:       {df['customer_tenure_days'].isna().mean():.3f}")
    print(f"duplicate txn_ids:    {df['txn_id'].duplicated().sum()}")
    print(f"ops_review_code mix:\n{df['ops_review_code'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
