#!/usr/bin/env python3
"""Weak baseline for the Dexwin Pay live assessment.

This is a starting point, not a target design. It runs in under a minute on
a laptop so you can get numbers on the board quickly, then improve it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    n_loaded = len(df)

    # Bare-minimum parse so the script does not crash on "$1,200.00".
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .replace("", pd.NA)
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Drop anything incomplete. Easy, and we lose a lot of rows.
    df = df.dropna().copy()

    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])

    # Encode every leftover object column, including IDs and free-form fields.
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=250)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print("=== Dexwin Pay baseline ===")
    print(f"rows loaded:       {n_loaded}")
    print(f"rows after dropna: {len(df)}")
    print(f"accuracy:          {acc:.4f}")
    print("confusion matrix (rows=true 0/1, cols=pred 0/1):")
    print(confusion_matrix(y_test, pred))
    # TODO: product asked for fewer false declines — add that later.


if __name__ == "__main__":
    main()
