from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "Amount",
    "hour",
    "velocity_1h",
    "avg_amount_24h",
    "amount_to_avg_ratio",
]
CATEGORICAL_FEATURES = ["channel", "merchant_category"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_feature_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in MODEL_FEATURES if column not in transactions.columns]
    if missing_columns:
        raise ValueError(f"Missing required features: {missing_columns}")
    return transactions[MODEL_FEATURES].copy()

