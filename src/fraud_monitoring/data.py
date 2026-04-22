from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_URL, RANDOM_SEED, RAW_DATA_PATH


def load_public_dataset(force_download: bool = False) -> pd.DataFrame:
    if force_download and RAW_DATA_PATH.exists():
        RAW_DATA_PATH.unlink()
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)
    dataset = pd.read_csv(DATA_URL)
    dataset.to_csv(RAW_DATA_PATH, index=False)
    return dataset


def stratified_sample(dataset: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sample_size >= len(dataset):
        return dataset.copy()

    fraud_transactions = dataset[dataset["Class"] == 1]
    non_fraud_transactions = dataset[dataset["Class"] == 0]

    target_non_fraud = max(sample_size - len(fraud_transactions), 0)
    sampled_non_fraud = non_fraud_transactions.sample(
        n=min(target_non_fraud, len(non_fraud_transactions)),
        random_state=RANDOM_SEED,
    )

    sampled = pd.concat([fraud_transactions, sampled_non_fraud], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def _add_account_rollups(account_transactions: pd.DataFrame) -> pd.DataFrame:
    rolling_view = account_transactions.sort_values("transaction_timestamp").copy()
    rolling_view = rolling_view.set_index("transaction_timestamp")
    rolling_view["velocity_1h"] = (
        rolling_view["Amount"].rolling("1h").count().fillna(1.0).astype(float)
    )
    rolling_view["avg_amount_24h"] = (
        rolling_view["Amount"].rolling("24h").mean().fillna(rolling_view["Amount"]).astype(float)
    )
    return rolling_view.reset_index()


def enrich_transactions(transactions: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    enriched = transactions.copy().reset_index(drop=True)

    enriched["transaction_timestamp"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        enriched["Time"], unit="s"
    )
    enriched["transaction_id"] = [
        f"TXN{index:08d}" for index in range(1, len(enriched) + 1)
    ]
    enriched["account_id"] = rng.integers(100_000, 999_999, size=len(enriched))

    enriched["channel"] = rng.choice(
        ["web", "mobile", "pos", "atm"], size=len(enriched), p=[0.42, 0.32, 0.22, 0.04]
    )
    enriched["merchant_category"] = rng.choice(
        ["grocery", "electronics", "travel", "health", "entertainment", "utilities"],
        size=len(enriched),
        p=[0.24, 0.19, 0.14, 0.13, 0.12, 0.18],
    )

    enriched = (
        enriched.sort_values(["account_id", "transaction_timestamp"])
        .groupby("account_id", group_keys=False)
        .apply(_add_account_rollups)
        .reset_index(drop=True)
    )

    enriched["hour"] = enriched["transaction_timestamp"].dt.hour.astype(int)
    enriched["day"] = enriched["transaction_timestamp"].dt.strftime("%Y-%m-%d")
    enriched["amount_to_avg_ratio"] = enriched["Amount"] / (enriched["avg_amount_24h"] + 1e-6)

    base_success_probability = np.where(enriched["Class"].eq(1), 0.55, 0.985)
    friction_penalty = (
        (enriched["velocity_1h"] >= 4).astype(float) * 0.08
        + (enriched["Amount"] > 1_000).astype(float) * 0.05
        + enriched["channel"].eq("atm").astype(float) * 0.03
    )
    success_probability = np.clip(base_success_probability - friction_penalty, 0.05, 0.995)
    enriched["is_success"] = rng.binomial(1, success_probability).astype(int)
    enriched["is_fraud"] = enriched["Class"].astype(int)

    return enriched.sort_values("transaction_timestamp").reset_index(drop=True)


def prepare_transactions(
    sample_size: int,
    force_download: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    source = load_public_dataset(force_download=force_download)
    sampled = stratified_sample(source, sample_size=sample_size)
    return enrich_transactions(sampled, seed=seed)

