from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_SAMPLE_SIZE,
    PIPELINE_SUMMARY_PATH,
    PROCESSED_TRANSACTIONS_PATH,
    RANDOM_SEED,
    ensure_directories,
)
from .data import prepare_transactions
from .database import write_monitoring_data
from .features import build_feature_frame
from .hybrid import HybridFraudDetector
from .models import save_artifacts, train_models


def run_training_pipeline(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    force_download: bool = False,
) -> dict[str, float | int]:
    ensure_directories()

    transactions = prepare_transactions(
        sample_size=sample_size,
        force_download=force_download,
        seed=RANDOM_SEED,
    )

    model_features = build_feature_frame(transactions)
    labels = transactions["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        model_features,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    trained_bundle = train_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    save_artifacts(trained_bundle)

    hybrid_detector = HybridFraudDetector(
        classifier=trained_bundle.classifier,
        anomaly_detector=trained_bundle.anomaly_detector,
    )
    predictions = hybrid_detector.score_transactions(model_features)
    predictions.insert(0, "transaction_id", transactions["transaction_id"].values)

    write_monitoring_data(transactions=transactions, predictions=predictions)

    combined_output = pd.concat(
        [
            transactions[["transaction_id", "transaction_timestamp", "Amount", "is_fraud", "is_success"]],
            predictions.drop(columns=["transaction_id"]),
        ],
        axis=1,
    )
    combined_output.to_csv(PROCESSED_TRANSACTIONS_PATH, index=False)

    summary: dict[str, float | int] = {
        "sample_size": int(sample_size),
        "rows_processed": int(len(transactions)),
        "fraud_rows": int(transactions["is_fraud"].sum()),
        "success_rate_pct": float(transactions["is_success"].mean() * 100),
        "avg_risk_score": float(predictions["risk_score"].mean()),
    }
    summary.update({f"metric_{key}": float(value) for key, value in trained_bundle.metrics.items()})

    PIPELINE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

