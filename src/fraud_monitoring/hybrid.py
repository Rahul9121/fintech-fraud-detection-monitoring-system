from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MODEL_FEATURES, NUMERIC_FEATURES
from .models import load_artifacts
from .rules import score_rules


def anomaly_to_risk(anomaly_decision_scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(3.0 * anomaly_decision_scores))


def combine_risk_components(
    ml_probability: np.ndarray | float,
    anomaly_risk: np.ndarray | float,
    rule_score: np.ndarray | float,
) -> np.ndarray:
    combined = (np.asarray(ml_probability) * 100 * 0.6) + (
        np.asarray(anomaly_risk) * 100 * 0.2
    ) + (np.asarray(rule_score) * 0.2)
    return np.clip(combined, 0, 100)


def risk_band(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def decision_from_band(band: str) -> str:
    if band == "high":
        return "block"
    if band == "medium":
        return "review"
    return "approve"


class HybridFraudDetector:
    def __init__(self, classifier, anomaly_detector) -> None:
        self.classifier = classifier
        self.anomaly_detector = anomaly_detector

    @classmethod
    def from_artifacts(cls) -> "HybridFraudDetector":
        classifier, anomaly_detector = load_artifacts()
        return cls(classifier=classifier, anomaly_detector=anomaly_detector)

    def score_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        missing_features = [feature for feature in MODEL_FEATURES if feature not in transactions.columns]
        if missing_features:
            raise ValueError(f"Missing required model features: {missing_features}")

        model_input = transactions[MODEL_FEATURES].copy()

        rule_results = [score_rules(row) for row in model_input.to_dict(orient="records")]
        rule_scores = np.array([result.score for result in rule_results], dtype=float)
        rule_reasons = [
            "; ".join(result.reasons) if result.reasons else "No high-risk rule triggered"
            for result in rule_results
        ]

        ml_probability = self.classifier.predict_proba(model_input)[:, 1]
        anomaly_decision_scores = self.anomaly_detector.decision_function(model_input[NUMERIC_FEATURES])
        anomaly_flags = (self.anomaly_detector.predict(model_input[NUMERIC_FEATURES]) == -1).astype(int)
        anomaly_risk = anomaly_to_risk(anomaly_decision_scores)

        risk_score = combine_risk_components(
            ml_probability=ml_probability,
            anomaly_risk=anomaly_risk,
            rule_score=rule_scores,
        )
        risk_score = np.where((anomaly_flags == 1) & (risk_score < 55), risk_score + 10, risk_score)
        risk_score = np.clip(risk_score, 0, 100)

        risk_bands = [risk_band(value) for value in risk_score]
        decisions = [decision_from_band(band) for band in risk_bands]

        return pd.DataFrame(
            {
                "ml_probability": ml_probability,
                "anomaly_score": anomaly_decision_scores,
                "anomaly_flag": anomaly_flags,
                "rule_score": rule_scores,
                "risk_score": risk_score,
                "risk_band": risk_bands,
                "decision": decisions,
                "rule_reasons": rule_reasons,
            }
        )

