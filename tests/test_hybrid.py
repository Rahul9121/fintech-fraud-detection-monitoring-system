import numpy as np

from fraud_monitoring.hybrid import (
    anomaly_to_risk,
    combine_risk_components,
    decision_from_band,
    risk_band,
)


def test_combined_risk_increases_with_signal_strength() -> None:
    low = combine_risk_components(ml_probability=0.05, anomaly_risk=0.1, rule_score=5)
    high = combine_risk_components(ml_probability=0.9, anomaly_risk=0.9, rule_score=85)
    assert float(high) > float(low)
    assert float(high) <= 100.0


def test_risk_band_and_decision_mapping() -> None:
    assert risk_band(20.0) == "low"
    assert risk_band(55.0) == "medium"
    assert risk_band(90.0) == "high"

    assert decision_from_band("low") == "approve"
    assert decision_from_band("medium") == "review"
    assert decision_from_band("high") == "block"


def test_anomaly_to_risk_monotonicity() -> None:
    anomaly_scores = np.array([0.8, 0.2, -0.3, -1.1])
    risks = anomaly_to_risk(anomaly_scores)
    assert risks[0] < risks[1] < risks[2] < risks[3]

