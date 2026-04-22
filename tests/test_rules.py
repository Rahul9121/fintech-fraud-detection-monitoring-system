from fraud_monitoring.rules import score_rules


def test_rule_engine_assigns_high_score_to_high_risk_transaction() -> None:
    result = score_rules(
        {
            "Amount": 2_500,
            "hour": 2,
            "velocity_1h": 7,
            "amount_to_avg_ratio": 5.2,
            "channel": "atm",
        }
    )
    assert result.score >= 80
    assert len(result.reasons) >= 4


def test_rule_engine_assigns_low_score_to_low_risk_transaction() -> None:
    result = score_rules(
        {
            "Amount": 55,
            "hour": 13,
            "velocity_1h": 1,
            "amount_to_avg_ratio": 0.8,
            "channel": "web",
        }
    )
    assert result.score < 20
    assert result.reasons == []

