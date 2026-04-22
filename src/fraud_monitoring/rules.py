from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class RuleResult:
    score: float
    reasons: list[str]


def score_rules(transaction: Mapping[str, object]) -> RuleResult:
    amount = float(transaction.get("Amount", 0.0))
    hour = int(transaction.get("hour", 0))
    velocity = float(transaction.get("velocity_1h", 1.0))
    ratio = float(transaction.get("amount_to_avg_ratio", 1.0))
    channel = str(transaction.get("channel", "")).lower()

    score = 0.0
    reasons: list[str] = []

    if amount >= 2_000:
        score += 30
        reasons.append("Very high transaction amount")
    elif amount >= 1_000:
        score += 20
        reasons.append("High transaction amount")

    if ratio >= 4:
        score += 25
        reasons.append("Amount is far above account baseline")
    elif ratio >= 2:
        score += 15
        reasons.append("Amount above normal account baseline")

    if velocity >= 6:
        score += 20
        reasons.append("High velocity in one-hour window")
    elif velocity >= 4:
        score += 12
        reasons.append("Elevated velocity in one-hour window")

    if hour <= 5:
        score += 10
        reasons.append("Transaction during high-risk hours")

    if channel == "atm" and amount > 500:
        score += 8
        reasons.append("Large ATM withdrawal pattern")

    return RuleResult(score=min(score, 100.0), reasons=reasons)

