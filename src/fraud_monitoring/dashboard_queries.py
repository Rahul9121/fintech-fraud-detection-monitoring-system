from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DB_PATH
from .database import get_connection


def get_overview_kpis(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    query = """
        SELECT
            COUNT(*) AS total_transactions,
            SUM(CASE WHEN COALESCE(t.is_fraud, 0) = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
            ROUND(100.0 * AVG(CASE WHEN COALESCE(t.is_success, 0) = 1 THEN 1.0 ELSE 0.0 END), 2)
                AS success_rate_pct,
            ROUND(AVG(COALESCE(p.risk_score, 0.0)), 2) AS avg_risk_score
        FROM transactions t
        LEFT JOIN predictions p
            ON t.transaction_id = p.transaction_id
    """
    with get_connection(db_path) as connection:
        return pd.read_sql_query(query, connection)


def get_fraud_trend(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    query = """
        SELECT
            DATE(t.transaction_timestamp) AS txn_date,
            COUNT(*) AS total_transactions,
            SUM(CASE WHEN COALESCE(t.is_fraud, 0) = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
            ROUND(
                100.0 * SUM(CASE WHEN COALESCE(t.is_fraud, 0) = 1 THEN 1 ELSE 0 END) / COUNT(*), 3
            ) AS fraud_rate_pct,
            ROUND(
                100.0 * AVG(CASE WHEN COALESCE(t.is_success, 0) = 1 THEN 1.0 ELSE 0.0 END), 3
            ) AS success_rate_pct,
            ROUND(AVG(COALESCE(p.risk_score, 0.0)), 3) AS avg_risk_score
        FROM transactions t
        LEFT JOIN predictions p
            ON t.transaction_id = p.transaction_id
        GROUP BY DATE(t.transaction_timestamp)
        ORDER BY txn_date
    """
    with get_connection(db_path) as connection:
        return pd.read_sql_query(query, connection)


def get_risk_distribution(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    query = """
        SELECT
            p.risk_band,
            COUNT(*) AS transactions
        FROM predictions p
        GROUP BY p.risk_band
        ORDER BY transactions DESC
    """
    with get_connection(db_path) as connection:
        return pd.read_sql_query(query, connection)


def get_channel_success_rates(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    query = """
        SELECT
            t.channel,
            COUNT(*) AS transactions,
            ROUND(
                100.0 * AVG(CASE WHEN COALESCE(t.is_success, 0) = 1 THEN 1.0 ELSE 0.0 END), 3
            ) AS success_rate_pct,
            ROUND(AVG(COALESCE(p.risk_score, 0.0)), 3) AS avg_risk_score
        FROM transactions t
        LEFT JOIN predictions p
            ON t.transaction_id = p.transaction_id
        GROUP BY t.channel
        ORDER BY transactions DESC
    """
    with get_connection(db_path) as connection:
        return pd.read_sql_query(query, connection)


def get_recent_alerts(db_path: Path | str = DB_PATH, limit: int = 50) -> pd.DataFrame:
    query = """
        SELECT
            t.transaction_id,
            t.transaction_timestamp,
            t.account_id,
            t.channel,
            t.merchant_category,
            t.Amount,
            p.risk_score,
            p.risk_band,
            p.decision,
            p.ml_probability,
            p.rule_score,
            p.rule_reasons
        FROM transactions t
        JOIN predictions p
            ON t.transaction_id = p.transaction_id
        WHERE p.risk_score >= 45
        ORDER BY p.risk_score DESC
        LIMIT ?
    """
    with get_connection(db_path) as connection:
        return pd.read_sql_query(query, connection, params=[limit])

