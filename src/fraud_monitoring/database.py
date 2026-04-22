from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DB_PATH

TRANSACTION_COLUMNS = [
    "transaction_id",
    "transaction_timestamp",
    "account_id",
    "channel",
    "merchant_category",
    "Amount",
    "hour",
    "velocity_1h",
    "avg_amount_24h",
    "amount_to_avg_ratio",
    "is_fraud",
    "is_success",
]

PREDICTION_COLUMNS = [
    "transaction_id",
    "ml_probability",
    "anomaly_score",
    "anomaly_flag",
    "rule_score",
    "risk_score",
    "risk_band",
    "decision",
    "rule_reasons",
]


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            transaction_timestamp TEXT NOT NULL,
            account_id INTEGER NOT NULL,
            channel TEXT NOT NULL,
            merchant_category TEXT NOT NULL,
            Amount REAL NOT NULL,
            hour INTEGER NOT NULL,
            velocity_1h REAL NOT NULL,
            avg_amount_24h REAL NOT NULL,
            amount_to_avg_ratio REAL NOT NULL,
            is_fraud INTEGER,
            is_success INTEGER
        );
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            ml_probability REAL NOT NULL,
            anomaly_score REAL NOT NULL,
            anomaly_flag INTEGER NOT NULL,
            rule_score REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_band TEXT NOT NULL,
            decision TEXT NOT NULL,
            rule_reasons TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
        );
        CREATE INDEX IF NOT EXISTS idx_transactions_timestamp
            ON transactions (transaction_timestamp);
        CREATE INDEX IF NOT EXISTS idx_predictions_risk_score
            ON predictions (risk_score);
        """
    )


def reset_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS predictions;
        DROP TABLE IF EXISTS transactions;
        """
    )
    initialize_database(connection)


def write_monitoring_data(
    transactions: pd.DataFrame,
    predictions: pd.DataFrame,
    db_path: Path | str = DB_PATH,
) -> None:
    transaction_frame = transactions[TRANSACTION_COLUMNS].copy()
    transaction_frame["transaction_timestamp"] = pd.to_datetime(
        transaction_frame["transaction_timestamp"]
    ).astype(str)
    transaction_frame["is_fraud"] = transaction_frame["is_fraud"].astype(int)
    transaction_frame["is_success"] = transaction_frame["is_success"].astype(int)

    prediction_frame = predictions[PREDICTION_COLUMNS].copy()
    prediction_frame["anomaly_flag"] = prediction_frame["anomaly_flag"].astype(int)

    with get_connection(db_path) as connection:
        reset_database(connection)
        transaction_frame.to_sql("transactions", connection, if_exists="append", index=False)
        prediction_frame.to_sql("predictions", connection, if_exists="append", index=False)
        connection.commit()


def append_live_prediction(
    transaction_row: dict[str, Any],
    prediction_row: dict[str, Any],
    db_path: Path | str = DB_PATH,
) -> None:
    with get_connection(db_path) as connection:
        initialize_database(connection)
        pd.DataFrame([transaction_row])[TRANSACTION_COLUMNS].to_sql(
            "transactions", connection, if_exists="append", index=False
        )
        pd.DataFrame([prediction_row])[PREDICTION_COLUMNS].to_sql(
            "predictions", connection, if_exists="append", index=False
        )
        connection.commit()


def load_monitoring_frame(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    query = """
        SELECT
            t.transaction_id,
            t.transaction_timestamp,
            t.account_id,
            t.channel,
            t.merchant_category,
            t.Amount,
            t.hour,
            t.velocity_1h,
            t.avg_amount_24h,
            t.amount_to_avg_ratio,
            t.is_fraud,
            t.is_success,
            p.ml_probability,
            p.anomaly_score,
            p.anomaly_flag,
            p.rule_score,
            p.risk_score,
            p.risk_band,
            p.decision,
            p.rule_reasons
        FROM transactions t
        LEFT JOIN predictions p
            ON t.transaction_id = p.transaction_id
    """
    with get_connection(db_path) as connection:
        frame = pd.read_sql_query(query, connection)

    if frame.empty:
        return frame
    frame["transaction_timestamp"] = pd.to_datetime(frame["transaction_timestamp"], errors="coerce")
    return frame

