from __future__ import annotations

import random
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fraud_monitoring.config import ANOMALY_PATH, CLASSIFIER_PATH, DB_PATH
from fraud_monitoring.dashboard_queries import get_overview_kpis
from fraud_monitoring.database import append_live_prediction, load_monitoring_frame
from fraud_monitoring.hybrid import HybridFraudDetector
from fraud_monitoring.pipeline import run_training_pipeline

st.set_page_config(page_title="Fintech Fraud Detection + Monitoring", layout="wide")
st.title("Fintech Fraud Detection + Transaction Monitoring")
st.caption("Hybrid system: rule-based heuristics + ML classifier + anomaly detection")
st.caption("Made by Rahul Ega")
st.caption(
    "Dataset: Credit Card Fraud Detection (European card transactions) "
    "from https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
)


def _artifacts_ready() -> bool:
    return DB_PATH.exists() and CLASSIFIER_PATH.exists() and ANOMALY_PATH.exists()


@st.cache_resource
def _load_detector() -> HybridFraudDetector:
    return HybridFraudDetector.from_artifacts()


@st.cache_data(ttl=60)
def _load_monitoring_data() -> pd.DataFrame:
    return load_monitoring_frame()


def _build_trend_frame(filtered_data: pd.DataFrame) -> pd.DataFrame:
    if filtered_data.empty:
        return pd.DataFrame()

    trend_frame = (
        filtered_data.assign(
            is_fraud_filled=filtered_data["is_fraud"].fillna(0).astype(float),
            is_success_filled=filtered_data["is_success"].fillna(0).astype(float),
            risk_score_filled=filtered_data["risk_score"].fillna(0).astype(float),
        )
        .groupby("transaction_date", as_index=False)
        .agg(
            total_transactions=("transaction_id", "count"),
            fraud_transactions=("is_fraud_filled", "sum"),
            success_rate_pct=("is_success_filled", "mean"),
            avg_risk_score=("risk_score_filled", "mean"),
        )
        .sort_values("transaction_date")
    )
    trend_frame["fraud_transactions"] = trend_frame["fraud_transactions"].round().astype(int)
    trend_frame["fraud_rate_pct"] = (
        trend_frame["fraud_transactions"] / trend_frame["total_transactions"] * 100.0
    )
    trend_frame["success_rate_pct"] = trend_frame["success_rate_pct"] * 100.0
    trend_frame[["fraud_rate_pct", "success_rate_pct", "avg_risk_score"]] = (
        trend_frame[["fraud_rate_pct", "success_rate_pct", "avg_risk_score"]].round(3)
    )
    trend_frame = trend_frame.rename(columns={"transaction_date": "txn_date"})
    trend_frame["txn_date"] = pd.to_datetime(trend_frame["txn_date"])
    return trend_frame


def _build_risk_distribution(filtered_data: pd.DataFrame) -> pd.DataFrame:
    if filtered_data.empty:
        return pd.DataFrame(columns=["risk_band", "transactions"])

    risk_distribution = (
        filtered_data["risk_band"]
        .dropna()
        .value_counts()
        .rename_axis("risk_band")
        .reset_index(name="transactions")
    )
    risk_order = {"low": 0, "medium": 1, "high": 2}
    risk_distribution["risk_order"] = risk_distribution["risk_band"].map(risk_order).fillna(99)
    risk_distribution = risk_distribution.sort_values(
        ["risk_order", "transactions"],
        ascending=[True, False],
    )
    return risk_distribution.drop(columns=["risk_order"]).reset_index(drop=True)


def _build_channel_success_rates(filtered_data: pd.DataFrame) -> pd.DataFrame:
    if filtered_data.empty:
        return pd.DataFrame(columns=["channel", "transactions", "success_rate_pct", "avg_risk_score"])

    channel_rates = (
        filtered_data.assign(
            is_success_filled=filtered_data["is_success"].fillna(0).astype(float),
            risk_score_filled=filtered_data["risk_score"].fillna(0).astype(float),
        )
        .groupby("channel", as_index=False)
        .agg(
            transactions=("transaction_id", "count"),
            success_rate_pct=("is_success_filled", "mean"),
            avg_risk_score=("risk_score_filled", "mean"),
        )
        .sort_values("transactions", ascending=False)
    )
    channel_rates["success_rate_pct"] = (channel_rates["success_rate_pct"] * 100.0).round(3)
    channel_rates["avg_risk_score"] = channel_rates["avg_risk_score"].round(3)
    return channel_rates


def _build_recent_alerts(filtered_data: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    if filtered_data.empty:
        return pd.DataFrame()

    alert_columns = [
        "transaction_id",
        "transaction_timestamp",
        "account_id",
        "channel",
        "merchant_category",
        "Amount",
        "risk_score",
        "risk_band",
        "decision",
        "ml_probability",
        "rule_score",
        "rule_reasons",
    ]
    alerts = filtered_data.loc[filtered_data["risk_score"].fillna(0) >= 45, alert_columns].copy()
    if alerts.empty:
        return alerts
    return alerts.sort_values("risk_score", ascending=False).head(limit).reset_index(drop=True)


def _run_bootstrap(sample_size: int) -> bool:
    try:
        with st.spinner("Preparing dataset, training models, and building monitoring database..."):
            summary = run_training_pipeline(sample_size=sample_size)
    except Exception as exc:
        st.error("Bootstrap failed. Try a smaller sample size (for example 8,000).")
        st.exception(exc)
        return False
    st.success("Bootstrap complete.")
    st.json(summary)
    _load_monitoring_data.clear()
    _load_detector.clear()
    return True


if not _artifacts_ready():
    st.warning("Model/database artifacts are missing. Build baseline artifacts first.")
    st.caption(
        "For Streamlit Cloud, start with 8,000-20,000 rows for faster first-time setup."
    )
    bootstrap_sample_size = st.slider(
        "Bootstrap sample size",
        min_value=2_000,
        max_value=30_000,
        value=8_000,
        step=2_000,
    )
    if st.button("Build demo artifacts", type="primary"):
        if _run_bootstrap(sample_size=bootstrap_sample_size):
            st.rerun()
    st.stop()


if st.sidebar.button("Refresh monitoring data"):
    _load_monitoring_data.clear()

raw_monitoring_data = _load_monitoring_data()
if raw_monitoring_data.empty:
    st.warning("No monitoring rows found. Rebuild artifacts from the bootstrap action.")
    st.stop()

raw_monitoring_data["transaction_timestamp"] = pd.to_datetime(
    raw_monitoring_data["transaction_timestamp"], errors="coerce"
)
raw_monitoring_data = raw_monitoring_data.dropna(subset=["transaction_timestamp"])
raw_monitoring_data["transaction_date"] = raw_monitoring_data["transaction_timestamp"].dt.date

min_date = raw_monitoring_data["transaction_date"].min()
max_date = raw_monitoring_data["transaction_date"].max()

selected_date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(selected_date_range, tuple):
    start_date, end_date = selected_date_range
else:
    start_date = selected_date_range
    end_date = selected_date_range

available_channels = sorted(raw_monitoring_data["channel"].dropna().unique().tolist())
selected_channels = st.sidebar.multiselect(
    "Channel",
    options=available_channels,
    default=available_channels,
)
available_risk_bands = ["low", "medium", "high"]
selected_risk_bands = st.sidebar.multiselect(
    "Risk band",
    options=available_risk_bands,
    default=available_risk_bands,
)

filtered_data = raw_monitoring_data[
    (raw_monitoring_data["transaction_date"] >= start_date)
    & (raw_monitoring_data["transaction_date"] <= end_date)
    & (raw_monitoring_data["channel"].isin(selected_channels))
    & (raw_monitoring_data["risk_band"].isin(selected_risk_bands))
].copy()

if filtered_data.empty:
    st.info("No rows match the selected filters.")
    st.stop()

overview = get_overview_kpis()
overview_row = overview.iloc[0].to_dict() if not overview.empty else {}

total_transactions = int(len(filtered_data))
fraud_transactions = int((filtered_data["is_fraud"] == 1).sum())
fraud_rate = (fraud_transactions / total_transactions) * 100 if total_transactions else 0.0
success_rate = float(filtered_data["is_success"].mean() * 100)
avg_risk_score = float(filtered_data["risk_score"].mean())

top_col1, top_col2, top_col3, top_col4 = st.columns(4)
top_col1.metric("Transactions (filtered)", f"{total_transactions:,}")
top_col2.metric("Fraud labels", f"{fraud_transactions:,}", f"{fraud_rate:.2f}%")
top_col3.metric("Success rate", f"{success_rate:.2f}%")
top_col4.metric("Avg risk score", f"{avg_risk_score:.2f}")

st.caption(
    "Global DB totals — "
    f"Transactions: {int(overview_row.get('total_transactions', 0)):,} | "
    f"Fraud labels: {int(overview_row.get('fraud_transactions', 0)):,} | "
    f"Success rate: {float(overview_row.get('success_rate_pct', 0.0)):.2f}%"
)

trend_frame = _build_trend_frame(filtered_data)

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Fraud trends over time")
    if trend_frame.empty:
        st.info("No trend data available.")
    else:
        st.line_chart(
            trend_frame.set_index("txn_date")[["fraud_rate_pct", "success_rate_pct"]],
            use_container_width=True,
        )

with chart_right:
    st.subheader("Risk-band distribution")
    risk_distribution = _build_risk_distribution(filtered_data)
    if risk_distribution.empty:
        st.info("No risk distribution data available.")
    else:
        st.bar_chart(
            risk_distribution.set_index("risk_band")["transactions"],
            use_container_width=True,
        )

channel_perf_col, alert_col = st.columns(2)

with channel_perf_col:
    st.subheader("Transaction success rate by channel")
    channel_rates = _build_channel_success_rates(filtered_data)
    if channel_rates.empty:
        st.info("No channel-level data available.")
    else:
        st.dataframe(channel_rates, use_container_width=True, hide_index=True)

with alert_col:
    st.subheader("Top risk alerts")
    recent_alerts = _build_recent_alerts(filtered_data, limit=20)
    if recent_alerts.empty:
        st.info("No alerts above threshold.")
    else:
        st.dataframe(recent_alerts, use_container_width=True, hide_index=True)

st.subheader("Live fraud prediction")
with st.form("live_prediction_form", clear_on_submit=False):
    form_col1, form_col2, form_col3 = st.columns(3)
    amount = form_col1.number_input("Amount", min_value=0.0, value=120.0, step=10.0)
    hour = form_col1.slider("Hour of day", min_value=0, max_value=23, value=12)

    velocity_1h = form_col2.slider("Transactions in last hour", min_value=1, max_value=10, value=1)
    avg_amount_24h = form_col2.number_input(
        "Average amount last 24h",
        min_value=1.0,
        value=100.0,
        step=5.0,
    )

    channel = form_col3.selectbox("Channel", options=available_channels)
    merchant_categories = sorted(raw_monitoring_data["merchant_category"].dropna().unique().tolist())
    merchant_category = form_col3.selectbox("Merchant category", options=merchant_categories)

    log_to_database = st.checkbox("Log prediction into monitoring database", value=True)
    submitted = st.form_submit_button("Score transaction", type="primary")

if submitted:
    average_amount_safe = max(float(avg_amount_24h), 1.0)
    feature_row = pd.DataFrame(
        [
            {
                "Amount": float(amount),
                "hour": int(hour),
                "velocity_1h": float(velocity_1h),
                "avg_amount_24h": average_amount_safe,
                "amount_to_avg_ratio": float(amount) / average_amount_safe,
                "channel": channel,
                "merchant_category": merchant_category,
            }
        ]
    )

    detector = _load_detector()
    prediction_row = detector.score_transactions(feature_row).iloc[0].to_dict()

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Risk score", f"{prediction_row['risk_score']:.2f}")
    result_col2.metric("ML fraud probability", f"{prediction_row['ml_probability']:.3f}")
    result_col3.metric("Decision", prediction_row["decision"].upper())
    st.write(f"**Risk band:** `{prediction_row['risk_band']}`")
    st.write(f"**Triggered rules:** {prediction_row['rule_reasons']}")

    if log_to_database:
        now_iso = pd.Timestamp.utcnow().isoformat()
        transaction_id = f"LIVE{uuid.uuid4().hex[:10].upper()}"
        transaction_row = {
            "transaction_id": transaction_id,
            "transaction_timestamp": now_iso,
            "account_id": random.randint(900_000, 999_999),
            "channel": channel,
            "merchant_category": merchant_category,
            "Amount": float(amount),
            "hour": int(hour),
            "velocity_1h": float(velocity_1h),
            "avg_amount_24h": average_amount_safe,
            "amount_to_avg_ratio": float(amount) / average_amount_safe,
            "is_fraud": None,
            "is_success": int(prediction_row["decision"] != "block"),
        }
        prediction_record = {
            "transaction_id": transaction_id,
            "ml_probability": float(prediction_row["ml_probability"]),
            "anomaly_score": float(prediction_row["anomaly_score"]),
            "anomaly_flag": int(prediction_row["anomaly_flag"]),
            "rule_score": float(prediction_row["rule_score"]),
            "risk_score": float(prediction_row["risk_score"]),
            "risk_band": str(prediction_row["risk_band"]),
            "decision": str(prediction_row["decision"]),
            "rule_reasons": str(prediction_row["rule_reasons"]),
        }
        append_live_prediction(transaction_row=transaction_row, prediction_row=prediction_record)
        _load_monitoring_data.clear()
        st.success("Prediction logged to monitoring database.")
