from __future__ import annotations
import os
import tempfile

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_storage_root() -> Path:
    configured_root = os.getenv("FRAUD_MONITORING_STORAGE_DIR")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    if os.access(PROJECT_ROOT, os.W_OK):
        return PROJECT_ROOT
    return Path(tempfile.gettempdir()) / "fintech_fraud_monitoring"


STORAGE_ROOT = _resolve_storage_root()
DATA_DIR = STORAGE_ROOT / "data"
ARTIFACTS_DIR = STORAGE_ROOT / "artifacts"

RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
PROCESSED_TRANSACTIONS_PATH = DATA_DIR / "processed_transactions.csv"
DB_PATH = DATA_DIR / "fraud_monitoring.db"

CLASSIFIER_PATH = ARTIFACTS_DIR / "classifier.joblib"
ANOMALY_PATH = ARTIFACTS_DIR / "anomaly.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
PIPELINE_SUMMARY_PATH = ARTIFACTS_DIR / "pipeline_summary.json"

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DEFAULT_SAMPLE_SIZE = 12_000
DOWNLOAD_TIMEOUT_SECONDS = 30
SYNTHETIC_DATASET_ROWS = 30_000
RANDOM_SEED = 42


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

