from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ANOMALY_PATH, CLASSIFIER_PATH, METRICS_PATH, MODEL_METADATA_PATH, RANDOM_SEED
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
try:
    import joblib
except ModuleNotFoundError:
    joblib = None


@dataclass
class TrainedModelBundle:
    classifier: Pipeline
    anomaly_detector: Pipeline
    metrics: dict[str, float]


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_classifier_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", _build_one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="liblinear",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def build_anomaly_pipeline() -> Pipeline:
    anomaly_model = IsolationForest(
        n_estimators=120,
        contamination=0.02,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    return Pipeline(steps=[("scaler", StandardScaler()), ("anomaly", anomaly_model)])


def _safe_metric(metric_fn, y_true: pd.Series, y_score: np.ndarray) -> float:
    try:
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return float("nan")


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> TrainedModelBundle:
    classifier = build_classifier_pipeline()
    classifier.fit(X_train, y_train)

    ml_probabilities = classifier.predict_proba(X_test)[:, 1]
    predicted_labels = (ml_probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predicted_labels,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "roc_auc": _safe_metric(roc_auc_score, y_test, ml_probabilities),
        "pr_auc": _safe_metric(average_precision_score, y_test, ml_probabilities),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "test_positive_rate": float(np.mean(y_test)),
    }

    anomaly_detector = build_anomaly_pipeline()
    anomaly_training_data = X_train.loc[y_train == 0, NUMERIC_FEATURES]
    if anomaly_training_data.empty:
        anomaly_training_data = X_train[NUMERIC_FEATURES]
    anomaly_detector.fit(anomaly_training_data)

    return TrainedModelBundle(
        classifier=classifier,
        anomaly_detector=anomaly_detector,
        metrics=metrics,
    )


def _dump_artifact(model: Pipeline, artifact_path: Path) -> None:
    if joblib is not None:
        joblib.dump(model, artifact_path)
        return
    with artifact_path.open("wb") as artifact_file:
        pickle.dump(model, artifact_file, protocol=pickle.HIGHEST_PROTOCOL)


def _load_artifact(artifact_path: Path) -> Pipeline:
    if joblib is not None:
        return joblib.load(artifact_path)
    try:
        with artifact_path.open("rb") as artifact_file:
            return pickle.load(artifact_file)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load model artifacts without joblib. "
            "Rebuild artifacts from the app bootstrap or install joblib."
        ) from exc


def save_artifacts(bundle: TrainedModelBundle) -> None:
    _dump_artifact(bundle.classifier, CLASSIFIER_PATH)
    _dump_artifact(bundle.anomaly_detector, ANOMALY_PATH)

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(bundle.metrics, indent=2), encoding="utf-8")


def load_artifacts() -> tuple[Pipeline, Pipeline]:
    classifier = _load_artifact(CLASSIFIER_PATH)
    anomaly_detector = _load_artifact(ANOMALY_PATH)
    return classifier, anomaly_detector

