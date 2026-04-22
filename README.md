# Fintech Fraud Detection + Transaction Monitoring System

This project builds a fintech-style fraud monitoring stack using a **public transaction dataset**, with:
- Rule-based risk detection
- Supervised fraud classification
- Unsupervised anomaly detection
- Hybrid risk scoring (rules + ML + anomaly)
- SQL-backed monitoring metrics
- Streamlit dashboard for trends, success rate, and live risk scoring

## Dataset choice
This implementation uses the **Credit Card Fraud Detection (European card dataset)** via a public mirror:
- `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

Why this dataset:
- Smaller and easier to work with than IEEE-CIS
- Standard benchmark for fraud use cases
- Includes clear fraud labels (`Class`)

## Stack
- Python (`pandas`, `scikit-learn`)
- SQL (`SQLite`)
- Streamlit
- Docker
- GitHub Actions

## Project structure
- `src/fraud_monitoring/data.py`: dataset ingestion + feature enrichment
- `src/fraud_monitoring/models.py`: classifier and anomaly model training
- `src/fraud_monitoring/rules.py`: rule engine
- `src/fraud_monitoring/hybrid.py`: hybrid risk scoring
- `src/fraud_monitoring/database.py`: SQL persistence
- `src/fraud_monitoring/dashboard_queries.py`: SQL metrics queries
- `src/fraud_monitoring/pipeline.py`: end-to-end build pipeline
- `app.py`: Streamlit dashboard
- `scripts/train_pipeline.py`: CLI to run pipeline

## Quick start (Windows PowerShell)
1) Install dependencies:
```powershell
python -m pip install -r requirements.txt
```

2) Train models + build monitoring DB:
```powershell
$env:PYTHONPATH = "src"
python scripts/train_pipeline.py --sample-size 50000
```

3) Launch dashboard:
```powershell
$env:PYTHONPATH = "src"
streamlit run app.py
```

The Streamlit app also supports one-click bootstrap when artifacts are not present.

## What the dashboard shows
- Fraud trends over time
- Transaction success rates
- Risk-band distribution (low / medium / high)
- Channel-level success and risk stats
- High-risk alert table
- Live transaction fraud scoring form

## Hybrid scoring logic
Final risk score is a weighted blend:
- `60%` ML fraud probability
- `20%` anomaly risk
- `20%` rule score

Risk bands:
- `low` (<45): approve
- `medium` (45-74.99): review
- `high` (>=75): block

## Docker
Build:
```powershell
docker build -t fintech-fraud-monitor .
```

Run:
```powershell
docker run --rm -p 8501:8501 fintech-fraud-monitor
```

Then open `http://localhost:8501`.

## Streamlit Cloud deployment idea
1) Push this repo to GitHub.
2) In Streamlit Community Cloud, create app from your repo.
3) Set entrypoint to `app.py`.
4) First run: click **Build demo artifacts** to create model/database assets.

## CI
GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- dependency install
- Python syntax compile smoke check
- unit tests (`pytest`)
