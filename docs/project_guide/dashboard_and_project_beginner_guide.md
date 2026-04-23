# Fintech Fraud Dashboard & Project Guide (Beginner Friendly)
## 1) What this project is trying to solve
Banks and fintech apps process huge numbers of transactions every day. A small percentage of these can be fraudulent. This project builds a practical fraud-monitoring system that:
- Scores transaction risk
- Flags suspicious behavior
- Tracks fraud and success trends in a dashboard
- Supports quick decisioning (approve / review / block)

In simple terms: it helps a team detect suspicious payments early and monitor fraud health over time.

## 2) Dataset used in this project
- Dataset: Credit Card Fraud Detection (European card transactions)
- Source: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
- Label column: `Class` (`1` = fraud, `0` = non-fraud)

Why this dataset was chosen:
- Standard public benchmark for fraud use cases
- Clean binary fraud labels for supervised ML
- Fast enough for demo and learning workflows
- Good for combining rules + ML + anomaly detection

## 3) What the pipeline does end-to-end
The project pipeline does the following:
1. Loads transaction data (or synthetic fallback if download fails)
2. Samples and enriches transactions with realistic fields:
   - channel
   - merchant category
   - transaction velocity
   - amount behavior ratios
3. Trains:
   - a fraud classifier (supervised)
   - an anomaly detector (unsupervised)
4. Applies rule-based checks for known risky patterns
5. Combines all signals into one hybrid risk score
6. Writes monitoring data to SQLite
7. Serves dashboard insights via Streamlit

## 4) How risk scoring works (core logic)
The final risk score blends three components:
- 60% ML fraud probability
- 20% anomaly risk
- 20% rule score

Risk bands:
- `low` (<45) → approve
- `medium` (45 to <75) → review
- `high` (>=75) → block

Why hybrid is useful:
- Rules are interpretable and catch obvious abuse
- ML learns fraud patterns from labeled history
- Anomaly model catches unusual behavior not seen before

## 5) What the dashboard shows
The dashboard provides:
- Filtered transaction count, fraud count/rate, success rate, average risk
- Fraud and success trends over time
- Risk-band distribution (low/medium/high)
- Channel-level success and risk statistics
- Top high-risk alert table
- Live transaction scoring form

The live form lets you input transaction details and get:
- risk score
- ML probability
- decision (approve/review/block)
- triggered rule reasons

## 6) Technologies used and why
- Python: core language for data + ML workflow
- pandas/numpy: data preparation and feature handling
- scikit-learn: classifier and anomaly detection models
- SQLite: lightweight monitoring database
- Streamlit: interactive dashboard and demo app
- joblib/pickle: model artifact persistence
- Docker: containerized local deployment option
- GitHub Actions: CI checks (compile + tests)

## 7) Project reasoning (design decisions)
Why this architecture is beginner-friendly and practical:
- Clear modular code (`data`, `models`, `rules`, `hybrid`, `database`, `pipeline`)
- Strong fallback behavior for reliability (synthetic data fallback)
- Fast bootstrap experience from inside Streamlit
- Separation between model training and dashboard serving
- Decision outputs are interpretable (risk band + rule reasons)

## 8) Interview prep (talk track + likely questions)
### 30-second explanation
"I built a hybrid fraud detection monitoring system using rules, supervised ML, and anomaly detection. I trained models on the credit card fraud dataset, generated combined risk scores, persisted predictions in SQLite, and built a Streamlit dashboard for fraud trends, alert monitoring, and live transaction scoring."

### Common interview questions with answer direction
1. Why hybrid instead of only ML?
   - Rules improve explainability and catch known fraud heuristics quickly.
   - Anomaly detection captures novel behavior.
   - ML handles non-linear patterns from labeled data.
2. How did you handle class imbalance?
   - Used stratified splitting/sampling and class-weighted logistic regression.
3. Why SQLite?
   - Lightweight, zero-admin, easy integration for demo monitoring.
4. How do you improve this for production?
   - Move to feature store + message queues + model service + monitoring stack.
   - Add drift monitoring, model versioning, and alert SLAs.
5. What metrics matter?
   - Recall (fraud catch rate), precision (false alert control), PR-AUC, ROC-AUC, decision throughput.

### Extra points to mention
- You included fallback data generation to keep the app deployable.
- You exposed clear decision categories (`approve/review/block`) for operations teams.
- You built both offline training and online scoring UX in one project.

## 9) Clear summary
This project is a full beginner-to-intermediate fraud analytics demo:
- It ingests and enriches transaction data
- Trains multiple fraud signals
- Produces a single interpretable risk score
- Stores results in SQL
- Visualizes operational KPIs in a live dashboard

It is strong as a portfolio project because it combines data engineering, ML, rule systems, product thinking, and deployable UI in one end-to-end flow.
