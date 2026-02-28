# Transaction Fraud Detection

Predictive modeling on 284,807 real European credit card transactions with a 0.17% fraud rate. Three models (Logistic Regression, Random Forest, XGBoost) compared using precision-recall analysis, cost-sensitive threshold optimization, and SHAP explainability.

## What This Demonstrates

- **Imbalanced classification** — handling extreme class imbalance (0.17% positive rate)
- **Precision-recall tradeoffs** — why accuracy is meaningless at this fraud rate
- **Cost-sensitive evaluation** — optimizing thresholds based on business costs (missed fraud vs. false alarms)
- **Model explainability** — SHAP values showing what drives fraud predictions

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dataset (~150MB) is downloaded automatically from OpenML on first run and cached locally.

## Architecture

```
engine/
  data_loader.py    — Load + cache creditcard fraud dataset from OpenML
  features.py       — Feature engineering (time patterns, amount, interactions)
  models.py         — Train LogReg, Random Forest, XGBoost with class balancing
  evaluation.py     — Cost matrix, threshold sweep, optimization
  explainability.py — SHAP values for any trained model

dashboard/
  styles.py         — Shared design tokens (Desaturated Cool palette)
  explorer.py       — Data overview, class distribution, feature distributions
  modeling.py       — PR curves, confusion matrices, F1 comparison
  threshold.py      — Interactive cost-sensitive threshold tuning
  explainability.py — SHAP feature importance and effect plots
```

## Dataset

Credit Card Fraud Detection from OpenML (originally from ULB machine learning group). 284,807 transactions over 2 days, 492 fraudulent. Features V1-V28 are PCA-transformed (anonymized), plus Time and Amount.

## Tech Stack

Python, scikit-learn, XGBoost, SHAP, Streamlit, Plotly
