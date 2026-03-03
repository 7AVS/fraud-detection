"""Transaction Fraud Detection — Streamlit Dashboard.

Predictive modeling on 284K real credit card transactions.
Precision-recall tradeoffs, cost-sensitive thresholds, and SHAP explainability.
"""

import streamlit as st

st.set_page_config(
    page_title="Transaction Fraud Detection",
    page_icon=None,
    layout="wide",
)

from dashboard.styles import CUSTOM_CSS
from engine.data_loader import load_data
from engine.features import engineer_features, get_feature_columns
from engine.models import split_and_scale, train_all_models

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="main-header">'
    "<h1>Transaction Fraud Detection</h1>"
    "<p>284,807 real credit card transactions | 0.17% fraud rate | "
    "3 models compared with cost-sensitive evaluation</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Load and prepare data
df_raw = load_data()
df = engineer_features(df_raw)
feature_cols = get_feature_columns(df)

X_train, X_test, y_train, y_test, scaler = split_and_scale(df, feature_cols)
results = train_all_models(X_train, X_test, y_train, y_test)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Explorer",
    "Model Comparison",
    "Threshold Tuning",
    "Explainability",
])

with tab1:
    from dashboard.explorer import render as render_explorer
    render_explorer(df_raw)

with tab2:
    from dashboard.modeling import render as render_modeling
    render_modeling(results, y_test)

with tab3:
    from dashboard.threshold import render as render_threshold
    render_threshold(results, y_test)

with tab4:
    from dashboard.explainability import render as render_explainability
    render_explainability(results, X_test, feature_cols)
