"""SHAP-based model explainability."""

import numpy as np
import streamlit as st
import shap


@st.cache_data(show_spinner="Computing SHAP values...")
def compute_shap_values(
    _model, _X_sample: np.ndarray, feature_names: list[str],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for a sample of test data.

    Returns (shap_values, X_sample) for plotting.
    Uses TreeExplainer for tree models, LinearExplainer for LogReg.
    """
    if model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(_model, _X_sample)
        sv = explainer.shap_values(_X_sample)
    else:
        explainer = shap.TreeExplainer(_model)
        sv = explainer.shap_values(_X_sample)
        # XGBoost may return a list for binary classification
        if isinstance(sv, list):
            sv = sv[1]

    return sv, _X_sample
