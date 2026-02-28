"""Tab 4: SHAP-based model explainability."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from engine.explainability import compute_shap_values
from dashboard.styles import COLORS, CHART_COLORS, PLOTLY_LAYOUT, AXIS_DEFAULTS, CUSTOM_CSS


def render(results: dict, X_test: np.ndarray, feature_names: list[str]):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Importance — SHAP Analysis</div>', unsafe_allow_html=True)
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows how each feature contributes to "
        "individual predictions. Higher absolute SHAP values mean the feature has more "
        "influence on the model's fraud/legitimate decision."
    )

    model_name = st.selectbox(
        "Select model for SHAP analysis",
        ["XGBoost", "Random Forest", "Logistic Regression"],
        key="shap_model",
    )

    model = results[model_name]["model"]

    # Use a sample for SHAP (full dataset is too slow)
    sample_size = min(2000, len(X_test))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[idx]

    shap_vals, X_used = compute_shap_values(model, X_sample, feature_names, model_name)

    # Mean absolute SHAP values — bar chart
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:15]

    top_features = [feature_names[i] for i in sorted_idx]
    top_values = mean_abs_shap[sorted_idx]

    fig_bar = go.Figure(go.Bar(
        x=top_values[::-1],
        y=top_features[::-1],
        orientation="h",
        marker_color=COLORS["slate"],
    ))
    fig_bar.update_layout(
        **PLOTLY_LAYOUT, height=450,
        xaxis=dict(**AXIS_DEFAULTS, title="Mean |SHAP Value|"),
        yaxis=dict(**AXIS_DEFAULTS),
        title=dict(text="Top 15 Features by Importance", font=dict(size=14, color=COLORS["dark"])),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # SHAP beeswarm-style (simplified: scatter plot of SHAP vs feature value for top features)
    st.markdown('<div class="section-header">Feature Effect on Predictions</div>', unsafe_allow_html=True)
    st.caption("Each dot is a transaction. X-axis shows SHAP value (positive = pushes toward fraud). "
               "Color shows the feature's raw value (high = rose, low = sage).")

    selected_feat = st.selectbox("Select feature to inspect", top_features, key="shap_feat")
    feat_idx = feature_names.index(selected_feat)

    feat_shap = shap_vals[:, feat_idx]
    feat_vals = X_used[:, feat_idx]

    # Normalize for color
    vmin, vmax = feat_vals.min(), feat_vals.max()
    if vmax - vmin > 0:
        normed = (feat_vals - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(feat_vals)

    colors = [
        f"rgb({int(122 + (153 - 122) * n)}, {int(158 + (123 - 158) * n)}, {int(142 + (122 - 142) * n)})"
        for n in normed
    ]

    fig_scatter = go.Figure(go.Scatter(
        x=feat_shap,
        y=np.random.RandomState(0).normal(0, 0.15, len(feat_shap)),
        mode="markers",
        marker=dict(color=colors, size=4, opacity=0.6),
        hovertemplate=f"{selected_feat}: %{{customdata:.3f}}<br>SHAP: %{{x:.4f}}<extra></extra>",
        customdata=feat_vals,
    ))
    fig_scatter.add_vline(x=0, line=dict(color=COLORS["muted"], width=1, dash="dash"))
    fig_scatter.update_layout(
        **PLOTLY_LAYOUT, height=300,
        xaxis=dict(**AXIS_DEFAULTS, title=f"SHAP Value (impact on fraud prediction)"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
