"""Tab 3: Cost-sensitive threshold optimization."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from engine.evaluation import threshold_sweep, find_optimal_threshold, cost_at_threshold
from dashboard.styles import COLORS, PLOTLY_LAYOUT, AXIS_DEFAULTS, CUSTOM_CSS


def render(results: dict, y_test: np.ndarray):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Cost-Sensitive Threshold Tuning</div>', unsafe_allow_html=True)
    st.caption(
        "Accuracy doesn't tell you the cost of errors. A missed fraud (false negative) "
        "might cost $5,000 in chargebacks. A false alarm (false positive) might cost $50 in "
        "investigation time. The optimal threshold depends on these costs."
    )

    # Model selector
    model_name = st.selectbox("Select model", list(results.keys()), index=2)
    y_prob = results[model_name]["y_prob"]

    # Cost inputs
    col1, col2 = st.columns(2)
    with col1:
        fn_cost = st.number_input(
            "Cost per missed fraud (FN)", value=5000, min_value=1, step=500,
            help="e.g. chargeback amount + investigation + reputation",
        )
    with col2:
        fp_cost = st.number_input(
            "Cost per false alarm (FP)", value=50, min_value=1, step=10,
            help="e.g. analyst time to investigate a legitimate transaction",
        )

    # Find optimal
    optimal = find_optimal_threshold(y_test, y_prob, fn_cost, fp_cost)
    default = cost_at_threshold(y_test, y_prob, 0.5, fn_cost, fp_cost)

    # KPI comparison
    st.markdown("---")
    cols = st.columns(4)
    labels = ["Optimal Threshold", "Total Cost (Optimal)", "Total Cost (Default 0.5)", "Cost Reduction"]
    values = [
        f"{optimal['threshold']:.3f}",
        f"${optimal['total_cost']:,.0f}",
        f"${default['total_cost']:,.0f}",
        f"{(1 - optimal['total_cost'] / max(default['total_cost'], 1)) * 100:.1f}%",
    ]
    for col, label, value in zip(cols, labels, values):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Threshold sweep visualization
    sweep = threshold_sweep(y_test, y_prob, fn_cost, fp_cost)
    thresholds = [s["threshold"] for s in sweep]

    # Dual-axis: metrics + cost
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=thresholds, y=[s["precision"] for s in sweep],
        name="Precision", line=dict(color=COLORS["sage"], width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=thresholds, y=[s["recall"] for s in sweep],
        name="Recall", line=dict(color=COLORS["teal"], width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=thresholds, y=[s["f1"] for s in sweep],
        name="F1", line=dict(color=COLORS["slate"], width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=thresholds, y=[s["total_cost"] for s in sweep],
        name="Total Cost", line=dict(color=COLORS["rose"], width=2, dash="dot"),
    ), secondary_y=True)

    # Mark optimal threshold
    fig.add_vline(
        x=optimal["threshold"],
        line=dict(color=COLORS["taupe"], width=2, dash="dash"),
        annotation_text=f"Optimal: {optimal['threshold']:.3f}",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT, height=450,
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    fig.update_xaxes(**AXIS_DEFAULTS, title="Decision Threshold")
    fig.update_yaxes(**AXIS_DEFAULTS, title="Metric Value", range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(**AXIS_DEFAULTS, title="Total Cost ($)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix at optimal threshold
    st.markdown('<div class="section-header">Confusion Matrix at Optimal Threshold</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for col, label, data in [
        (col1, "Optimal Threshold", optimal),
        (col2, "Default (0.5)", default),
    ]:
        with col:
            st.caption(f"**{label}** — Threshold: {data['threshold']:.3f}")
            st.markdown(
                f"TP: {data['tp']:,} | FP: {data['fp']:,} | "
                f"FN: {data['fn']:,} | TN: {data['tn']:,}"
            )
            st.markdown(
                f"Precision: {data['precision']:.3f} | "
                f"Recall: {data['recall']:.3f} | "
                f"F1: {data['f1']:.3f}"
            )
            st.markdown(f"**Total cost: ${data['total_cost']:,.0f}**")
