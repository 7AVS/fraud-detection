"""Tab 2: Model comparison — PR curves, confusion matrices."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dashboard.styles import COLORS, CHART_COLORS, PLOTLY_LAYOUT, AXIS_DEFAULTS, CUSTOM_CSS


def render(results: dict, y_test: np.ndarray):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Model summary KPIs
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{res["ap"]:.3f}</div>'
            f'<div class="kpi-label">{name}<br>Average Precision</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Precision-Recall curves
    st.markdown('<div class="section-header">Precision-Recall Curves</div>', unsafe_allow_html=True)
    st.caption("Average Precision (area under PR curve) is the primary metric for imbalanced classification. "
               "Higher is better. A random classifier scores ~0.0017 (the fraud rate).")

    fig_pr = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        fig_pr.add_trace(go.Scatter(
            x=res["recall"], y=res["precision"],
            mode="lines", name=f'{name} (AP={res["ap"]:.3f})',
            line=dict(color=CHART_COLORS[i], width=2),
        ))

    # Baseline
    fraud_rate = y_test.sum() / len(y_test)
    fig_pr.add_trace(go.Scatter(
        x=[0, 1], y=[fraud_rate, fraud_rate],
        mode="lines", name="Random baseline",
        line=dict(color=COLORS["muted"], width=1, dash="dash"),
    ))

    fig_pr.update_layout(
        **PLOTLY_LAYOUT, height=450,
        xaxis=dict(**AXIS_DEFAULTS, title="Recall", range=[0, 1]),
        yaxis=dict(**AXIS_DEFAULTS, title="Precision", range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    st.plotly_chart(fig_pr, width="stretch")

    # Confusion matrices at best-F1 threshold
    st.markdown('<div class="section-header">Confusion Matrices (at Best-F1 Threshold)</div>', unsafe_allow_html=True)

    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        cm = res["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()

        with col:
            st.caption(f"**{name}**  \nThreshold: {res['best_threshold']:.3f}")

            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Predicted Legit", "Predicted Fraud"],
                y=["Actual Legit", "Actual Fraud"],
                text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                       [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
                texttemplate="%{text}",
                colorscale=[[0, COLORS["sage"]], [1, COLORS["rose"]]],
                showscale=False,
            ))
            cm_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"}
            fig_cm.update_layout(
                **cm_layout, height=250,
                xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_cm, width="stretch")
            st.caption(f"F1: {res['best_f1']:.3f} | Precision: {cm[1,1]/(cm[1,1]+cm[0,1]):.3f} | Recall: {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}")

    # Best F1 comparison bar chart
    st.markdown('<div class="section-header">Best F1 Score Comparison</div>', unsafe_allow_html=True)
    names = list(results.keys())
    f1s = [results[n]["best_f1"] for n in names]

    fig_f1 = go.Figure(go.Bar(
        x=names, y=f1s,
        marker_color=CHART_COLORS[:len(names)],
        text=[f"{f:.3f}" for f in f1s],
        textposition="outside",
    ))
    fig_f1.update_layout(
        **PLOTLY_LAYOUT, height=300,
        yaxis=dict(**AXIS_DEFAULTS, title="F1 Score", range=[0, max(f1s) * 1.15]),
        xaxis=dict(**AXIS_DEFAULTS),
    )
    st.plotly_chart(fig_f1, width="stretch")
