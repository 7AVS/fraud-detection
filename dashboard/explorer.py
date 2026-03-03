"""Tab 1: Data exploration — class distribution, feature distributions."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dashboard.styles import COLORS, PLOTLY_LAYOUT, AXIS_DEFAULTS, CUSTOM_CSS


def render(df: pd.DataFrame):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # KPI row
    n_total = len(df)
    n_fraud = int(df["Class"].sum())
    n_legit = n_total - n_fraud
    fraud_rate = n_fraud / n_total * 100

    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Total Transactions", "Legitimate", "Fraudulent", "Fraud Rate"],
        [f"{n_total:,}", f"{n_legit:,}", f"{n_fraud:,}", f"{fraud_rate:.3f}%"],
    ):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Class distribution
    st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
    st.caption("The extreme imbalance — 0.17% fraud — makes accuracy a meaningless metric. "
               "A model that predicts 'not fraud' for everything scores 99.83% accuracy.")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Legitimate", "Fraudulent"],
        y=[n_legit, n_fraud],
        marker_color=[COLORS["sage"], COLORS["rose"]],
        text=[f"{n_legit:,}", f"{n_fraud:,}"],
        textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        yaxis=dict(**AXIS_DEFAULTS, title="Count"),
        xaxis=dict(**AXIS_DEFAULTS),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    # Amount distribution
    st.markdown('<div class="section-header">Transaction Amount Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Legitimate transactions")
        legit_amounts = df[df["Class"] == 0]["Amount"]
        fig_legit = go.Figure()
        fig_legit.add_trace(go.Histogram(
            x=legit_amounts.clip(upper=500),
            nbinsx=50,
            marker_color=COLORS["sage"],
            opacity=0.8,
        ))
        fig_legit.update_layout(
            **PLOTLY_LAYOUT, height=280,
            xaxis=dict(**AXIS_DEFAULTS, title="Amount (capped at $500)"),
            yaxis=dict(**AXIS_DEFAULTS, title="Count"),
        )
        st.plotly_chart(fig_legit, width="stretch")

    with col2:
        st.caption("Fraudulent transactions")
        fraud_amounts = df[df["Class"] == 1]["Amount"]
        fig_fraud = go.Figure()
        fig_fraud.add_trace(go.Histogram(
            x=fraud_amounts.clip(upper=500),
            nbinsx=50,
            marker_color=COLORS["rose"],
            opacity=0.8,
        ))
        fig_fraud.update_layout(
            **PLOTLY_LAYOUT, height=280,
            xaxis=dict(**AXIS_DEFAULTS, title="Amount (capped at $500)"),
            yaxis=dict(**AXIS_DEFAULTS, title="Count"),
        )
        st.plotly_chart(fig_fraud, width="stretch")

    # Top distinguishing features
    st.markdown('<div class="section-header">Top Distinguishing Features (V14, V17, V12)</div>', unsafe_allow_html=True)
    st.caption("PCA-transformed features that differ most between fraud and legitimate transactions.")

    for feat in ["V14", "V17", "V12"]:
        fig_feat = go.Figure()
        fig_feat.add_trace(go.Histogram(
            x=df[df["Class"] == 0][feat], name="Legitimate",
            marker_color=COLORS["sage"], opacity=0.6, nbinsx=80,
        ))
        fig_feat.add_trace(go.Histogram(
            x=df[df["Class"] == 1][feat], name="Fraudulent",
            marker_color=COLORS["rose"], opacity=0.6, nbinsx=80,
        ))
        fig_feat.update_layout(
            **PLOTLY_LAYOUT, height=250, barmode="overlay",
            title=dict(text=feat, font=dict(size=14, color=COLORS["dark"])),
            xaxis=dict(**AXIS_DEFAULTS), yaxis=dict(**AXIS_DEFAULTS, title="Count"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(fig_feat, width="stretch")
