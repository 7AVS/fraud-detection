"""Load the Credit Card Fraud dataset from OpenML."""

import os
import pandas as pd
import streamlit as st


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "creditcard.parquet")


@st.cache_data(show_spinner="Loading credit card fraud dataset...")
def load_data() -> pd.DataFrame:
    """Load the creditcard fraud dataset, caching locally as parquet."""
    if os.path.exists(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)

    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("creditcardfraud", version=1, as_frame=True, parser="auto")
    df = bunch.frame

    # Ensure target is integer
    df["Class"] = df["Class"].astype(int)

    # Ensure numeric types
    for col in df.columns:
        if col != "Class":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df
