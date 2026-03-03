"""Feature engineering for fraud detection."""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features on top of raw PCA components."""
    out = df.copy()

    # Time column only exists in Kaggle version, not OpenML
    if "Time" in out.columns:
        out["Hour"] = (out["Time"] / 3600).astype(int) % 24
        out["IsNight"] = ((out["Hour"] >= 0) & (out["Hour"] < 6)).astype(int)

    # Amount buckets
    out["LogAmount"] = np.log1p(out["Amount"])
    out["AmountBucket"] = pd.cut(
        out["Amount"],
        bins=[-0.01, 5, 25, 100, 500, float("inf")],
        labels=["micro", "small", "medium", "large", "xlarge"],
    ).cat.codes

    # Interaction features from top PCA components
    out["V1_V2"] = out["V1"] * out["V2"]
    out["V1_V3"] = out["V1"] * out["V3"]

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns (everything except target)."""
    return [c for c in df.columns if c != "Class"]
