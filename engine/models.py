"""Train and evaluate fraud detection models."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier


def split_and_scale(
    df: pd.DataFrame, feature_cols: list[str], test_size: float = 0.3
):
    """Split into train/test and scale features."""
    X = df[feature_cols].values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler


@st.cache_data(show_spinner="Training models (this may take a minute)...")
def train_all_models(
    _X_train: np.ndarray, _X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
):
    """Train LogReg, Random Forest, and XGBoost. Return results dict."""
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            max_depth=10, random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6,
            scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
            random_state=42, eval_metric="aucpr",
            use_label_encoder=False, n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(_X_train, y_train)
        y_prob = model.predict_proba(_X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        # Find best F1 threshold
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        y_pred_best = (y_prob >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_best)

        results[name] = {
            "model": model,
            "y_prob": y_prob,
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "ap": ap,
            "best_threshold": best_threshold,
            "best_f1": f1_scores[best_idx],
            "confusion_matrix": cm,
        }

    return results
