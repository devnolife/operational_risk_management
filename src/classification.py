"""
src/classification.py — Anggota 2: Supervised Learning (Klasifikasi Risiko)

Memprediksi isFraud (transaksi berisiko tinggi). Menangani class imbalance
dan membandingkan beberapa model. Fokus metrik: recall, F1, ROC-AUC, PR-AUC.

Jalankan:
    python -m src.classification
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data import TARGET_COL, get_feature_columns, load_processed

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# Penyiapan data
# ----------------------------------------------------------------------------
def prepare_xy(df: pd.DataFrame | None = None):
    """Pisahkan fitur numerik (X) dan target (y)."""
    if df is None:
        df = load_processed()
    feature_cols = get_feature_columns(df)
    # Hanya kolom numerik/boolean untuk model
    X = df[feature_cols].select_dtypes(include=[np.number, "bool"]).astype(float)
    y = df[TARGET_COL].astype(int)
    return X, y


def split(X, y, test_size: float = 0.25):
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


# ----------------------------------------------------------------------------
# Definisi model
# ----------------------------------------------------------------------------
def build_models() -> dict[str, Pipeline]:
    """Bangun beberapa pipeline model dengan penanganan imbalance."""
    models: dict[str, Pipeline] = {}

    # Logistic Regression — baseline, scaling + class_weight
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    # Random Forest — class_weight balanced
    models["RandomForest"] = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE)),
    ])

    # SMOTE + Logistic Regression — contoh oversampling
    models["SMOTE+LogReg"] = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="aucpr", n_jobs=-1, random_state=RANDOM_STATE)),
        ])

    return models


# ----------------------------------------------------------------------------
# Evaluasi
# ----------------------------------------------------------------------------
def evaluate(model, X_test, y_test) -> dict:
    """Hitung metrik yang relevan untuk data imbalance."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_all(df: pd.DataFrame | None = None, save_best: bool = True):
    """Latih semua model dan kembalikan tabel perbandingan + model terbaik."""
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = split(X, y)

    results = {}
    fitted = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        fitted[name] = model
        print(f"[{name}] recall={results[name]['recall']:.3f} "
              f"f1={results[name]['f1']:.3f} pr_auc={results[name]['pr_auc']:.3f}")

    # Pilih model terbaik berdasarkan PR-AUC (paling relevan untuk imbalance)
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\nModel terbaik (PR-AUC): {best_name}")

    if save_best:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = MODELS_DIR / "best_classifier.joblib"
        joblib.dump({"name": best_name, "model": fitted[best_name],
                     "features": list(X.columns)}, out)
        print(f"Tersimpan: {out}")

    comparison = pd.DataFrame({
        k: {m: v[k] for m, v in results.items()}
        for k in ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    })
    return comparison, fitted, (X_test, y_test)


def get_feature_importance(model, feature_names: list[str]) -> pd.Series | None:
    """Ambil feature importance dari model bila tersedia."""
    clf = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return None
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


if __name__ == "__main__":
    comparison, fitted, _ = train_all()
    print("\n=== Perbandingan Model ===")
    print(comparison.round(4))
