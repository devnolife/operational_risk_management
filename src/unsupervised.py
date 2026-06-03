"""
src/unsupervised.py — Anggota 3: Unsupervised Learning

Dua bagian:
  1. Deteksi anomali  : Isolation Forest & Local Outlier Factor.
  2. Klastering        : K-Means (elbow/silhouette) & DBSCAN.

Tujuan: menemukan transaksi anomali (potensi fraud/loss event) dan
mensegmentasi profil risiko operasional.

Jalankan:
    python -m src.unsupervised
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.data import TARGET_COL, get_feature_columns, load_processed

RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# Penyiapan matriks fitur (numerik, terstandardisasi)
# ----------------------------------------------------------------------------
def prepare_matrix(df: pd.DataFrame | None = None, sample: int | None = 20_000):
    if df is None:
        df = load_processed()
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=RANDOM_STATE).reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].select_dtypes(include=[np.number, "bool"]).astype(float)
    X_scaled = StandardScaler().fit_transform(X)
    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    return df, X, X_scaled, y


# ----------------------------------------------------------------------------
# 1. Deteksi anomali
# ----------------------------------------------------------------------------
def isolation_forest(X_scaled, contamination: float = 0.01):
    model = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=RANDOM_STATE, n_jobs=-1)
    labels = model.fit_predict(X_scaled)          # -1 anomali, 1 normal
    scores = -model.score_samples(X_scaled)        # makin tinggi makin anomali
    is_anomaly = (labels == -1).astype(int)
    return is_anomaly, scores


def local_outlier_factor(X_scaled, contamination: float = 0.01, n_neighbors: int = 20):
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
    labels = model.fit_predict(X_scaled)
    scores = -model.negative_outlier_factor_
    is_anomaly = (labels == -1).astype(int)
    return is_anomaly, scores


def evaluate_anomaly(is_anomaly, scores, y) -> dict:
    """Bandingkan hasil anomali dengan label fraud (jika ada)."""
    out: dict = {}
    if y is not None:
        out["roc_auc_vs_fraud"] = roc_auc_score(y, scores)
        out["report"] = classification_report(
            y, is_anomaly, output_dict=True, zero_division=0)
    out["n_anomaly"] = int(is_anomaly.sum())
    out["anomaly_rate"] = float(is_anomaly.mean())
    return out


# ----------------------------------------------------------------------------
# 2. Klastering
# ----------------------------------------------------------------------------
def kmeans_search(X_scaled, k_range=range(2, 9)) -> pd.DataFrame:
    """Cari jumlah klaster optimal via inertia (elbow) & silhouette."""
    rows = []
    # Silhouette pada subsample agar cepat
    n = X_scaled.shape[0]
    idx = np.random.default_rng(RANDOM_STATE).choice(
        n, size=min(5000, n), replace=False)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled[idx], labels[idx])
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_kmeans(X_scaled, k: int = 4):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def fit_dbscan(X_scaled, eps: float = 1.5, min_samples: int = 10):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)   # -1 = noise
    return db, labels


def cluster_profile(df: pd.DataFrame, labels, feature_cols: list[str]) -> pd.DataFrame:
    """Ringkas karakteristik tiap klaster untuk interpretasi profil risiko."""
    tmp = df.copy()
    tmp["cluster"] = labels
    agg = {c: "mean" for c in feature_cols if c in tmp.columns}
    if TARGET_COL in tmp.columns:
        agg[TARGET_COL] = "mean"   # proporsi fraud per klaster
    profile = tmp.groupby("cluster").agg(agg)
    profile["count"] = tmp.groupby("cluster").size()
    return profile


if __name__ == "__main__":
    df, X, X_scaled, y = prepare_matrix()

    print("=== Deteksi Anomali ===")
    iso_anom, iso_scores = isolation_forest(X_scaled)
    print("Isolation Forest:", evaluate_anomaly(iso_anom, iso_scores, y))
    lof_anom, lof_scores = local_outlier_factor(X_scaled)
    print("LOF n_anomaly:", int(lof_anom.sum()))

    print("\n=== Pencarian k (K-Means) ===")
    search = kmeans_search(X_scaled)
    print(search.round(3).to_string(index=False))
    best_k = int(search.loc[search["silhouette"].idxmax(), "k"])
    print("k terbaik (silhouette):", best_k)

    km, labels = fit_kmeans(X_scaled, k=best_k)
    profile = cluster_profile(df, labels,
                              ["amount", "oldbalanceOrg", "amountToBalanceRatio"])
    print("\n=== Profil Klaster ===")
    print(profile.round(3))
