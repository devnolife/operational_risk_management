"""
app.py — Dashboard Streamlit (kontribusi bersama 3 anggota)

Multipage:
  1. Overview & EDA            (Anggota 1)
  2. Klasifikasi Risiko        (Anggota 2)
  3. Anomali & Klastering      (Anggota 3)

Jalankan:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import (
    PROCESSED_PATH,
    TARGET_COL,
    get_feature_columns,
    load_processed,
)

st.set_page_config(page_title="Operational Risk Management",
                   page_icon="⚠️", layout="wide")


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_processed()


@st.cache_data(show_spinner=False)
def get_sample(n: int = 20_000) -> pd.DataFrame:
    df = get_data()
    if len(df) > n:
        return df.sample(n, random_state=42).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚠️ Operational Risk Management")
st.sidebar.caption("Tugas Data Mining S2 — Kelompok 3 orang")
page = st.sidebar.radio(
    "Halaman",
    ["Overview & EDA", "Klasifikasi Risiko", "Anomali & Klastering"],
)

if not PROCESSED_PATH.exists():
    st.error(
        "Data processed belum ada. Jalankan dulu:\n\n"
        "`python -m src.make_sample_data`  (data contoh)  lalu  `python -m src.data`\n\n"
        "atau unduh PaySim asli dari Kaggle ke `data/raw/`."
    )
    st.stop()


# ============================================================================
# Halaman 1 — Overview & EDA (Anggota 1)
# ============================================================================
if page == "Overview & EDA":
    st.title("Overview & Exploratory Data Analysis")
    df = get_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi", f"{len(df):,}")
    c2.metric("Jumlah Fraud", f"{int(df[TARGET_COL].sum()):,}")
    c3.metric("Proporsi Fraud", f"{df[TARGET_COL].mean():.3%}")
    c4.metric("Jumlah Fitur", f"{df.shape[1]}")

    st.subheader("Cuplikan Data")
    st.dataframe(df.head(20), use_container_width=True)

    type_cols = [c for c in df.columns if c.startswith("type_")]
    if type_cols:
        st.subheader("Fraud per Tipe Transaksi")
        rows = []
        for c in type_cols:
            mask = df[c] == True  # noqa: E712
            rows.append({
                "tipe": c.replace("type_", ""),
                "jumlah": int(mask.sum()),
                "fraud_rate": float(df.loc[mask, TARGET_COL].mean()) if mask.sum() else 0.0,
            })
        tdf = pd.DataFrame(rows)
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(
            px.bar(tdf, x="tipe", y="jumlah", title="Volume per Tipe"),
            use_container_width=True)
        col_b.plotly_chart(
            px.bar(tdf, x="tipe", y="fraud_rate", title="Fraud Rate per Tipe"),
            use_container_width=True)

    st.subheader("Distribusi Nominal (amount)")
    samp = get_sample()
    fig = px.histogram(samp, x="amount", color=TARGET_COL, nbins=60,
                       log_y=True, title="Distribusi amount (skala log)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Korelasi Fitur Numerik")
    num = get_sample().select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, aspect="auto",
                              title="Matriks Korelasi"), use_container_width=True)


# ============================================================================
# Halaman 2 — Klasifikasi Risiko (Anggota 2)
# ============================================================================
elif page == "Klasifikasi Risiko":
    st.title("Klasifikasi Risiko (Prediksi Fraud)")
    st.caption("Membandingkan beberapa model dengan penanganan class imbalance. "
               "Fokus pada **recall** dan **PR-AUC**.")

    from src.classification import (
        get_feature_importance,
        train_all,
    )

    if st.button("Latih & Bandingkan Model", type="primary"):
        with st.spinner("Melatih model..."):
            comparison, fitted, (X_test, y_test) = train_all(save_best=True)
        st.session_state["comparison"] = comparison
        st.session_state["fitted"] = fitted
        st.session_state["xtest_cols"] = list(X_test.columns)

    if "comparison" in st.session_state:
        comparison = st.session_state["comparison"]
        st.subheader("Perbandingan Metrik")
        st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=1),
                     use_container_width=True)

        st.plotly_chart(
            px.bar(comparison.T.reset_index().melt(id_vars="index"),
                   x="index", y="value", color="variable", barmode="group",
                   labels={"index": "Model", "value": "Skor", "variable": "Metrik"},
                   title="Perbandingan Model"),
            use_container_width=True)

        model_name = st.selectbox("Lihat feature importance untuk model",
                                  list(st.session_state["fitted"].keys()))
        imp = get_feature_importance(
            st.session_state["fitted"][model_name],
            st.session_state["xtest_cols"])
        if imp is not None:
            st.plotly_chart(
                px.bar(imp.head(15).sort_values(),
                       orientation="h", title=f"Top 15 Fitur — {model_name}"),
                use_container_width=True)
        else:
            st.info("Model ini tidak menyediakan feature importance.")
    else:
        st.info("Klik tombol di atas untuk melatih model.")


# ============================================================================
# Halaman 3 — Anomali & Klastering (Anggota 3)
# ============================================================================
elif page == "Anomali & Klastering":
    st.title("Deteksi Anomali & Segmentasi Risiko")

    from src.unsupervised import (
        cluster_profile,
        evaluate_anomaly,
        fit_kmeans,
        isolation_forest,
        kmeans_search,
        local_outlier_factor,
        prepare_matrix,
    )

    tab1, tab2 = st.tabs(["Deteksi Anomali", "Klastering"])

    df, X, X_scaled, y = prepare_matrix()

    with tab1:
        method = st.radio("Metode", ["Isolation Forest", "Local Outlier Factor"],
                          horizontal=True)
        contamination = st.slider("Contamination (proporsi anomali)",
                                  0.005, 0.05, 0.01, 0.005)
        if method == "Isolation Forest":
            anom, scores = isolation_forest(X_scaled, contamination)
        else:
            anom, scores = local_outlier_factor(X_scaled, contamination)

        ev = evaluate_anomaly(anom, scores, y)
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Anomali", f"{ev['n_anomaly']:,}")
        c2.metric("Anomaly Rate", f"{ev['anomaly_rate']:.3%}")
        if "roc_auc_vs_fraud" in ev:
            c3.metric("ROC-AUC vs Fraud", f"{ev['roc_auc_vs_fraud']:.3f}")

        plot_df = df.copy()
        plot_df["anomaly_score"] = scores
        plot_df["is_anomaly"] = anom
        st.plotly_chart(
            px.scatter(plot_df.sample(min(5000, len(plot_df)), random_state=42),
                       x="amount", y="amountToBalanceRatio",
                       color="is_anomaly", hover_data=["anomaly_score"],
                       title="Anomali pada Ruang amount vs ratio"),
            use_container_width=True)

    with tab2:
        st.subheader("Pencarian Jumlah Klaster")
        if st.button("Cari k optimal (elbow & silhouette)"):
            search = kmeans_search(X_scaled)
            st.session_state["ksearch"] = search
        if "ksearch" in st.session_state:
            search = st.session_state["ksearch"]
            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(px.line(search, x="k", y="inertia", markers=True,
                                     title="Elbow (Inertia)"),
                             use_container_width=True)
            cc2.plotly_chart(px.line(search, x="k", y="silhouette", markers=True,
                                     title="Silhouette Score"),
                             use_container_width=True)

        k = st.slider("Jumlah klaster (k)", 2, 8, 4)
        km, labels = fit_kmeans(X_scaled, k=k)
        profile = cluster_profile(
            df, labels,
            ["amount", "oldbalanceOrg", "newbalanceOrig", "amountToBalanceRatio"])
        st.subheader("Profil Klaster (interpretasi risiko)")
        st.dataframe(profile.style.format("{:.3f}"), use_container_width=True)

        plot_df = df.copy()
        plot_df["cluster"] = labels.astype(str)
        st.plotly_chart(
            px.scatter(plot_df.sample(min(5000, len(plot_df)), random_state=42),
                       x="amount", y="oldbalanceOrg", color="cluster",
                       title="Visualisasi Klaster"),
            use_container_width=True)
