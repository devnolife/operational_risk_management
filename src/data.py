"""
src/data.py — Anggota 1: Data Engineering & EDA

Modul fondasi bersama: memuat dataset PaySim, membersihkan, dan melakukan
feature engineering. Fungsi di sini dipakai oleh modul klasifikasi (Anggota 2)
dan unsupervised (Anggota 3) serta dashboard.

Jalankan langsung untuk membuat data/processed/paysim_processed.parquet:
    python -m src.data
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Path konstanta
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "paysim_processed.parquet"

# Kolom asli PaySim
NUMERIC_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
TARGET_COL = "isFraud"


# ----------------------------------------------------------------------------
# Pemuatan data
# ----------------------------------------------------------------------------
def find_raw_csv() -> Path:
    """Cari file CSV PaySim di data/raw (nama bisa bervariasi)."""
    candidates = list(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Tidak ada CSV di {RAW_DIR}. Unduh dataset PaySim dulu:\n"
            "  kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip\n"
            "atau unduh manual dari Kaggle dan ekstrak ke data/raw/."
        )
    # Pilih file terbesar (file utama PaySim ~470MB)
    return max(candidates, key=lambda p: p.stat().st_size)


def load_raw(nrows: int | None = None, sample_frac: float | None = None,
             random_state: int = 42) -> pd.DataFrame:
    """Muat data mentah PaySim.

    Args:
        nrows: batasi jumlah baris yang dibaca (untuk eksplorasi cepat).
        sample_frac: ambil sampel acak proporsi tertentu setelah dibaca.
    """
    csv_path = find_raw_csv()
    df = pd.read_csv(csv_path, nrows=nrows)
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Pembersihan
# ----------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data: hapus duplikat, validasi tipe, tangani missing value."""
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # Kolom numerik: paksa numerik, isi NA dengan 0 (saldo tidak diketahui = 0)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # type sebagai kategori string
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper().str.strip()

    # Target integer 0/1
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Buat fitur turunan yang relevan untuk deteksi fraud / risiko operasional."""
    df = df.copy()

    # Error saldo: selisih yang tidak konsisten sering menandakan transaksi fraud
    df["errorBalanceOrig"] = (
        df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    )
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    # Flag rekening pengirim/penerima yang saldonya nol (sering pada fraud)
    df["origZeroBalance"] = ((df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0)).astype(int)
    df["destZeroBalance"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(int)

    # Rasio jumlah transaksi terhadap saldo awal pengirim
    df["amountToBalanceRatio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)

    # Fitur waktu dari step (1 step = 1 jam, total 30 hari simulasi)
    if "step" in df.columns:
        df["hourOfDay"] = df["step"] % 24
        df["day"] = df["step"] // 24

    # One-hot encoding tipe transaksi
    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], prefix="type")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Kembalikan daftar kolom fitur untuk modeling (buang ID & target)."""
    drop_cols = {
        TARGET_COL,
        "isFlaggedFraud",
        "nameOrig",
        "nameDest",
    }
    return [c for c in df.columns if c not in drop_cols]


# ----------------------------------------------------------------------------
# Pipeline lengkap
# ----------------------------------------------------------------------------
def build_processed(nrows: int | None = None, sample_frac: float | None = None,
                    save: bool = True) -> pd.DataFrame:
    """Pipeline penuh: load -> clean -> feature engineering -> simpan."""
    df = load_raw(nrows=nrows, sample_frac=sample_frac)
    df = clean(df)
    df = engineer_features(df)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_PATH, index=False)
        print(f"Tersimpan: {PROCESSED_PATH}  ({df.shape[0]:,} baris, {df.shape[1]} kolom)")
    return df


def load_processed() -> pd.DataFrame:
    """Muat data processed; jika belum ada, bangun dari raw."""
    if PROCESSED_PATH.exists():
        return pd.read_parquet(PROCESSED_PATH)
    return build_processed()


if __name__ == "__main__":
    out = build_processed()
    print(out.head())
    if TARGET_COL in out.columns:
        rate = out[TARGET_COL].mean()
        print(f"Proporsi fraud: {rate:.4%} ({out[TARGET_COL].sum():,} dari {len(out):,})")
