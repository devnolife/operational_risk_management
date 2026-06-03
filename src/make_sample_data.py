"""
src/make_sample_data.py — Pembuat data contoh berformat PaySim.

CATATAN: Ini HANYA untuk menguji pipeline saat dataset Kaggle asli belum
tersedia. Skema kolom dibuat identik dengan PaySim sehingga seluruh kode
(data.py, classification.py, unsupervised.py, app.py) bisa dijalankan.

Ganti file ini dengan data asli:
    kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip

Jalankan:
    python -m src.make_sample_data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = RAW_DIR / "PS_sample_synthetic.csv"

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]


def make_sample(n: int = 50_000, fraud_rate: float = 0.012,
                random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    step = rng.integers(1, 743, size=n)
    ttype = rng.choice(TYPES, size=n, p=[0.34, 0.08, 0.35, 0.20, 0.03])
    amount = np.round(rng.gamma(shape=2.0, scale=4000.0, size=n), 2)

    oldbalanceOrg = np.round(rng.gamma(2.0, 8000.0, size=n), 2)
    newbalanceOrig = np.maximum(oldbalanceOrg - amount, 0).round(2)
    oldbalanceDest = np.round(rng.gamma(2.0, 8000.0, size=n), 2)
    newbalanceDest = (oldbalanceDest + amount).round(2)

    is_fraud = (rng.random(n) < fraud_rate).astype(int)

    # Fraud hanya pada TRANSFER & CASH_OUT (sesuai karakteristik PaySim)
    eligible = np.isin(ttype, ["TRANSFER", "CASH_OUT"])
    is_fraud = (is_fraud & eligible).astype(int)

    # Pola fraud: pengirim dikuras habis, penerima tidak bertambah konsisten
    fraud_idx = is_fraud == 1
    amount[fraud_idx] = oldbalanceOrg[fraud_idx]
    newbalanceOrig[fraud_idx] = 0.0
    newbalanceDest[fraud_idx] = 0.0

    is_flagged = ((amount > 200_000) & (ttype == "TRANSFER")).astype(int)

    name_orig = np.array([f"C{i:09d}" for i in rng.integers(0, 10**9, size=n)])
    name_dest = np.array([f"C{i:09d}" for i in rng.integers(0, 10**9, size=n)])

    return pd.DataFrame({
        "step": step,
        "type": ttype,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": name_dest,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFraud": is_fraud,
        "isFlaggedFraud": is_flagged,
    })


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = make_sample()
    df.to_csv(OUT_PATH, index=False)
    print(f"Tersimpan: {OUT_PATH}  ({len(df):,} baris)")
    print(f"Proporsi fraud: {df['isFraud'].mean():.4%}")
