# Operational Risk Management — Data Mining (S2)

Proyek kelompok (3 orang) untuk mata kuliah Data Mining. Topik: **Operational Risk Management** dengan fokus pada *fraud / loss-event detection* — salah satu kategori utama risiko operasional menurut kerangka Basel.

## Dataset
**PaySim — Synthetic Financial Datasets For Fraud Detection**
Kaggle: [`ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1)

Fitur: `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.

> Alternatif: `mlg-ulb/creditcardfraud`.

## Struktur Proyek
```
operational_risk_management/
├── data/raw/            # dataset mentah (PS_*.csv)
├── data/processed/      # hasil preprocessing (parquet)
├── notebooks/           # eksplorasi per anggota
├── src/
│   ├── data.py          # Anggota 1: load & preprocessing
│   ├── classification.py# Anggota 2: model klasifikasi
│   └── unsupervised.py  # Anggota 3: anomali & klaster
├── models/              # model tersimpan (.joblib)
├── app.py               # dashboard Streamlit (bersama)
├── requirements.txt
└── README.md
```

## Pembagian Tugas
| Anggota | Tanggung jawab | File utama |
|---------|----------------|-----------|
| 1 | Data cleaning, feature engineering, EDA | `src/data.py` |
| 2 | Klasifikasi (LogReg/RF/XGBoost) + handling imbalance | `src/classification.py` |
| 3 | Deteksi anomali (Isolation Forest, LOF) + klastering (K-Means, DBSCAN) | `src/unsupervised.py` |
| Bersama | Dashboard Streamlit + laporan | `app.py` |

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Mengunduh Dataset
1. Buat akun Kaggle → Account → Create New API Token → unduh `kaggle.json`.
2. Letakkan `kaggle.json` di `%USERPROFILE%\.kaggle\kaggle.json`.
3. Jalankan:
   ```bash
   kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip
   ```
   Atau unduh manual dari halaman Kaggle dan ekstrak ke `data/raw/`.

## Menjalankan
```bash
# Preprocessing (Anggota 1)
python -m src.data

# Dashboard (bersama)
streamlit run app.py
```

## Catatan
- Class imbalance ekstrem → utamakan metrik **recall / PR-AUC**, bukan accuracy.
- Dataset besar (~6jt baris); gunakan sampling saat eksplorasi.
