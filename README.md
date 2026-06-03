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
├── src/                 # pipeline data mining PaySim (pelengkap)
│   ├── data.py          # Anggota 1: load & preprocessing
│   ├── classification.py# Anggota 2: model klasifikasi
│   └── unsupervised.py  # Anggota 3: anomali & klaster
├── orm/                 # METODOLOGI BUKU (Giudici Bab 12) — inti proyek
│   ├── categories.py    # Basel 8 BL × 7 ET = 56 kategori + skala ordinal
│   ├── data_sources.py  # 3 aliran data: internal / expert / external + scaling
│   ├── scorecard.py     # scorecard: median + Gini → rating A/AA/AAA (Gbr 12.1)
│   ├── actuarial.py     # L=ΣX; Poisson×lognormal MC → VaR 99.9% (Gbr 12.2)
│   └── integrated.py    # integrasi Bayesian + BIA → perbandingan VaR (Gbr 12.2)
├── models/              # model tersimpan (.joblib)
├── app.py               # dashboard Streamlit — 3 halaman ORM (1 per anggota)
├── requirements.txt
└── README.md
```

## Metodologi Buku — Giudici (2009), Bab 12 "Operational Risk Management"

Proyek mengikuti metodologi pengukuran risiko operasional dari buku *Applied Data
Mining for Business and Industry* (Bab 12). Paket [`orm/`](orm/) mengimplementasikan:

| Tahap (buku) | Modul | Inti |
|--------------|-------|------|
| §12.2 Exploratory analysis | `orm/data_sources.py` | Tiga aliran data — **internal loss** (fraud PaySim), **expert opinion** (self-assessment sintetis), **external loss** (konsorsium sintetis) — digabung lewat **scaling**. Kerangka Basel **56 kategori**. |
| §12.4 Scorecard (Gbr 12.1) | `orm/scorecard.py` | **Median kelas** → huruf dasar (A=risiko rendah); **indeks Gini ternormalisasi** → kelipatan huruf (konsensus → AAA). Konvensi lampu lalu lintas (A=hijau, B=kuning, C+=merah). Koreksi kontrol/KRI pada *perceived loss*. |
| §12.3 Actuarial (Gbr 12.2) | `orm/actuarial.py` | $L=\sum_{i=1}^{N} X_i$; frekuensi **Poisson** × severity **lognormal**, dikonvolusi via **Monte Carlo** → **VaR 99.9%** (historis-empiris vs parametrik). |
| §12.4 Integrated Bayesian + BIA (Gbr 12.2) | `orm/integrated.py` | Gabungan internal + 1 titik self-assessment + eksternal → **VaR terintegrasi** (simple & Monte Carlo); **Basic Indicator Approach** = 15% gross income; tabel perbandingan VaR. |

**Pemetaan data → buku.** Transaksi fraud PaySim (`amount`) dipakai sebagai
**internal loss data** untuk kategori *Payment & Settlement / External Fraud*
(sesuai buku yang memetakan satu area; hlm. 230). Opini expert dan kerugian
eksternal bersifat **sintetis** namun setia pada metodologi, karena data expert/
DIPO asli tidak tersedia. Semua VaR memakai basis periode sama (harian) agar
**urutan antar-metode** konsisten; besaran absolut mengikuti karakteristik PaySim.

Menjalankan modul ORM secara langsung:
```bash
python -m orm.data_sources   # ringkasan 3 aliran data
python -m orm.scorecard      # scorecard 56 kategori (median + Gini)
python -m orm.actuarial      # VaR aktuaria (Monte Carlo)
python -m orm.integrated     # perbandingan VaR (analog Gambar 12.2)
```

Dashboard **fokus 3 halaman** di sidebar — **1 anggota = 1 halaman**:
`1 — Scorecard (Self-Assessment)` (Anggota 1), `2 — VaR Aktuaria` (Anggota 2),
`3 — VaR Integrasi & BIA` (Anggota 3). Analisis PaySim (`src/`) tetap tersedia
sebagai pelengkap lewat CLI (`python -m src.data`), tidak menambah halaman dashboard.

## Pembagian Tugas
Buku Giudici Bab 12 memakai **tiga model pengukuran risiko operasional**, jadi
**1 anggota = 1 model/modul**. Modul fondasi (`orm/categories.py`) dan dashboard
dikerjakan bersama.

| Anggota | Model buku (modul utama) | Inti yang dikerjakan | File utama | PaySim pelengkap |
|---------|--------------------------|----------------------|------------|------------------|
| 1 | **Model Scorecard / Self-Assessment** (§12.4, Gbr 12.1) | Median kelas + indeks Gini ternormalisasi → rating A/AA/AAA + lampu lalu lintas untuk 56 kategori | `orm/scorecard.py` | EDA & preprocessing `src/data.py` |
| 2 | **Model Aktuaria** (§12.3, Gbr 12.2) | L=ΣX; frekuensi Poisson × severity lognormal → Monte Carlo → VaR 99.9% + diagnostik ekor | `orm/actuarial.py` | Klasifikasi fraud `src/classification.py` |
| 3 | **Model Terintegrasi Bayesian + BIA** (§12.4, Gbr 12.2) | Gabung 3 aliran data → VaR terintegrasi (simple & MC) + Basic Indicator Approach + tabel perbandingan | `orm/integrated.py` + `orm/data_sources.py` | Anomali & klaster `src/unsupervised.py` |
| Bersama | Fondasi & dashboard | Kerangka Basel 56 kategori + skala ordinal; dashboard Streamlit + laporan | `orm/categories.py`, `app.py` | — |

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
# 1. Preprocessing PaySim → data/processed/paysim_processed.parquet
python -m src.data            # (atau: python -m src.make_sample_data lalu python -m src.data)

# 2. Metodologi buku (Giudici Bab 12)
python -m orm.scorecard       # scorecard self-assessment (Gbr 12.1)
python -m orm.integrated      # perbandingan VaR (Gbr 12.2)

# 3. Dashboard (fokus 3 model buku, 1 halaman per anggota)
streamlit run app.py
```

Dashboard **fokus 3 halaman** — satu per anggota: **Scorecard (Self-Assessment)**,
**VaR Aktuaria**, dan **VaR Integrasi & BIA**. Analisis PaySim (`src/`) dijalankan
terpisah via CLI sebagai pelengkap.

## Catatan
- Modul `orm/` mengikuti metodologi buku; `src/` adalah analisis data mining PaySim pelengkap.
- Class imbalance ekstrem → utamakan metrik **recall / PR-AUC**, bukan accuracy.
- Dataset besar (~6jt baris); gunakan sampling saat eksplorasi.
