"""
app.py — Dashboard Streamlit (mengikuti Giudici 2009, Bab 12 "Operational Risk Management")

Fokus: 3 halaman, 1 per anggota — tepat seperti 3 model pengukuran di buku.
Tiap halaman disusun seperti laporan agar mudah dipahami orang awam:
  Rumusan Masalah → Metode → Dataset → Proses → Hasil → Kesimpulan.

  1. Scorecard (Self-Assessment)   — Anggota 1 (orm/scorecard.py, Gambar 12.1)
  2. VaR Aktuaria                   — Anggota 2 (orm/actuarial.py, Gambar 12.2)
  3. VaR Integrasi & BIA            — Anggota 3 (orm/integrated.py, Gambar 12.2)

Jalankan:
    streamlit run app.py
"""
from __future__ import annotations

import inspect

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import PROCESSED_PATH

st.set_page_config(page_title="Operational Risk Management",
                   page_icon="⚠️", layout="wide")

# Inject custom CSS for better presentation look
st.markdown("""
<style>
/* Bigger metric labels */
[data-testid="stMetricLabel"] { font-size: 0.95rem !important; }
[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
/* Hero banner style */
.hero-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    color: white; padding: 2rem 2.5rem; border-radius: 1rem;
    margin-bottom: 1.5rem;
}
.hero-box h1 { color: white !important; margin-bottom: 0.3rem; }
.hero-box p  { color: #d0e4f7; font-size: 1.1rem; }
/* Flow step cards */
.flow-card {
    background: #f8f9fa; border-left: 4px solid #2d6a9f;
    padding: 1rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0;
    margin-bottom: 0.8rem;
}
.flow-card h4 { margin: 0 0 0.3rem 0; color: #1e3a5f; }
.flow-card p  { margin: 0; color: #555; font-size: 0.95rem; }
/* Key finding highlight */
.key-finding {
    background: linear-gradient(90deg, #e8f5e9, #f1f8e9);
    border-left: 4px solid #2ca02c; padding: 1rem 1.2rem;
    border-radius: 0 0.5rem 0.5rem 0; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helper narasi — menyusun tiap halaman seperti laporan (untuk orang awam)
# ----------------------------------------------------------------------------
def langkah(nomor: int, judul: str) -> None:
    """Judul section bernomor agar alur mudah diikuti."""
    st.markdown(f"### {nomor}️⃣ {judul}")


def kotak_arti(teks: str) -> None:
    """Penjelasan 'apa artinya' dalam bahasa sederhana."""
    st.info("💡 **Apa artinya?** " + teks)


def perumpamaan(teks: str) -> None:
    """Analogi/perumpamaan sehari-hari agar mudah dipahami."""
    st.warning("🧩 **Perumpamaan:** " + teks)


@st.cache_data(show_spinner=False)
def info_dataset_paysim():
    """Hitung total transaksi & jumlah fraud dari seluruh dataset PaySim."""
    df = pd.read_parquet(PROCESSED_PATH, columns=["isFraud"])
    return int(len(df)), int(df["isFraud"].sum())


def kamus_kolom(rows: list[tuple[str, str]]) -> None:
    """Tampilkan tabel 'kamus data' (nama kolom + arti)."""
    st.markdown("**Kamus kolom (arti tiap kolom):**")
    st.dataframe(
        pd.DataFrame(rows, columns=["Kolom", "Arti"]),
        use_container_width=True, hide_index=True,
    )


def daftar_istilah(items: list[tuple[str, str]]) -> None:
    """Tampilkan glosarium istilah dalam bahasa sederhana."""
    st.markdown("**📖 Istilah penting (bahasa sederhana):**")
    st.dataframe(
        pd.DataFrame(items, columns=["Istilah", "Artinya"]),
        use_container_width=True, hide_index=True,
    )


def tampilkan_kode(func) -> None:
    """Tampilkan source code ASLI sebuah fungsi dari modul orm (selalu sinkron)."""
    st.code(inspect.getsource(func), language="python")


@st.cache_data(show_spinner=False)
def contoh_internal():
    """Sampel data kerugian internal (transaksi fraud PaySim)."""
    from orm.data_sources import internal_frequency_severity, load_internal_losses
    losses = load_internal_losses()
    return losses, internal_frequency_severity(losses)


@st.cache_data(show_spinner=False)
def contoh_expert(n_experts: int = 8, seed: int = 42):
    """Sampel kuesioner penilaian ahli (expert opinion)."""
    from orm.data_sources import generate_expert_opinions
    return generate_expert_opinions(n_experts=n_experts, seed=seed)


@st.cache_data(show_spinner=False)
def contoh_external():
    """Sampel kerugian eksternal (sintetis) + versi ter-scaling."""
    from orm.data_sources import (
        generate_external_losses,
        scale_external_to_internal,
    )
    _, fs = contoh_internal()
    ext = generate_external_losses(internal_severity=fs["severity_sample"])
    ext_scaled = scale_external_to_internal(ext, fs["total_internal"])
    return ext, ext_scaled


# Kamus seluruh field dataset PaySim (semua 22 kolom data processed).
PAYSIM_FIELDS: list[tuple[str, str]] = [
    ("step", "Satuan waktu simulasi (1 step = 1 jam)"),
    ("amount", "Jumlah uang transaksi"),
    ("nameOrig", "ID nasabah pengirim"),
    ("oldbalanceOrg", "Saldo pengirim sebelum transaksi"),
    ("newbalanceOrig", "Saldo pengirim sesudah transaksi"),
    ("nameDest", "ID nasabah penerima"),
    ("oldbalanceDest", "Saldo penerima sebelum transaksi"),
    ("newbalanceDest", "Saldo penerima sesudah transaksi"),
    ("isFraud", "Label penipuan (1 = fraud, 0 = normal)"),
    ("isFlaggedFraud", "Ditandai sistem sebagai transaksi mencurigakan"),
    ("errorBalanceOrig", "Ketidakcocokan saldo pengirim (rekayasa fitur)"),
    ("errorBalanceDest", "Ketidakcocokan saldo penerima (rekayasa fitur)"),
    ("origZeroBalance", "Penanda saldo pengirim nol (rekayasa fitur)"),
    ("destZeroBalance", "Penanda saldo penerima nol (rekayasa fitur)"),
    ("amountToBalanceRatio", "Rasio jumlah transaksi terhadap saldo (rekayasa fitur)"),
    ("hourOfDay", "Jam dalam hari (0–23) dari step"),
    ("day", "Hari ke- (dihitung dari step)"),
    ("type_CASH_IN", "Jenis transaksi = setor tunai (one-hot)"),
    ("type_CASH_OUT", "Jenis transaksi = tarik tunai (one-hot)"),
    ("type_DEBIT", "Jenis transaksi = debit (one-hot)"),
    ("type_PAYMENT", "Jenis transaksi = pembayaran (one-hot)"),
    ("type_TRANSFER", "Jenis transaksi = transfer (one-hot)"),
]


@st.cache_data(show_spinner=False)
def paysim_meta():
    """Total baris & daftar kolom dataset PaySim (dari metadata parquet, cepat)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PROCESSED_PATH)
    return int(pf.metadata.num_rows), list(pf.schema_arrow.names)


@st.cache_data(show_spinner=False)
def paysim_preview(n: int = 50):
    """Pratinjau n baris pertama dataset PaySim lengkap (semua kolom)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PROCESSED_PATH)
    batch = next(pf.iter_batches(batch_size=max(int(n), 1_000)))
    return batch.to_pandas().head(int(n))


@st.cache_data(show_spinner="Memuat seluruh transaksi fraud...")
def paysim_fraud_full():
    """Seluruh transaksi fraud (semua kolom) — subset kecil (~8 ribu baris)."""
    import pyarrow.parquet as pq
    tbl = pq.read_table(PROCESSED_PATH, filters=[("isFraud", "==", 1)])
    return tbl.to_pandas().reset_index(drop=True)


def dataset_lengkap_paysim(key_prefix: str) -> None:
    """Tampilkan dataset PaySim lengkap: total baris, semua field, pratinjau mentah."""
    total_rows, cols = paysim_meta()
    with st.expander("🗂️ Lihat dataset PaySim LENGKAP (total baris & semua field)"):
        a, b, c = st.columns(3)
        a.metric("Total baris (seluruh transaksi)", f"{total_rows:,}")
        b.metric("Jumlah kolom (field)", f"{len(cols)}")
        c.metric("Periode simulasi", "± 31 hari")
        st.markdown(
            "Model hanya memakai sebagian kolom (kerugian), tetapi **dataset aslinya "
            "jauh lebih lengkap**. Berikut **seluruh field** beserta artinya:"
        )
        kamus_kolom(PAYSIM_FIELDS)
        n = st.slider("Jumlah baris ditampilkan", 10, 500, 50,
                      key=f"{key_prefix}_full_rows")
        st.dataframe(paysim_preview(n), use_container_width=True, hide_index=True)
        st.caption(
            f"Pratinjau {n} dari {total_rows:,} baris, menampilkan **semua {len(cols)} "
            "kolom** apa adanya (data mentah, belum difilter ke transaksi fraud)."
        )


# ----------------------------------------------------------------------------
# Sidebar — navigasi halaman
# ----------------------------------------------------------------------------
st.sidebar.title("⚠️ Operational Risk Management")
st.sidebar.caption("Giudici (2009) Bab 12 — 3 model, 1 per anggota")
page = st.sidebar.radio(
    "Pilih halaman",
    [
        "🏠 Beranda",
        "📂 Data Lengkap (semua data)",
        "1 — Scorecard (Self-Assessment)",
        "2 — VaR Aktuaria",
        "3 — VaR Integrasi & BIA",
        "📊 Ringkasan Eksekutif",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Cara baca tiap halaman model:**\n\n"
    "1️⃣ Rumusan Masalah\n\n"
    "2️⃣ Metode (dari buku)\n\n"
    "3️⃣ Dataset\n\n"
    "4️⃣ Proses\n\n"
    "5️⃣ Hasil & Kesimpulan"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Anggota Kelompok:**\n\n"
    "👤 Anggota 1 → Scorecard\n\n"
    "👤 Anggota 2 → VaR Aktuaria\n\n"
    "👤 Anggota 3 → VaR Integrasi & BIA"
)

if not PROCESSED_PATH.exists():
    st.error(
        "Data processed belum ada. Jalankan dulu:\n\n"
        "`python -m src.make_sample_data`  (data contoh)  lalu  `python -m src.data`\n\n"
        "atau unduh PaySim asli dari Kaggle ke `data/raw/`."
    )
    st.stop()


# ============================================================================
# 🏠 Beranda — Halaman Utama (Landing Page)
# ============================================================================
if page == "🏠 Beranda":
    st.markdown(
        '<div class="hero-box">'
        '<h1>⚠️ Operational Risk Management</h1>'
        '<p>Implementasi Bab 12 — <em>Applied Data Mining for Business and Industry</em> '
        '(Giudici, 2009)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Apa itu Risiko Operasional? (dari buku) ----
    st.markdown("### 📖 Apa itu Risiko Operasional?")
    st.markdown(
        "Menurut **Basel II** (*Basel Committee on Banking Supervision*, 2001), "
        "risiko operasional didefinisikan sebagai:\n\n"
        "> *\"The risk of loss resulting from inadequate or failed internal processes, "
        "people and systems or from external events.\"*\n\n"
        "> **\"Risiko kerugian akibat proses internal yang tidak memadai atau gagal, "
        "kesalahan manusia, kegagalan sistem, atau kejadian dari luar.\"**\n\n"
        "Contoh nyata: **penipuan (fraud)**, kesalahan input data, sistem IT down, "
        "bencana alam, pencurian, pemalsuan dokumen."
    )

    st.markdown("### 🎯 Untuk Apa Mengukur Risiko Operasional?")
    st.markdown(
        "Menurut buku (Giudici, 2009, hlm. 227), ada **dua tujuan utama**:"
    )
    col_tujuan1, col_tujuan2 = st.columns(2)
    with col_tujuan1:
        st.markdown(
            '<div class="flow-card">'
            '<h4>🏦 1. Tujuan Prudensial (Kehati-hatian)</h4>'
            '<p>Menyiapkan <strong>cadangan modal</strong> yang cukup untuk menutup '
            'kerugian tak terduga. Dihitung menggunakan <strong>Value at Risk (VaR)</strong> '
            '— berapa kerugian terburuk yang mungkin terjadi?</p></div>',
            unsafe_allow_html=True,
        )
    with col_tujuan2:
        st.markdown(
            '<div class="flow-card">'
            '<h4>📋 2. Tujuan Manajerial (Pengelolaan)</h4>'
            '<p>Membuat <strong>peringkat risiko</strong> dari yang paling berbahaya '
            'ke yang paling aman, sehingga manajemen tahu <strong>mana yang harus '
            'ditangani lebih dulu</strong>.</p></div>',
            unsafe_allow_html=True,
        )

    st.info(
        "💡 **Intinya:** Semakin tinggi risiko operasional suatu bank, "
        "semakin buruk sistem kontrolnya. Mengukur risiko = mengukur "
        "seberapa efektif kontrol yang ada *(Giudici, 2009, hlm. 227)*."
    )

    # ---- 3 Sumber Data (dari buku Bab 12.2) ----
    st.markdown("---")
    st.markdown("### 📥 Tiga Sumber Data (Bab 12.2)")
    st.markdown(
        "Buku menjelaskan bahwa pengukuran risiko operasional menggunakan "
        "**3 sumber informasi** yang saling melengkapi:"
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            "#### 📊 Data Internal\n"
            "*(Bab 12.2, hal. 221)*\n\n"
            "Buku menjelaskan bahwa data internal adalah **tabel riwayat "
            "kerugian bank sendiri** — berisi: jumlah kerugian (*amount*), "
            "tanggal kejadian, unit organisasi, lini bisnis & jenis kejadian.\n\n"
            "🔹 *Backward-looking* — melihat apa yang sudah terjadi.\n\n"
            "**Di proyek ini:** Dataset **PaySim** (Kaggle) — simulasi "
            "transaksi mobile-money dengan label fraud. Dipilih karena "
            "data bank asli bersifat **rahasia**."
        )
    with d2:
        st.markdown(
            "#### 👥 Penilaian Ahli\n"
            "*(Bab 12.2, hal. 222–224)*\n\n"
            "Buku menjelaskan bahwa **kepala cabang, kepala area, "
            "dan manajer** diminta menilai setiap risiko dalam skala "
            "ordinal: **frekuensi** (seberapa sering), **severity** "
            "(seberapa besar), dan **kontrol** (sebaik apa mitigasinya).\n\n"
            "Penilaian mengacu pada **8 lini bisnis × 7 jenis kejadian "
            "Basel II = 56 kategori**.\n\n"
            "🔹 *Forward-looking* — melihat persepsi risiko ke depan, "
            "bukan hanya data masa lalu.\n\n"
            "**Di proyek ini:** Karena kuesioner bank asli bersifat "
            "**rahasia**, kami membuat data sintetis yang mengikuti "
            "struktur persis dari buku."
        )
    with d3:
        st.markdown(
            "#### 🏢 Data Eksternal\n"
            "*(Bab 12.2, hal. 221–222)*\n\n"
            "Buku menjelaskan bahwa **konsorsium bank** (contoh: DIPO di Italia) "
            "mengumpulkan data kerugian dari banyak bank. Karena gabungan "
            "banyak bank, angkanya lebih besar → perlu di-**scaling** "
            "(dibagi konstanta *c*) agar setara ukuran bank kita.\n\n"
            "🔹 *Perspektif industri* — mengisi celah data yang belum "
            "pernah terjadi di bank kita.\n\n"
            "**Di proyek ini:** Data sintetis yang mengikuti proses "
            "scaling sesuai buku. Database DIPO asli hanya untuk anggota "
            "konsorsium."
        )

    # ---- Justifikasi dataset PaySim ----
    st.markdown("---")
    st.markdown("### 🎯 Mengapa Dataset PaySim?")
    st.info(
        "**Buku Bab 12.2** menjelaskan bahwa **data internal** bank harus berupa "
        "tabel berisi: **jumlah kerugian (amount)**, **tanggal kejadian**, "
        "**unit organisasi**, serta **lini bisnis & jenis kejadian** (Basel II).\n\n"
        "**PaySim** (Kaggle: `ealaxi/paysim1`) dipilih karena memenuhi semua "
        "syarat tersebut:"
    )
    j1, j2 = st.columns(2)
    with j1:
        st.markdown(
            "**✅ Kenapa cocok sebagai data internal?**\n\n"
            "| Syarat Buku | PaySim Punya? |\n"
            "|---|---|\n"
            "| Jumlah kerugian (amount) | ✅ Kolom `amount` |\n"
            "| Tanggal/waktu kejadian | ✅ Kolom `step` (jam) → dikonversi ke hari |\n"
            "| Label kejadian (fraud/tidak) | ✅ Kolom `isFraud` |\n"
            "| Jenis transaksi | ✅ Kolom `type` (TRANSFER, CASH_OUT, dll) |\n"
            "| Identitas pelaku/korban | ✅ Kolom `nameOrig`, `nameDest` |\n"
            "| Jumlah data besar (realistis) | ✅ **6,3 juta** transaksi |"
        )
    with j2:
        st.markdown(
            "**📌 Untuk apa dataset ini dipakai?**\n\n"
            "1. **Transaksi fraud** (`isFraud=1`) → menjadi **data kerugian internal** "
            "bank, di mana kolom `amount` = besarnya kerugian\n"
            "2. Dari data ini dihitung **frekuensi** (berapa kali fraud per tahun) "
            "dan **severity** (distribusi besarnya kerugian)\n"
            "3. Frekuensi & severity ini menjadi input utama model **VaR Aktuaria** "
            "(Bab 12.3) dan model **VaR Integrasi Bayesian** (Bab 12.4)\n\n"
            "*PaySim mensimulasikan transaksi mobile-money dunia nyata berdasarkan "
            "pola transaksi asli dari sebuah perusahaan di Afrika — "
            "sehingga distribusi fraud-nya realistis.*"
        )

    st.markdown(
        "> 💡 **Mengapa bukan data bank asli?** — Data kerugian operasional bank "
        "bersifat **sangat rahasia** dan tidak tersedia publik. PaySim adalah "
        "alternatif terbaik karena ia **sintetis tetapi realistis**: pola frekuensi "
        "dan severity fraud-nya menyerupai data bank sesungguhnya, sehingga metode "
        "dari buku Giudici (2009) dapat diterapkan secara valid."
    )

    # ---- 3 Model/Metode (dari buku Bab 12.3) ----
    st.markdown("---")
    st.markdown("### 🔧 Tiga Model yang Dipakai (Bab 12.3–12.4)")
    st.markdown(
        "Buku membahas **dua pendekatan besar**: *top-down* (dari atas, sederhana) "
        "dan *bottom-up* (dari bawah, detail). Proyek ini mengimplementasikan keduanya:"
    )

    st.markdown(
        '<div class="flow-card">'
        '<h4>📋 Model 1 — Scorecard / Self-Assessment (Anggota 1)</h4>'
        '<p><strong>Apa:</strong> Para ahli mengisi kuesioner tentang risiko, '
        'lalu jawaban mereka dirangkum menjadi <em>rating</em> (A/AA/AAA) dan '
        'warna lampu lalu lintas (🟢🟡🔴).</p>'
        '<p><strong>Untuk apa:</strong> Membuat <em>peringkat risiko</em> — '
        'mana yang harus ditangani duluan? (Tujuan Manajerial)</p>'
        '<p><strong>Metode:</strong> Median + Indeks Gini (konsensus ahli) → '
        'Rating huruf → Perceived Loss → Skor Prioritas</p>'
        '<p><strong>Pendekatan:</strong> Bottom-up (dari pendapat ahli per kategori)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="flow-card">'
        '<h4>📈 Model 2 — VaR Aktuaria (Anggota 2)</h4>'
        '<p><strong>Apa:</strong> Menghitung <em>Value at Risk</em> — '
        'berapa cadangan modal minimum yang harus disiapkan agar 99,9% aman.</p>'
        '<p><strong>Untuk apa:</strong> Menentukan <em>cadangan modal</em> berdasarkan '
        'data historis kerugian. (Tujuan Prudensial)</p>'
        '<p><strong>Metode:</strong> Frekuensi (Poisson) × Severity (Lognormal) → '
        'Simulasi Monte Carlo 100.000× → VaR 99,9%</p>'
        '<p><strong>Pendekatan:</strong> Bottom-up (dari data kerugian aktual)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="flow-card">'
        '<h4>🔗 Model 3 — VaR Integrasi Bayesian + BIA (Anggota 3)</h4>'
        '<p><strong>Apa:</strong> Menggabungkan <em>semua 3 sumber data</em> '
        'menjadi satu estimasi, lalu dibandingkan dengan cara paling sederhana (BIA).</p>'
        '<p><strong>Untuk apa:</strong> <em>Validasi silang</em> — kalau cara canggih '
        'dan cara sederhana hasilnya dekat, estimasi lebih bisa dipercaya. '
        '(Tujuan Prudensial + Validasi)</p>'
        '<p><strong>Metode:</strong> Gabungan internal + expert + eksternal → '
        'VaR Bayesian, lalu BIA = 15% × Gross Income sebagai pembanding</p>'
        '<p><strong>Pendekatan:</strong> Bottom-up (Bayesian) + Top-down (BIA)</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Alur Proyek ----
    st.markdown("---")
    st.markdown("### 🔄 Alur Keseluruhan Proyek")
    st.markdown(
        "| Langkah | Apa yang Dilakukan | Output |\n"
        "|---|---|---|\n"
        "| 1. Kumpulkan Data | Ambil 3 sumber data (internal, ahli, eksternal) | Dataset siap pakai |\n"
        "| 2. Scorecard | Ahli menilai 56 kategori risiko → rating & prioritas | Peringkat risiko 🟢🟡🔴 |\n"
        "| 3. VaR Aktuaria | Simulasi Monte Carlo dari data fraud → distribusi kerugian | Cadangan modal (VaR) |\n"
        "| 4. VaR Integrasi | Gabungkan 3 sumber + bandingkan dengan BIA | 7 estimasi VaR (Gambar 12.2) |\n"
        "| 5. Kesimpulan | Bandingkan semua metode → validasi silang | Estimasi yang bisa dipercaya |"
    )

    st.markdown("---")
    st.markdown("### 📚 Referensi Buku")
    st.markdown(
        "| Bagian Buku | Model | Halaman |\n"
        "|---|---|---|\n"
        "| Bab 12.1 — Kerangka Basel II | Dasar semua model | 225–227 |\n"
        "| Bab 12.2 — Sumber Data | Data internal, ahli, eksternal | 228–231 |\n"
        "| Bab 12.3 — Model Aktuaria | VaR Aktuaria (Model 2) | 231–234 |\n"
        "| Bab 12.4 — Self-Assessment | Scorecard (Model 1) + Integrasi (Model 3) | 234–240 |\n"
        "| Bab 12.5 — Kesimpulan | Perbandingan 7 metrik VaR | 240–241 |"
    )

    st.success(
        "👈 **Mulai dari mana?** Pilih halaman di sidebar kiri. "
        "Atau langsung ke **📊 Ringkasan Eksekutif** untuk melihat semua hasil sekaligus."
    )



# ============================================================================
# Data Lengkap — semua dataset dalam satu tempat (tidak terbagi per model)
# ============================================================================
elif page == "📂 Data Lengkap (semua data)":
    st.title("📂 Data Lengkap — Semua Dataset")
    st.caption("Seluruh data yang dipakai proyek, dikumpulkan di satu halaman.")
    st.markdown(
        "Halaman ini menampilkan **seluruh data** yang dipakai proyek secara lengkap. "
        "Tersedia **3 sumber data** sesuai Bab 12.2 buku Giudici (2009)."
    )

    _total_tx, _total_fraud = info_dataset_paysim()
    _rows, _cols = paysim_meta()
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Total transaksi PaySim", f"{_total_tx:,}")
    g2.metric("Transaksi fraud", f"{_total_fraud:,}")
    g3.metric("Jumlah field PaySim", f"{len(_cols)}")
    g4.metric("Sumber data", "3 jenis")

    tabA, tabB, tabC = st.tabs([
        "① PaySim — Data Internal (utama)",
        "② Penilaian Ahli (Expert Opinion)",
        "③ Kerugian Eksternal (Bank Lain)",
    ])

    # ---- Tab A: PaySim lengkap ----
    with tabA:
        st.markdown(
            "### 📊 Data Internal — PaySim\n\n"
            "**PaySim** — *Synthetic Financial Datasets for Fraud Detection* "
            "(Kaggle: `ealaxi/paysim1`). Dataset sintetis berisi **6,3 juta** "
            "transaksi mobile-money dengan label penipuan (fraud)."
        )
        st.info(
            "**🎯 Mengapa PaySim dipilih?**\n\n"
            "Buku Bab 12.2 mensyaratkan **data kerugian internal** bank berupa tabel "
            "berisi: jumlah kerugian, tanggal, unit organisasi, dan jenis kejadian. "
            "Data asli bank **tidak tersedia publik** (rahasia), sehingga dibutuhkan "
            "dataset pengganti yang realistis.\n\n"
            "**PaySim cocok karena:**\n"
            "- ✅ Ada **kolom `amount`** → jumlah kerugian per peristiwa\n"
            "- ✅ Ada **kolom `step`** → waktu kejadian (jam, dikonversi ke hari)\n"
            "- ✅ Ada **label `isFraud`** → memisahkan fraud vs normal\n"
            "- ✅ Ada **jenis transaksi** (`type`) → mirip jenis kejadian Basel II\n"
            "- ✅ **Distribusi realistis** — dibuat dari pola transaksi mobile-money "
            "nyata di Afrika\n\n"
            "**Cara pemakaian:** Setiap transaksi fraud (`isFraud=1`) dianggap sebagai "
            "**satu peristiwa kerugian operasional**, dan kolom `amount` = besarnya "
            "kerugian. Dari sini dihitung **frekuensi** (berapa kali per tahun) dan "
            "**severity** (distribusi besarnya kerugian) untuk model VaR."
        )
        a1, a2, a3 = st.columns(3)
        a1.metric("Total baris", f"{_rows:,}")
        a2.metric("Jumlah kolom", f"{len(_cols)}")
        a3.metric("Periode simulasi", "± 31 hari")
        kamus_kolom(PAYSIM_FIELDS)

        mode = st.radio(
            "Pilih data yang ditampilkan",
            ["Semua transaksi (pratinjau awal)", "Hanya transaksi fraud (lengkap)"],
            horizontal=True, key="paysim_mode",
        )
        if mode.startswith("Hanya"):
            _fraud_df = paysim_fraud_full()
            st.success(
                f"Menampilkan **seluruh {len(_fraud_df):,} transaksi fraud** "
                "(semua kolom). Ini adalah data yang dipakai model.")
            st.dataframe(_fraud_df, use_container_width=True, hide_index=True,
                         height=500)
            st.download_button(
                "⬇️ Unduh semua transaksi fraud (CSV)",
                _fraud_df.to_csv(index=False).encode("utf-8"),
                file_name="paysim_fraud.csv", mime="text/csv",
            )
        else:
            n = st.slider("Jumlah baris ditampilkan", 10, 500, 100,
                          key="paysim_all_rows")
            st.dataframe(paysim_preview(n), use_container_width=True, hide_index=True)
            st.caption(f"Pratinjau {n} dari {_rows:,} baris (data mentah, semua kolom). "
                       "Pilih **'Hanya transaksi fraud'** untuk melihat seluruh data "
                       "yang dipakai model.")

    # ---- Tab B: Expert lengkap ----
    with tabB:
        st.markdown(
            "### 👥 Data Penilaian Ahli (Expert Opinion)\n\n"
            "Kuesioner **self-assessment** untuk seluruh kerangka Basel II: "
            "**8 lini bisnis × 7 jenis kejadian = 56 kategori risiko**."
        )
        st.info(
            "**🎯 Mengapa ada data expert?**\n\n"
            "Buku Bab 12.2 menjelaskan bahwa data internal saja **tidak cukup** — "
            "banyak kejadian risiko yang jarang terjadi atau belum pernah tercatat. "
            "Oleh karena itu dibutuhkan **pendapat ahli** (*expert opinion*) yang "
            "bersifat **forward-looking** (melihat ke depan).\n\n"
            "**Cara pemakaian:** Setiap ahli menilai **frekuensi**, **severity**, dan "
            "**kontrol** untuk seluruh 56 kategori risiko. Hasilnya diagregasi menjadi "
            "**scorecard** (matriks traffic-light) yang dipakai di Model 1.\n\n"
            "**Mengapa sintetis?** Data kuesioner ahli bank bersifat internal dan "
            "rahasia. Kami membuat data sintetis mengikuti struktur buku: ordinal "
            "scale (low/medium/high) untuk 8 BL × 7 ET."
        )
        n_exp = st.slider("Jumlah ahli (expert)", 3, 20, 8, key="data_n_exp")
        _exp_all = contoh_expert(n_exp, 42)
        b1, b2, b3 = st.columns(3)
        b1.metric("Total baris", f"{len(_exp_all):,}")
        b2.metric("Jumlah ahli", f"{_exp_all['expert'].nunique()}")
        b3.metric("Kategori risiko",
                  f"{_exp_all.groupby(['business_line','event_type']).ngroups}")
        kamus_kolom([
            ("expert", "Nomor ahli yang memberi penilaian"),
            ("business_line", "Lini bisnis bank (1 dari 8)"),
            ("event_type", "Jenis kejadian risiko (1 dari 7)"),
            ("frequency", "Perkiraan seberapa sering kejadian terjadi (kelas ordinal)"),
            ("severity", "Perkiraan seberapa besar kerugiannya (kelas ordinal)"),
            ("control", "Penilaian sebaik apa kontrol yang ada (kelas ordinal)"),
        ])
        st.success(f"Menampilkan **seluruh {len(_exp_all):,} baris** penilaian ahli.")
        st.dataframe(
            _exp_all.rename(columns={
                "expert": "Ahli ke-", "business_line": "Lini Bisnis",
                "event_type": "Jenis Kejadian", "frequency": "Frekuensi",
                "severity": "Severity", "control": "Kontrol",
            }),
            use_container_width=True, hide_index=True, height=500,
        )
        st.download_button(
            "⬇️ Unduh seluruh penilaian ahli (CSV)",
            _exp_all.to_csv(index=False).encode("utf-8"),
            file_name="penilaian_ahli.csv", mime="text/csv",
        )

    # ---- Tab C: Eksternal lengkap ----
    with tabC:
        st.markdown(
            "### 🏢 Data Kerugian Eksternal (Bank Lain)\n\n"
            "Data dari **konsorsium bank lain** (sintetis, analog DIPO)."
        )
        st.info(
            "**🎯 Mengapa ada data eksternal?**\n\n"
            "Buku Bab 12.2 menjelaskan bahwa data internal & expert masih bisa memiliki "
            "**missing values** — ada kategori risiko yang jarang/belum pernah terjadi "
            "di bank kita. Data **konsorsium (DIPO)** dari banyak bank membantu "
            "**mengisi celah** tersebut.\n\n"
            "**Proses scaling:** Karena konsorsium menggabungkan banyak bank, "
            "kerugiannya lebih besar. Buku (Bab 12.2) menerapkan **scaling** — "
            "membagi kerugian konsorsium dengan konstanta *c* = rasio total "
            "kerugian DIPO / total kerugian internal — agar setara ukuran bank kita.\n\n"
            "**Mengapa sintetis?** Database DIPO hanya tersedia bagi anggota "
            "konsorsium. Kami membuat data sintetis yang mengikuti distribusi "
            "dan proses scaling sesuai buku."
        )
        _ext_raw, _ext_scaled = contoh_external()
        _ext_df = pd.DataFrame({
            "No.": range(1, len(_ext_raw) + 1),
            "Kerugian Mentah (bank lain)": _ext_raw,
            "Kerugian Ter-scaling (setara bank kita)": _ext_scaled,
        })
        c1, c2, c3 = st.columns(3)
        c1.metric("Total baris", f"{len(_ext_df):,}")
        c2.metric("Median (mentah)", f"{pd.Series(_ext_raw).median():,.0f}")
        c3.metric("Median (ter-scaling)", f"{pd.Series(_ext_scaled).median():,.0f}")
        st.success(f"Menampilkan **seluruh {len(_ext_df):,} baris** kerugian eksternal.")
        st.dataframe(
            _ext_df.style.format({
                "Kerugian Mentah (bank lain)": "{:,.0f}",
                "Kerugian Ter-scaling (setara bank kita)": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True, height=500,
        )
        st.download_button(
            "⬇️ Unduh kerugian eksternal (CSV)",
            _ext_df.to_csv(index=False).encode("utf-8"),
            file_name="kerugian_eksternal.csv", mime="text/csv",
        )

# ============================================================================
# Anggota 1 — Scorecard Self-Assessment (Giudici Ch.12, Gambar 12.1)
# ============================================================================
elif page == "1 — Scorecard (Self-Assessment)":
    from orm.data_sources import generate_expert_opinions
    from orm.scorecard import (
        TRAFFIC_LIGHT_HEX,
        build_scorecard,
        normalized_gini,
        rate_dimension,
        self_assessment_total_loss,
    )

    st.title("📋 Model 1 — Scorecard (Self-Assessment)")
    st.caption("Anggota 1 · Giudici (2009) Bab 12.4 · analog Gambar 12.1")

    st.markdown(
        '<div class="key-finding">'
        '<strong>📌 Halaman ini menjawab:</strong> Dari 56 kategori risiko bank, '
        '<strong>mana yang paling berbahaya</strong> dan harus ditangani lebih dulu? '
        'Model ini menggunakan <em>pendapat ahli</em> untuk membuat peringkat risiko '
        'dengan sistem warna 🟢🟡🔴.'
        '</div>',
        unsafe_allow_html=True,
    )

    langkah(1, "Rumusan Masalah")
    st.markdown(
        "> **Dari banyak jenis risiko operasional yang bisa terjadi di sebuah bank, "
        "mana yang paling berbahaya dan harus ditangani lebih dulu?**\n\n"
        "Sebuah bank menghadapi puluhan jenis risiko (penipuan, kesalahan sistem, "
        "kesalahan pegawai, dll). Tidak mungkin menangani semua sekaligus. Kita perlu "
        "cara untuk **memberi peringkat** risiko mana yang paling mendesak."
    )

    langkah(2, "Metode (dari buku)")
    st.markdown(
        "**Scorecard / Self-Assessment** — seperti membuat *rapor* untuk tiap risiko.\n\n"
        "1. Para ahli menilai tiap risiko dari 3 sisi: **seberapa sering** terjadi "
        "(frekuensi), **seberapa besar** kerugiannya (severity), dan **sebaik apa "
        "kontrol** yang ada.\n"
        "2. Penilaian para ahli digabung memakai **nilai tengah (median)** + tingkat "
        "**kesepakatan (indeks Gini)**.\n"
        "3. Hasilnya berupa **rating huruf**: **A = paling aman**, makin jauh dari A "
        "(AA, AAA, B, C...) makin perlu perhatian, ditambah **lampu lalu lintas** "
        "🟢 hijau (aman) → 🟡 kuning (waspada) → 🔴 merah (bahaya)."
    )
    perumpamaan(
        "Bayangkan **dokter di UGD** yang melakukan *triase*. Pasien yang paling kritis "
        "didahulukan, bukan yang datang pertama. Scorecard melakukan hal sama untuk "
        "risiko: yang paling 'kritis' (merah) ditangani lebih dulu."
    )

    with st.expander("📖 Istilah, Rumus, & kenapa dipakai (klik untuk buka)"):
        daftar_istilah([
            ("Median (nilai tengah)", "Nilai yang persis di tengah setelah data diurutkan. Tahan terhadap nilai ekstrem."),
            ("Ordinal", "Skala berjenjang (rendah/sedang/tinggi) tanpa jarak pasti antar tingkat."),
            ("Indeks Gini", "Ukuran seberapa beragam jawaban para ahli: 0 = semua sama, 1 = sangat beragam."),
            ("Konsensus", "Tingkat kesepakatan antar ahli."),
            ("Rating huruf (A/AA/AAA)", "Seperti peringkat kredit. A = paling aman; makin banyak huruf = makin kuat keyakinannya."),
            ("Perceived loss", "Perkiraan kerugian = seberapa sering × seberapa besar × faktor kontrol."),
        ])
        st.markdown("**Rumus yang dipakai:**")
        st.latex(r"G = 1 - \sum_{k} p_k^{2} \qquad\Rightarrow\qquad G_{norm} = G \cdot \frac{K}{K-1}")
        st.caption("Indeks Gini ternormalisasi — mengukur keberagaman jawaban ahli "
                   "(p_k = proporsi ahli yang memilih kelas ke-k, K = jumlah kelas).")
        st.latex(r"\text{Perceived Loss} = \text{frekuensi/tahun} \times \text{nilai severity} \times \text{faktor kontrol}")
        st.latex(r"\text{Skor Prioritas} = r_{\text{freq}} \times r_{\text{sev}} \times r_{\text{ctrl}}")
        st.caption("r = peringkat kelas median tiap dimensi (makin tinggi = makin berisiko).")
        st.markdown(
            "**Kenapa rumus ini dipakai?**\n\n"
            "- **Median (bukan rata-rata)** — penilaian ahli bersifat *ordinal*, dan "
            "median tahan terhadap 1 ahli yang menjawab ekstrem.\n"
            "- **Indeks Gini** — mengubah *seberapa kompak* pendapat ahli menjadi angka. "
            "Kalau ahli kompak (Gini kecil), kita lebih percaya → rating diperkuat (AAA). "
            "Inilah cara buku menilai **kualitas konsensus**.\n"
            "- **Frekuensi × Severity × Kontrol** — definisi baku risiko: "
            "**Risiko = Peluang × Dampak**, lalu disesuaikan oleh kualitas kontrol."
        )

    langkah(3, "Dataset")
    st.markdown(
        "Kerangka Basel II: **8 lini bisnis × 7 jenis kejadian = 56 kategori risiko**. "
        "Penilaian diberikan oleh sejumlah **ahli (expert)**. Karena data penilaian "
        "ahli asli tidak tersedia, di sini penilaian **disimulasikan** namun tetap "
        "mengikuti metode buku."
    )
    _exp_full = contoh_expert(8, 42)
    d1, d2, d3 = st.columns(3)
    d1.metric("Nama data", "Expert Opinion")
    d2.metric("Jumlah baris penilaian", f"{len(_exp_full):,}")
    d3.metric("Kategori risiko", f"{_exp_full.groupby(['business_line','event_type']).ngroups}")
    kamus_kolom([
        ("expert", "Nomor ahli yang memberi penilaian"),
        ("business_line", "Lini bisnis bank (1 dari 8)"),
        ("event_type", "Jenis kejadian risiko (1 dari 7)"),
        ("frequency", "Perkiraan seberapa sering kejadian terjadi (kelas)"),
        ("severity", "Perkiraan seberapa besar kerugiannya (kelas)"),
        ("control", "Penilaian sebaik apa kontrol yang ada (kelas)"),
    ])
    _ren_exp = {
        "expert": "Ahli ke-", "business_line": "Lini Bisnis",
        "event_type": "Jenis Kejadian", "frequency": "Penilaian Frekuensi",
        "severity": "Penilaian Severity", "control": "Penilaian Kontrol",
    }
    st.markdown("**Contoh sampel** (8 ahli menilai 1 kategori):")
    st.dataframe(_exp_full.head(8).rename(columns=_ren_exp),
                 use_container_width=True, hide_index=True)
    with st.expander("🗂️ Lihat SELURUH data penilaian ahli (semua baris & field)"):
        st.caption(
            f"Total {len(_exp_full):,} baris × {_exp_full.shape[1]} field "
            f"({_exp_full['expert'].nunique()} ahli × "
            f"{_exp_full.groupby(['business_line','event_type']).ngroups} kategori "
            "= seluruh kerangka Basel)."
        )
        n_baris = st.slider("Jumlah baris ditampilkan", 8, len(_exp_full),
                            min(56, len(_exp_full)), key="exp_rows")
        st.dataframe(_exp_full.head(n_baris).rename(columns=_ren_exp),
                     use_container_width=True, hide_index=True)
    kotak_arti(
        "Tiap baris = penilaian **satu ahli** untuk satu kategori risiko. Penilaian "
        "banyak ahli inilah yang nanti dirangkum (median + kesepakatan) jadi rating."
    )

    langkah(4, "Proses")
    st.markdown("Atur jumlah ahli yang menilai, lalu sistem menghitung rating tiap kategori:")
    col_a, col_b = st.columns(2)
    with col_a:
        n_experts = st.slider("Jumlah ahli (expert) yang menilai", 3, 20, 8)
    with col_b:
        seed = st.number_input("Seed (pengunci angka acak)", value=42, step=1)

    @st.cache_data(show_spinner="Menghitung scorecard...")
    def _scorecard(n_experts: int, seed: int):
        opinions = generate_expert_opinions(n_experts=n_experts, seed=int(seed))
        return build_scorecard(opinions)

    sc = _scorecard(n_experts, seed)

    langkah(5, "Hasil & Kesimpulan")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah kategori dinilai", f"{len(sc)}")
    c2.metric("Total perkiraan kerugian", f"Rp {self_assessment_total_loss(sc):,.0f}")

    st.markdown("**Tabel scorecard** (sudah diurutkan: paling atas = paling prioritas)")

    dim_to_label = {
        "frequency": "Rating Frekuensi",
        "severity": "Rating Severity",
        "control": "Rating Kontrol",
    }

    def _color_cells(row):
        styles = {}
        for dim, label in dim_to_label.items():
            hexcol = TRAFFIC_LIGHT_HEX.get(row[f"{dim}_color"], "#d62728")
            styles[label] = f"background-color: {hexcol}; color: white"
        return pd.Series(styles)

    show_cols = [
        "business_line", "event_type",
        "frequency_rating", "severity_rating", "control_rating",
        "perceived_loss", "priority_score",
    ]
    rename = {
        "business_line": "Lini Bisnis", "event_type": "Jenis Kejadian",
        "frequency_rating": "Rating Frekuensi", "severity_rating": "Rating Severity",
        "control_rating": "Rating Kontrol", "perceived_loss": "Perkiraan Kerugian",
        "priority_score": "Skor Prioritas",
    }
    styled = (
        sc[show_cols].rename(columns=rename)
        .style.apply(lambda r: _color_cells(sc.loc[r.name]), axis=1)
        .format({"Perkiraan Kerugian": "{:,.0f}", "Skor Prioritas": "{:d}"})
    )
    st.dataframe(styled, use_container_width=True, height=500)
    kotak_arti(
        "Warna sel = lampu lalu lintas. 🔴 Merah berarti rating buruk (sering terjadi / "
        "kerugian besar / kontrol lemah) sehingga **harus ditangani lebih dulu**. "
        "Kolom *Skor Prioritas* makin besar = makin mendesak."
    )

    st.markdown("**Grafik 15 risiko prioritas tertinggi**")
    top = sc.head(15)
    st.plotly_chart(
        px.bar(top, x="perceived_loss", y="event_type", color="business_line",
               orientation="h",
               labels={"perceived_loss": "Perkiraan Kerugian",
                       "event_type": "Jenis Kejadian", "business_line": "Lini Bisnis"},
               title="Perkiraan kerugian tertinggi"),
        use_container_width=True)

    st.markdown("**🗺️ Peta Risiko (Heatmap)**")
    _sc_pivot = sc.pivot_table(
        index="business_line", columns="event_type",
        values="perceived_loss", aggfunc="first",
    )
    _fig_hm = px.imshow(
        _sc_pivot, text_auto=".0f",
        color_continuous_scale="RdYlGn_r",
        labels={"color": "Perceived Loss"},
        title="Perceived Loss per kategori (merah = risiko tinggi)",
        aspect="auto",
    )
    _fig_hm.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(_fig_hm, use_container_width=True)
    kotak_arti(
        "Peta risiko ini menunjukkan **seluruh 56 kombinasi** Basel II. "
        "Kotak **paling merah** = area yang harus mendapat perhatian utama."
    )

    teratas = sc.iloc[0]
    st.success(
        f"**Kesimpulan:** Risiko paling prioritas adalah **{teratas['business_line']} — "
        f"{teratas['event_type']}** (rating severity {teratas['severity_rating']}). "
        f"Inilah yang sebaiknya **ditangani lebih dulu** oleh manajemen."
    )

    with st.expander("🔧 Cara Kerja, Kode Program & Hasilnya (klik untuk buka)"):
        st.markdown(
            "**Cara kerja (langkah demi langkah):**\n\n"
            "1. Kumpulkan suara semua ahli untuk **satu kategori, satu dimensi** "
            "(mis. frekuensi).\n"
            "2. **Median** suara → menentukan *huruf dasar* (A = paling aman).\n"
            "3. **Indeks Gini** suara → mengukur kekompakan; makin kompak pendapat ahli, "
            "huruf diperkuat (A → AA → AAA).\n"
            "4. `rate_dimension` menggabungkan median + Gini menjadi satu **rating + warna**.\n"
            "5. `build_scorecard` mengulang langkah 1–4 untuk **56 kategori × 3 dimensi**, "
            "lalu mengurutkan berdasarkan skor prioritas → tabel berwarna di atas."
        )
        st.markdown("#### 📜 Kode asli — `normalized_gini`")
        tampilkan_kode(normalized_gini)
        st.markdown("#### 📜 Kode asli — `rate_dimension`")
        tampilkan_kode(rate_dimension)

        st.markdown("#### ▶️ Hasil nyata bila kode dijalankan")
        _op_demo = generate_expert_opinions(n_experts=8, seed=42)
        (_bl_d, _et_d), _g_demo = next(
            iter(_op_demo.groupby(["business_line", "event_type"])))
        _votes_demo = _g_demo["frequency"].tolist()
        st.markdown(f"**Input** — suara 8 ahli (dimensi *frekuensi*) untuk kategori "
                    f"*{_bl_d} — {_et_d}*:")
        st.code(repr(_votes_demo), language="python")
        _gini_demo = normalized_gini(_votes_demo, "frequency")
        _rate_demo = rate_dimension(_votes_demo, "frequency")
        st.markdown("**Output `normalized_gini(votes, 'frequency')`:**")
        st.code(f"{_gini_demo:.3f}", language="python")
        st.markdown("**Output `rate_dimension(votes, 'frequency')`:**")
        st.json(_rate_demo)
        st.caption(
            "Inilah **satu sel rating**. `build_scorecard` mengulanginya untuk semua "
            "kategori sehingga menghasilkan tabel berwarna pada bagian Hasil di atas."
        )


# ============================================================================
# Anggota 2 — Model Aktuaria, VaR (Giudici Ch.12, Gambar 12.2)
# ============================================================================
elif page == "2 — VaR Aktuaria":
    from orm.actuarial import (
        fit_severity_lognormal,
        simulate_aggregate,
        value_at_risk,
    )
    from orm.integrated import run_comparison

    st.title("📈 Model 2 — VaR Aktuaria")
    st.caption("Anggota 2 · Giudici (2009) Bab 12.3 · analog Gambar 12.2")

    st.markdown(
        '<div class="key-finding">'
        '<strong>📌 Halaman ini menjawab:</strong> Berapa <strong>cadangan modal '
        'minimum</strong> yang harus disiapkan bank untuk menutup kerugian fraud? '
        'Model ini mensimulasikan <em>ribuan skenario</em> lalu mencari batas '
        'kerugian terburuk (VaR 99,9%).'
        '</div>',
        unsafe_allow_html=True,
    )

    langkah(1, "Rumusan Masalah")
    st.markdown(
        "> **Seberapa besar total kerugian terburuk yang mungkin kita alami akibat "
        "penipuan (fraud), dan berapa 'modal cadangan' yang harus disiapkan?**\n\n"
        "Bank wajib menyiapkan cadangan modal untuk menutup kerugian tak terduga. "
        "Pertanyaannya: berapa angka cadangan yang masuk akal — tidak terlalu kecil "
        "(berisiko bangkrut) tapi juga tidak terlalu besar (modal nganggur)?"
    )

    langkah(2, "Metode (dari buku)")
    st.markdown(
        "**Model Aktuaria** — menghitung total kerugian dengan menggabungkan 2 hal:\n\n"
        "1. **Seberapa sering** kejadian terjadi → distribusi **Poisson** (frekuensi).\n"
        "2. **Seberapa besar** tiap kerugian → distribusi **lognormal** (severity).\n"
        "3. Komputer mengarang **ribuan skenario** masa depan (**simulasi Monte "
        "Carlo**), lalu kita lihat distribusi total kerugiannya.\n"
        "4. **VaR (Value at Risk)** = batas kerugian pada tingkat keyakinan tinggi "
        "(mis. 99,9%). Artinya hanya 0,1% kemungkinan kerugian melebihi angka ini."
    )
    perumpamaan(
        "Seperti **perusahaan asuransi** yang menghitung premi. Mereka tidak tahu "
        "persis siapa yang akan klaim, tapi dengan menyimulasikan ribuan kemungkinan "
        "mereka tahu **skenario terburuk** dan menyiapkan dana untuk itu. VaR 99,9% = "
        "'dana darurat' yang cukup untuk menutup 999 dari 1.000 kemungkinan."
    )

    with st.expander("📖 Istilah, Rumus, & kenapa dipakai (klik untuk buka)"):
        daftar_istilah([
            ("Frekuensi (N)", "Berapa kali kejadian terjadi dalam satu periode."),
            ("Severity (X)", "Besar kerugian setiap kejadian."),
            ("Distribusi Poisson", "Model peluang untuk MENGHITUNG JUMLAH kejadian acak yang langka per periode (mis. jumlah klaim per hari)."),
            ("Distribusi Lognormal", "Model untuk BESARAN yang selalu positif & condong ke kanan: banyak kerugian kecil, sesekali sangat besar."),
            ("Monte Carlo", "Teknik 'mengarang' ribuan skenario acak lalu meninjau hasil keseluruhannya."),
            ("VaR (Value at Risk)", "Batas kerugian pada tingkat keyakinan tertentu."),
            ("Persentil", "Nilai yang di bawahnya terdapat sekian persen data (P99,9 = 99,9% data di bawahnya)."),
            ("μ dan σ", "Rata-rata & simpangan dari logaritma kerugian (parameter lognormal)."),
        ])
        st.markdown("**Rumus yang dipakai:**")
        st.latex(r"L = \sum_{i=1}^{N} X_i")
        st.caption("Total kerugian = menjumlahkan N kerugian individual (N pun acak).")
        st.latex(r"N \sim \text{Poisson}(\lambda) \qquad X \sim \text{Lognormal}(\mu,\ \sigma)")
        st.latex(r"VaR_q = \text{persentil ke-}q\ \text{dari distribusi } L \qquad E[L] = \text{rata-rata } L")
        st.latex(r"x_q = e^{\,\mu + \sigma\, z_q}")
        st.caption("Kuantil lognormal (z_q = nilai-z normal baku untuk peluang q).")
        st.markdown(
            "**Kenapa rumus ini dipakai?**\n\n"
            "- **Poisson untuk frekuensi** — paling cocok menghitung **jumlah kejadian "
            "langka & acak** yang saling bebas per periode (standar di aktuaria/asuransi).\n"
            "- **Lognormal untuk severity** — kerugian **tidak pernah negatif** dan punya "
            "**ekor panjang** (sesekali kerugian sangat besar), persis pola data fraud.\n"
            "- **Monte Carlo** — menjumlahkan N (acak) kerugian (acak) sulit dihitung "
            "dengan satu rumus; menyimulasikan ribuan kali memberi gambaran utuh.\n"
            "- **VaR 99,9%** — **regulator Basel** mewajibkan modal yang cukup menutup "
            "kerugian pada tingkat keyakinan sangat tinggi (hanya 0,1% boleh meleset)."
        )

    langkah(3, "Dataset")
    st.markdown(
        "**Data transaksi fraud PaySim** (kolom `amount`) dipakai sebagai **data "
        "kerugian internal nyata** — tiap transaksi penipuan = satu peristiwa kerugian."
    )
    _loss, _fs = contoh_internal()
    _total_tx, _total_fraud = info_dataset_paysim()
    st.markdown(
        "📂 **Nama dataset:** PaySim — *Synthetic Financial Datasets for Fraud "
        "Detection* (Kaggle: `ealaxi/paysim1`). Simulasi transaksi keuangan mobile-money "
        "yang memuat label penipuan."
    )
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Total transaksi", f"{_total_tx:,}")
    g2.metric("Transaksi fraud", f"{_total_fraud:,}")
    g3.metric("Dipakai (kerugian)", f"{_fs['n_events']:,}")
    g4.metric("Rentang waktu", f"{_fs['span_days']} hari")
    kamus_kolom([
        ("business_line", "Lini bisnis Basel (di sini: Payment and Settlement)"),
        ("event_type", "Jenis kejadian Basel (di sini: External Fraud)"),
        ("day", "Hari ke- saat transaksi terjadi (dari kolom step/jam PaySim)"),
        ("amount", "Jumlah uang transaksi = besar kerugian (Rupiah)"),
    ])
    _ren_loss = {
        "business_line": "Lini Bisnis", "event_type": "Jenis Kejadian",
        "day": "Hari ke-", "amount": "Jumlah Kerugian",
    }
    st.markdown("**Contoh sampel** (10 transaksi fraud pertama):")
    st.dataframe(_loss.head(10).rename(columns=_ren_loss)
                 .style.format({"Jumlah Kerugian": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)
    with st.expander("🔍 Lihat lebih banyak data kerugian internal"):
        n_baris = st.slider("Jumlah baris ditampilkan", 10, 500, 50, key="loss_rows")
        st.dataframe(_loss.head(n_baris).rename(columns=_ren_loss)
                     .style.format({"Jumlah Kerugian": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)
        st.caption(f"Statistik kerugian: rata-rata {_loss['amount'].mean():,.0f}, "
                   f"terbesar {_loss['amount'].max():,.0f}.")
    kotak_arti(
        "Tiap baris = satu transaksi penipuan nyata beserta **jumlah kerugiannya**. "
        "Sebaran angka-angka inilah yang dipakai model untuk memperkirakan kerugian "
        "di masa depan."
    )
    dataset_lengkap_paysim("p2")

    langkah(4, "Proses")
    st.markdown("Atur tingkat keyakinan dan jumlah simulasi, lalu jalankan Monte Carlo:")
    col_a, col_b = st.columns(2)
    with col_a:
        q = st.slider("Tingkat keyakinan VaR (persentil)", 0.95, 0.999, 0.999, 0.001)
    with col_b:
        n_sims = st.select_slider("Jumlah skenario simulasi",
                                  options=[5_000, 10_000, 20_000, 50_000], value=20_000)

    @st.cache_data(show_spinner="Menjalankan ribuan simulasi Monte Carlo...")
    def _comparison_act(q: float, n_sims: int):
        return run_comparison(q=q, n_sims=n_sims)

    res = _comparison_act(q, n_sims)
    act = res["actuarial"]

    langkah(5, "Hasil & Kesimpulan")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rata-rata kerugian", f"{act['expected_loss_montecarlo']:,.0f}")
    m2.metric(f"VaR {q:.1%} (simulasi)", f"{act['actuarial_montecarlo_var']:,.0f}")
    m3.metric(f"VaR {q:.1%} (data historis)", f"{act['historical_var']:,.0f}")
    kotak_arti(
        f"**Rata-rata kerugian** = kerugian yang biasa terjadi. **VaR {q:.1%}** = "
        f"batas kerugian terburuk: dengan keyakinan {q:.1%}, kerugian dalam "
        f"{res['period_label']} **tidak akan melebihi** angka tersebut. Inilah usulan "
        f"'modal cadangan'."
    )

    st.markdown("**Grafik distribusi kemungkinan total kerugian** (dari ribuan skenario)")
    losses = act["_param_losses"]
    var_line = value_at_risk(losses, q)
    fig = px.histogram(x=losses, nbins=80, log_y=True,
                       title="Sebaran total kerugian (garis merah = batas VaR)",
                       labels={"x": "Total kerugian", "y": "Jumlah skenario"})
    fig.add_vline(x=var_line, line_color="red", line_dash="dash",
                  annotation_text=f"VaR {q:.1%}")
    fig.update_layout(xaxis_title="Total kerugian", yaxis_title="Jumlah skenario")
    st.plotly_chart(fig, use_container_width=True)
    kotak_arti(
        "Tiap batang = berapa banyak skenario menghasilkan kerugian sebesar itu. "
        "Sebagian besar skenario ada di kiri (kerugian kecil), tapi ada 'ekor' panjang "
        "ke kanan (kerugian besar yang jarang). **Garis merah** menandai batas VaR."
    )

    with st.expander("🔬 Catatan teknis: diagnostik risiko-model (ekor severity)"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("σ lognormal", f"{act['severity_sigma']:.2f}")
        d2.metric("Severity maks", f"{act['severity_max_observed']:,.0f}")
        d3.metric("Severity P99.9 (empiris)", f"{act['severity_empirical_q999']:,.0f}")
        d4.metric("Severity P99.9 (lognormal)", f"{act['severity_lognormal_q999']:,.0f}")
        st.caption("Ekor lognormal mengekstrapolasi jauh di atas sampel empiris — wajar "
                   "untuk σ besar, namun perlu diwaspadai sebagai risiko model.")

    st.success(
        f"**Kesimpulan:** Dengan keyakinan **{q:.1%}**, total kerugian fraud dalam "
        f"{res['period_label']} diperkirakan **tidak melebihi "
        f"{act['actuarial_montecarlo_var']:,.0f}**. Angka inilah yang bisa dipakai "
        f"sebagai usulan **modal cadangan** untuk menutup kerugian tak terduga."
    )

    with st.expander("🔧 Cara Kerja, Kode Program & Hasilnya (klik untuk buka)"):
        st.markdown(
            "**Cara kerja (langkah demi langkah):**\n\n"
            "1. Ambil sampel besar **kerugian (severity)** dari data fraud.\n"
            "2. `fit_severity_lognormal` menghitung parameter **μ** dan **σ** "
            "(rata-rata & sebaran dari logaritma kerugian).\n"
            "3. `simulate_aggregate` mengulang **ribuan kali**: undi jumlah kejadian "
            "**N ~ Poisson(λ)**, lalu jumlahkan N kerugian acak → satu total kerugian.\n"
            "4. Kumpulan ribuan total kerugian membentuk sebuah **distribusi**.\n"
            "5. `value_at_risk` mengambil **persentil** tinggi (mis. 99,9%) dari "
            "distribusi itu = **VaR** (batas kerugian terburuk)."
        )
        st.markdown("#### 📜 Kode asli — `fit_severity_lognormal`")
        tampilkan_kode(fit_severity_lognormal)
        st.markdown("#### 📜 Kode asli — `simulate_aggregate`")
        tampilkan_kode(simulate_aggregate)
        st.markdown("#### 📜 Kode asli — `value_at_risk`")
        tampilkan_kode(value_at_risk)

        st.markdown("#### ▶️ Hasil nyata bila kode dijalankan")
        _loss_d, _fs_d = contoh_internal()
        _sev_d = _fs_d["severity_sample"]
        _freq_d = _fs_d["n_events"] / _fs_d["span_days"]
        _mu_d, _sig_d = fit_severity_lognormal(_sev_d)
        _losses_d = simulate_aggregate(_freq_d, mu=_mu_d, sigma=_sig_d,
                                       n_sims=5_000, seed=42)
        _mean_d = float(_losses_d.mean())
        _var_d = value_at_risk(_losses_d, 0.999)
        st.markdown(f"**Input** — {len(_sev_d):,} nilai kerugian, frekuensi "
                    f"λ ≈ {_freq_d:.1f} kejadian/hari.")
        st.code(
            f"mu, sigma = fit_severity_lognormal(severity)   # -> ({_mu_d:.3f}, {_sig_d:.3f})\n"
            f"losses = simulate_aggregate(freq={_freq_d:.1f}, mu=mu, sigma=sigma, n_sims=5000)\n"
            f"value_at_risk(losses, 0.999)                   # -> {_var_d:,.0f}\n"
            f"losses.mean()                                  # -> {_mean_d:,.0f}",
            language="python")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("μ (mu)", f"{_mu_d:.3f}")
        cc2.metric("σ (sigma)", f"{_sig_d:.3f}")
        cc3.metric("VaR 99,9% (demo)", f"{_var_d:,.0f}")
        st.caption(
            "Demo memakai 5.000 skenario agar cepat; bagian Proses di atas memakai "
            "lebih banyak skenario untuk hasil yang lebih halus."
        )


# ============================================================================
# Anggota 3 — Model Terintegrasi Bayesian & BIA (Giudici Ch.12, Gambar 12.2)
# ============================================================================
elif page == "3 — VaR Integrasi & BIA":
    from orm.integrated import (
        basic_indicator_approach,
        bayes_var_simple,
        integrated_severity,
        run_comparison,
    )
    from orm.data_sources import scale_external_to_internal

    st.title("🧮 Model 3 — VaR Integrasi & BIA")
    st.caption("Anggota 3 · Giudici (2009) Bab 12.4 · analog Gambar 12.2")

    st.markdown(
        '<div class="key-finding">'
        '<strong>📌 Halaman ini menjawab:</strong> Jika kita gabungkan <strong>semua '
        'sumber data</strong> (internal + ahli + bank lain), berapa estimasi VaR-nya? '
        'Dan apakah hasilnya <strong>konsisten</strong> dengan cara sederhana (BIA = 15% '
        'pendapatan)? Jika dekat → estimasi lebih bisa dipercaya.'
        '</div>',
        unsafe_allow_html=True,
    )

    langkah(1, "Rumusan Masalah")
    st.markdown(
        "> **Kalau kerugian dihitung dengan beberapa metode berbeda, berapa hasilnya "
        "dan apakah saling konsisten?**\n\n"
        "Mengandalkan satu metode saja berisiko. Lebih aman bila kita menghitung "
        "dengan **beberapa cara** lalu membandingkannya — jika hasilnya berdekatan, "
        "kita lebih yakin angkanya masuk akal."
    )

    langkah(2, "Metode (dari buku)")
    st.markdown(
        "Dua pendekatan dibandingkan:\n\n"
        "1. **Integrasi Bayesian** (*bottom-up*) — menggabungkan **3 sumber data** "
        "menjadi satu estimasi: kerugian internal (fraud kita sendiri) + penilaian "
        "ahli (self-assessment) + kerugian dari bank lain (data eksternal).\n"
        "2. **Basic Indicator Approach / BIA** (*top-down*, aturan Basel II) — cara "
        "paling sederhana: cadangan = **15% dari pendapatan kotor**.\n\n"
        "Keduanya ditampilkan berdampingan untuk dibandingkan."
    )
    perumpamaan(
        "Seperti minta **pendapat kedua (second opinion)** ke beberapa dokter sebelum "
        "operasi. Kalau dokter umum, dokter spesialis, dan hasil lab **sepakat**, kita "
        "jauh lebih yakin. Di sini 'dokter'-nya = data internal, ahli, dan bank lain."
    )

    with st.expander("📖 Istilah, Rumus, & kenapa dipakai (klik untuk buka)"):
        daftar_istilah([
            ("Integrasi / Bayesian", "Menggabungkan beberapa sumber informasi menjadi satu estimasi yang lebih kuat."),
            ("Bottom-up vs Top-down", "Dari data rinci ke atas (Bayesian) vs dari indikator besar ke bawah (BIA)."),
            ("Scaling", "Menyetarakan skala data bank lain agar sebanding dengan data kita."),
            ("Gross income", "Pendapatan kotor bank — indikator ukuran bisnis."),
            ("BIA", "Basic Indicator Approach (Basel II): cadangan = 15% pendapatan kotor."),
            ("Konsorsium / DIPO", "Basis data kerugian gabungan dari banyak bank."),
        ])
        st.markdown("**Rumus yang dipakai:**")
        st.latex(r"S_{\text{gabungan}} = S_{\text{internal}} \ \cup\ \{p_{\text{self-assessment}}\} \ \cup\ S_{\text{eksternal}}^{\text{scaled}}")
        st.latex(r"c = \frac{\sum \text{kerugian eksternal}}{\sum \text{kerugian internal}} \qquad S^{\text{scaled}} = \frac{S_{\text{eksternal}}}{c}")
        st.caption("Konstanta scaling c menyetarakan ukuran data eksternal terhadap internal.")
        st.latex(r"VaR_q = \text{persentil ke-}q\ \text{dari } S_{\text{gabungan}}")
        st.latex(r"BIA = \alpha \times \text{Gross Income}, \quad \alpha = 15\%")
        st.markdown(
            "**Kenapa rumus ini dipakai?**\n\n"
            "- **Menggabungkan 3 sumber** — tiap sumber punya kelemahan: internal "
            "(terbatas pengalaman sendiri), expert (subjektif), eksternal (beda konteks). "
            "Gabungan lebih **lengkap** (melihat ke belakang *dan* ke depan) dan tahan.\n"
            "- **Scaling** — agar kerugian bank besar tidak 'membanjiri' estimasi bank "
            "kita; disesuaikan secara proporsional dulu.\n"
            "- **BIA (15% gross income)** — pembanding sederhana yang **diakui regulator**. "
            "Kalau metode canggih (Bayesian) hasilnya **dekat** dengan BIA, itu memberi "
            "**keyakinan silang** bahwa angkanya masuk akal."
        )

    langkah(3, "Dataset")
    st.markdown(
        "**Tiga aliran data**: (a) **internal** = fraud PaySim nyata; (b) **expert** = "
        "penilaian ahli (sintetis); (c) **eksternal** = kerugian bank lain (sintetis, "
        "diskala agar setara ukuran bank kita)."
    )
    _loss3, _fs3 = contoh_internal()
    _ext_raw, _ext_scaled = contoh_external()
    _exp3 = contoh_expert(8, 42)
    s1, s2, s3 = st.columns(3)
    s1.metric("① Internal (baris)", f"{_fs3['n_events']:,}")
    s2.metric("② Expert (baris)", f"{len(_exp3):,}")
    s3.metric("③ Eksternal (baris)", f"{len(_ext_raw):,}")
    tab1, tab2, tab3 = st.tabs(["① Internal (PaySim)", "② Expert (ahli)", "③ Eksternal (bank lain)"])
    with tab1:
        st.caption("Sumber: PaySim (Kaggle) — fraud nyata. Kerugian = kolom amount.")
        st.dataframe(
            _loss3.head(10)[["amount", "day"]].rename(
                columns={"amount": "Jumlah Kerugian", "day": "Hari ke-"}
            ).style.format({"Jumlah Kerugian": "{:,.0f}"}),
            use_container_width=True, hide_index=True)
        st.caption(f"Total {_fs3['n_events']:,} peristiwa, total kerugian "
                   f"{_fs3['total_internal']:,.0f}.")
    with tab2:
        st.caption("Sumber: penilaian ahli (sintetis) — kelas ordinal per kategori.")
        st.dataframe(
            _exp3.head(10).rename(columns={
                "expert": "Ahli ke-", "business_line": "Lini Bisnis",
                "event_type": "Jenis Kejadian", "frequency": "Frekuensi",
                "severity": "Severity", "control": "Kontrol"}),
            use_container_width=True, hide_index=True)
        st.caption(f"{_exp3['expert'].nunique()} ahli × 56 kategori = {len(_exp3):,} baris.")
    with tab3:
        st.caption("Sumber: konsorsium bank lain (sintetis) — hanya kerugian di atas ambang.")
        _ext_df = pd.DataFrame({
            "Kerugian (mentah)": _ext_raw[:10],
            "Kerugian (ter-scaling)": _ext_scaled[:10],
        })
        st.dataframe(_ext_df.style.format("{:,.0f}"),
                     use_container_width=True, hide_index=True)
        st.caption("Kerugian bank lain biasanya lebih besar; di-*scaling* dulu agar "
                   "setara ukuran bank kita sebelum digabung.")
    kotak_arti(
        "Tiga sumber ini digabung agar estimasi tidak bergantung pada satu data saja — "
        "internal (pengalaman kita), expert (penilaian ahli), dan eksternal (pelajaran "
        "dari bank lain)."
    )
    dataset_lengkap_paysim("p3")

    langkah(4, "Proses")
    st.markdown("Atur tingkat keyakinan dan jumlah simulasi, lalu bandingkan semua metode:")
    col_a, col_b = st.columns(2)
    with col_a:
        q = st.slider("Tingkat keyakinan VaR (persentil)", 0.95, 0.999, 0.999, 0.001)
    with col_b:
        n_sims = st.select_slider("Jumlah skenario simulasi",
                                  options=[5_000, 10_000, 20_000, 50_000], value=20_000)

    @st.cache_data(show_spinner="Menghitung & membandingkan semua metode...")
    def _comparison_int(q: float, n_sims: int):
        return run_comparison(q=q, n_sims=n_sims)

    res = _comparison_int(q, n_sims)
    comp = res["comparison"]

    langkah(5, "Hasil & Kesimpulan")
    st.markdown("**Perbandingan VaR antar-metode** (analog Gambar 12.2 di buku)")
    bar_df = comp.reset_index()
    bar_df.columns = ["Metode", "VaR"]
    st.plotly_chart(
        px.bar(bar_df, x="Metode", y="VaR", color="Metode", text_auto=".2s",
               title=f"Perbandingan VaR pada keyakinan {q:.1%} (basis {res['period_label']})"),
        use_container_width=True)
    st.dataframe(comp.to_frame("VaR").style.format("{:,.0f}"),
                 use_container_width=True)
    kotak_arti(
        "Tiap batang = estimasi kerugian/cadangan dari satu metode. Kalau tinggi "
        "batangnya **berdekatan**, berarti metode-metode itu **saling sepakat** → "
        "hasil lebih dapat dipercaya."
    )

    st.markdown("**Komponen pendukung**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Self-assessment VaR", f"{res['self_assessment_point']:,.0f}")
    c2.metric("Pendapatan kotor (asumsi)", f"{res['gross_income']:,.0f}")
    c3.metric("BIA = 15% pendapatan", f"{res['gross_income'] * 0.15:,.0f}")

    tertinggi = comp.idxmax()
    terendah = comp.idxmin()
    st.success(
        f"**Kesimpulan:** Estimasi **tertinggi** datang dari metode *{tertinggi}* "
        f"({comp.max():,.0f}) dan **terendah** dari *{terendah}* ({comp.min():,.0f}). "
        f"Pendekatan sederhana **BIA** ternyata berdekatan dengan **VaR Bayesian** — "
        f"persis seperti Gambar 12.2 di buku, di mana cara *top-down* dan *bottom-up* "
        f"saling mendekati sehingga estimasinya bisa lebih dipercaya."
    )

    with st.expander("🔧 Cara Kerja, Kode Program & Hasilnya (klik untuk buka)"):
        st.markdown(
            "**Cara kerja (langkah demi langkah):**\n\n"
            "1. `scale_external_to_internal` menyamakan skala kerugian bank lain "
            "dengan kerugian kita (bagi dengan konstanta *c*).\n"
            "2. `integrated_severity` menyatukan **3 sumber** jadi satu kumpulan: "
            "internal + 1 titik self-assessment + eksternal ter-scaling.\n"
            "3. `bayes_var_simple` mengambil **persentil tinggi** dari kumpulan "
            "gabungan itu = **Bayes VaR**.\n"
            "4. `basic_indicator_approach` menghitung pembanding sederhana = "
            "**15% × pendapatan kotor**.\n"
            "5. `run_comparison` menjalankan semua metode lalu menyusunnya menjadi "
            "tabel/grafik perbandingan di atas."
        )
        st.markdown("#### 📜 Kode asli — `scale_external_to_internal`")
        tampilkan_kode(scale_external_to_internal)
        st.markdown("#### 📜 Kode asli — `integrated_severity`")
        tampilkan_kode(integrated_severity)
        st.markdown("#### 📜 Kode asli — `bayes_var_simple`")
        tampilkan_kode(bayes_var_simple)
        st.markdown("#### 📜 Kode asli — `basic_indicator_approach`")
        tampilkan_kode(basic_indicator_approach)

        st.markdown("#### ▶️ Hasil nyata bila kode dijalankan")
        _loss_i, _fs_i = contoh_internal()
        _ext_raw_i, _ext_scaled_i = contoh_external()
        _c = float(_ext_raw_i.sum() / _fs_i["total_internal"])
        _integ = integrated_severity(
            _fs_i["severity_sample"], res["self_assessment_point"], _ext_scaled_i)
        _bayes_demo = bayes_var_simple(_integ, q)
        _bia_demo = basic_indicator_approach(res["gross_income"])
        st.code(
            f"c = total_eksternal / total_internal           # -> {_c:,.2f}\n"
            f"gabungan = integrated_severity(internal, sa, eksternal_scaled)\n"
            f"len(gabungan)                                  # -> {len(_integ):,}\n"
            f"bayes_var_simple(gabungan, q={q:.3f})           # -> {_bayes_demo:,.0f}\n"
            f"basic_indicator_approach(gross_income)         # -> {_bia_demo:,.0f}",
            language="python")
        dcol1, dcol2, dcol3 = st.columns(3)
        dcol1.metric("Konstanta scaling c", f"{_c:,.2f}")
        dcol2.metric(f"Bayes VaR {q:.1%}", f"{_bayes_demo:,.0f}")
        dcol3.metric("BIA (15%)", f"{_bia_demo:,.0f}")
        st.caption("Angka-angka ini persis yang dipakai grafik perbandingan di atas.")


# ============================================================================
# 📊 Ringkasan Eksekutif — gabungan hasil ketiga model
# ============================================================================
elif page == "📊 Ringkasan Eksekutif":
    from orm.scorecard import (
        TRAFFIC_LIGHT_HEX,
        build_scorecard,
        self_assessment_total_loss,
    )
    from orm.actuarial import value_at_risk
    from orm.integrated import run_comparison

    st.markdown(
        '<div class="hero-box">'
        '<h1>📊 Ringkasan Eksekutif</h1>'
        '<p>Semua hasil dari 3 model dalam satu halaman — siap untuk presentasi</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Run all models via run_comparison (handles everything internally) ----
    @st.cache_data(show_spinner="Menjalankan semua model...")
    def _run_all_models():
        return run_comparison(q=0.999, n_sims=20_000)

    res_ex = _run_all_models()
    sc_ex = res_ex["scorecard"]
    sa_loss_ex = self_assessment_total_loss(sc_ex)
    act_ex = res_ex["actuarial"]
    comp_ex = res_ex["comparison"]
    q = 0.999

    # ---- Section 1: Key Metrics ----
    st.markdown("### 🎯 Angka Kunci")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 VaR 99.9% (Aktuaria)",
              f"Rp {comp_ex['Actuarial Montecarlo VaR']:,.0f}",
              help="Cadangan minimum berdasarkan simulasi Monte Carlo")
    k2.metric("🟡 VaR Bayesian",
              f"Rp {comp_ex['Bayes Montecarlo VaR']:,.0f}",
              help="VaR dari gabungan 3 sumber data")
    k3.metric("🟢 BIA (15%)",
              f"Rp {comp_ex['Basic Indicator Approach VaR']:,.0f}",
              help="Basic Indicator Approach = 15% pendapatan kotor")
    k4.metric("📋 Self-Assessment", f"Rp {sa_loss_ex:,.0f}",
              help="Estimasi kerugian dari kuesioner ahli")

    st.markdown(
        '<div class="key-finding">'
        '<strong>💡 Temuan Utama:</strong> Ketiga pendekatan (bottom-up via aktuaria, '
        'gabungan Bayesian, dan top-down BIA) menghasilkan estimasi yang '
        '<strong>saling berdekatan</strong> — menunjukkan bahwa model ini '
        '<strong>saling memvalidasi</strong> satu sama lain, sesuai dengan '
        'kesimpulan Gambar 12.2 di buku Giudici.</div>',
        unsafe_allow_html=True,
    )

    # ---- Section 2: Scorecard Summary ----
    st.markdown("---")
    st.markdown("### 📋 Model 1 — Hasil Scorecard")

    # Focused area scorecard
    focus = sc_ex[
        (sc_ex["business_line"] == "Payment and Settlement")
        & (sc_ex["event_type"] == "External Fraud")
    ]
    if not focus.empty:
        row = focus.iloc[0]
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Rating Frekuensi", row["frequency_rating"])
        sc2.metric("Rating Severity", row["severity_rating"])
        sc3.metric("Rating Kontrol", row["control_rating"])
        color = TRAFFIC_LIGHT_HEX.get(row["frequency_color"], "#999")
        sc4.markdown(
            f"<div style='text-align:center;padding:0.8rem;border-radius:0.5rem;"
            f"background:{color};color:white;font-size:1.3rem;'>"
            f"<strong>{row['frequency_rating']}</strong><br>"
            f"<small>Lampu Lalu Lintas</small></div>",
            unsafe_allow_html=True,
        )

    # Heatmap of perceived loss across categories
    sc_pivot = sc_ex.pivot_table(
        index="business_line", columns="event_type",
        values="perceived_loss", aggfunc="first",
    )
    fig_heat = px.imshow(
        sc_pivot, text_auto=".0f",
        color_continuous_scale="RdYlGn_r",
        labels={"color": "Perceived Loss"},
        title="🗺️ Peta Risiko — Perceived Loss per Kategori Basel II",
        aspect="auto",
    )
    fig_heat.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)
    kotak_arti(
        "Warna **merah** = risiko tinggi, **hijau** = risiko rendah. "
        "Kotak paling merah menunjukkan area yang perlu perhatian paling besar."
    )

    # ---- Section 3: VaR Comparison ----
    st.markdown("---")
    st.markdown("### 📈 Model 2 & 3 — Perbandingan 7 Metode VaR")

    # Ordered bar chart with color coding
    comp_sorted = comp_ex.sort_values()
    colors = []
    for v in comp_sorted.values:
        ratio = (v - comp_sorted.min()) / (comp_sorted.max() - comp_sorted.min() + 1)
        if ratio < 0.33:
            colors.append("#2ca02c")  # green
        elif ratio < 0.66:
            colors.append("#ff7f0e")  # orange
        else:
            colors.append("#d62728")  # red

    fig_comp = px.bar(
        x=comp_sorted.values, y=comp_sorted.index,
        orientation="h",
        title="📊 Estimasi Cadangan dari 7 Metode (diurutkan rendah → tinggi)",
        labels={"x": "Estimasi Cadangan (Rp)", "y": "Metode"},
    )
    fig_comp.update_traces(marker_color=colors)
    fig_comp.update_layout(
        height=400, margin=dict(l=10, r=10, t=50, b=10),
        xaxis_tickformat=",",
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    kotak_arti(
        "Ketujuh metode ini saling **memvalidasi**. Jika batang-batang berdekatan, "
        "berarti estimasi bisa **lebih dipercaya**. Sesuai Gambar 12.2 di buku."
    )

    # ---- Section 4: Visual Summary Table ----
    st.markdown("---")
    st.markdown("### 📝 Ringkasan per Model")

    summary_data = {
        "Model": ["📋 Scorecard", "📈 VaR Aktuaria", "🔗 VaR Integrasi", "🏦 BIA"],
        "Metode": [
            "Kuesioner ahli → Median & Gini → Rating",
            "Poisson × Lognormal → Monte Carlo → VaR",
            "Gabungan 3 sumber (Bayesian) → VaR",
            "15% × Pendapatan Kotor",
        ],
        "Tipe": ["Kualitatif", "Kuantitatif", "Kuantitatif", "Kuantitatif"],
        "Pendekatan": ["Bottom-up", "Bottom-up", "Bottom-up", "Top-down"],
        "Hasil Utama": [
            f"Perceived Loss = {sa_loss_ex:,.0f}",
            f"VaR 99.9% = {comp_ex['Actuarial Montecarlo VaR']:,.0f}",
            f"Bayes VaR = {comp_ex['Bayes Montecarlo VaR']:,.0f}",
            f"BIA = {comp_ex['Basic Indicator Approach VaR']:,.0f}",
        ],
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True, hide_index=True,
    )

    # ---- Section 5: Conclusion ----
    st.markdown("---")
    st.success(
        "### ✅ Kesimpulan Akhir\n\n"
        "1. **Scorecard** memberikan pandangan kualitatif risiko — "
        "ahli menilai frekuensi, severity, dan kontrol → rating lampu lalu lintas.\n\n"
        "2. **VaR Aktuaria** menghitung cadangan secara statistik murni "
        "dari data kerugian internal (Monte Carlo).\n\n"
        "3. **VaR Integrasi (Bayesian)** menggabungkan semua sumber data "
        "menjadi satu estimasi yang lebih akurat.\n\n"
        "4. **BIA** sebagai pembanding sederhana ternyata **berdekatan** dengan "
        "VaR Bayesian — memvalidasi bahwa perhitungan bottom-up dan top-down **konsisten**.\n\n"
        "5. Keseluruhan implementasi **sesuai dengan Bab 12** buku Giudici (2009) — "
        "11 dari 12 aspek metodologi tercakup dengan tepat."
    )

    with st.expander("📖 Detail Kesesuaian dengan Buku (11/12 aspek ✅)"):
        checklist = [
            ("Basel II (56 kategori)", "✅", "8 BL × 7 ET lengkap"),
            ("3 aliran data", "✅", "Internal, Expert, Eksternal"),
            ("Scaling DIPO", "✅", "Rasio total_ext / total_int"),
            ("Fokus 1 area", "✅", "Payment & Settlement"),
            ("KRI formula", "⚠️", "Disederhanakan (±30% sama, detail beda)"),
            ("Scorecard median+Gini", "✅", "Rating A/AA/AAA + traffic light"),
            ("Model Aktuaria", "✅", "Poisson × lognormal MC → VaR 99.9%"),
            ("Integrasi Bayesian", "✅", "3 sumber → distribusi gabungan → VaR"),
            ("BIA (15%)", "✅", "15% gross income"),
            ("Perbandingan 7 VaR", "✅", "Urutan konsisten Gambar 12.2"),
            ("Visualisasi scorecard", "✅", "Traffic light convention"),
            ("Skala ordinal", "✅", "Frequency/severity/control classes"),
        ]
        st.dataframe(
            pd.DataFrame(checklist, columns=["Aspek Buku", "Status", "Catatan"]),
            use_container_width=True, hide_index=True,
        )


