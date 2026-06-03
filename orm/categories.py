"""
orm/categories.py — Kerangka klasifikasi Basel II untuk risiko operasional.

Risk Management Group dari Basel Committee mendefinisikan klasifikasi standar
kerugian operasional ke dalam 8 business line dan 7 event type, menghasilkan
8 x 7 = 56 kemungkinan kategori (Giudici 2009, hlm. 228).

Modul ini juga mendefinisikan skala ordinal yang dipakai pada kuesioner
self-assessment (Bab 12.4): frekuensi, severity, dan kualitas kontrol. Setiap
skala diurutkan berdasarkan tingkat risiko yang MENINGKAT, sehingga peringkat
(rank) 1 = risiko terendah (huruf A).
"""
from __future__ import annotations

from itertools import product

# ---------------------------------------------------------------------------
# 8 Business line Basel II
# ---------------------------------------------------------------------------
BUSINESS_LINES: list[str] = [
    "Corporate Finance",
    "Trading and Sales",
    "Retail Banking",
    "Commercial Banking",
    "Payment and Settlement",
    "Agency Services",
    "Asset Management",
    "Retail Brokerage",
]

# Faktor beta Standardised Approach (Basel II) untuk tiap business line.
# Dipakai pada pendekatan top-down (Bab 12.3).
BETA_FACTORS: dict[str, float] = {
    "Corporate Finance": 0.18,
    "Trading and Sales": 0.18,
    "Retail Banking": 0.12,
    "Commercial Banking": 0.15,
    "Payment and Settlement": 0.18,
    "Agency Services": 0.15,
    "Asset Management": 0.12,
    "Retail Brokerage": 0.12,
}

# ---------------------------------------------------------------------------
# 7 Event type Basel II
# ---------------------------------------------------------------------------
EVENT_TYPES: list[str] = [
    "Internal Fraud",
    "External Fraud",
    "Employment Practices and Workplace Safety",
    "Clients, Products and Business Practices",
    "Damage to Physical Assets",
    "Business Disruption and System Failures",
    "Execution, Delivery and Process Management",
]


def all_categories() -> list[tuple[str, str]]:
    """Kembalikan 56 kategori (business_line, event_type)."""
    return list(product(BUSINESS_LINES, EVENT_TYPES))


def category_id(business_line: str, event_type: str) -> str:
    """ID ringkas kategori, mis. 'BL5-ET2' (Payment & Settlement / External Fraud)."""
    bl = BUSINESS_LINES.index(business_line) + 1
    et = EVENT_TYPES.index(event_type) + 1
    return f"BL{bl}-ET{et}"


# ---------------------------------------------------------------------------
# Skala ordinal kuesioner self-assessment (Bab 12.4)
# Setiap daftar diurutkan dari RISIKO TERENDAH -> TERTINGGI.
# Peringkat (rank) = indeks + 1, sehingga rank 1 = huruf A (risiko terendah).
# ---------------------------------------------------------------------------

# Frekuensi: 4 kelas (daily, weekly, monthly, yearly).
# 'yearly' = paling jarang = risiko terendah (buku: median yearly -> huruf A).
FREQUENCY_CLASSES: list[str] = ["yearly", "monthly", "weekly", "daily"]

# Severity: tipikal 6-7 kelas, dari 'irrelevant loss' -> 'catastrophic loss'.
SEVERITY_CLASSES: list[str] = [
    "irrelevant",
    "minor",
    "moderate",
    "significant",
    "major",
    "severe",
    "catastrophic",
]

# Kontrol: 3 kelas. 'effective' = risiko terendah, 'not effective' = tertinggi.
CONTROL_CLASSES: list[str] = ["effective", "to be adjusted", "not effective"]

# Faktor koreksi perceived loss berdasarkan kualitas kontrol — bentuk sederhana
# dari faktor KRI F_{i,j,k} = 1 +/- 0.3 R pada Bab 12.2 (Giudici 2009, hlm. 230):
# kontrol buruk menaikkan ekspektasi kerugian, kontrol baik menurunkannya.
CONTROL_LOSS_FACTOR: dict[str, float] = {
    "effective": 0.7,
    "to be adjusted": 1.0,
    "not effective": 1.3,
}

# Perkiraan nilai moneter (EUR) tiap kelas severity — dipakai untuk mengubah
# opini severity kualitatif menjadi perkiraan kerugian (perceived loss).
SEVERITY_MIDPOINTS: dict[str, float] = {
    "irrelevant": 1_000.0,
    "minor": 5_000.0,
    "moderate": 25_000.0,
    "significant": 100_000.0,
    "major": 500_000.0,
    "severe": 2_000_000.0,
    "catastrophic": 10_000_000.0,
}

# Perkiraan frekuensi tahunan tiap kelas frekuensi.
FREQUENCY_PER_YEAR: dict[str, float] = {
    "yearly": 1.0,
    "monthly": 12.0,
    "weekly": 52.0,
    "daily": 250.0,  # hari kerja per tahun
}

SCALES: dict[str, list[str]] = {
    "frequency": FREQUENCY_CLASSES,
    "severity": SEVERITY_CLASSES,
    "control": CONTROL_CLASSES,
}


def class_rank(dimension: str, label: str) -> int:
    """Peringkat risiko 1-based untuk sebuah label kelas (1 = risiko terendah)."""
    return SCALES[dimension].index(label) + 1


def rank_to_letter(rank: int, n_classes: int) -> str:
    """Petakan peringkat risiko -> huruf rating dasar (A, B, C, ...).

    rank 1 (risiko terendah) -> 'A'. Buku memakai A untuk risiko rendah,
    B medium, C lebih tinggi, dst (Giudici 2009, hlm. 232).
    """
    rank = max(1, min(rank, n_classes))
    return chr(ord("A") + rank - 1)
