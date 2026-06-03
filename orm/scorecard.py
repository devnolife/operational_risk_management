"""
orm/scorecard.py — Model scorecard self-assessment (Bab 12.4, Gambar 12.1).

Untuk tiap event risiko, distribusi opini expert pada frekuensi / severity /
kontrol diringkas menjadi sebuah *rating*:

  * Lokasi    : KELAS MEDIAN dipakai untuk menetapkan huruf dasar (A = risiko
                terendah, B, C, ...).
  * Konsensus : INDEKS GINI ternormalisasi dipakai untuk menggandakan/melipat-
                tigakan huruf. Jika semua expert sepakat (Gini = 0) maka A -> AAA;
                jika heterogenitas maksimum (Gini ternormalisasi = 1) maka A tetap
                A; kasus antara -> AA (Giudici 2009, hlm. 232-233).

Konvensi lampu lalu lintas untuk visualisasi: A = hijau, B = kuning, C dst = merah.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orm.categories import (
    CONTROL_LOSS_FACTOR,
    FREQUENCY_PER_YEAR,
    SEVERITY_MIDPOINTS,
    SCALES,
    rank_to_letter,
)

# Warna lampu lalu lintas per huruf dasar.
TRAFFIC_LIGHT: dict[str, str] = {
    "A": "green",
    "B": "yellow",
    "C": "red",
}
TRAFFIC_LIGHT_HEX: dict[str, str] = {
    "green": "#2ca02c",
    "yellow": "#f1c40f",
    "red": "#d62728",
}


def median_class_rank(votes: list[str], dimension: str) -> int:
    """Peringkat kelas median (1-based) dari daftar label opini expert.

    Memakai KELAS yang benar-benar dipilih (bukan interpolasi). Untuk jumlah
    expert genap dengan dua kelas tengah berbeda, dipilih upper median
    (konservatif terhadap risiko).
    """
    if not votes:
        raise ValueError("Daftar votes kosong.")
    classes = SCALES[dimension]
    ranks = np.sort(np.array([classes.index(v) + 1 for v in votes], dtype=int))
    upper_median = int(ranks[len(ranks) // 2])  # upper median utk n genap
    return int(np.clip(upper_median, 1, len(classes)))


def normalized_gini(votes: list[str], dimension: str) -> float:
    """Indeks heterogenitas Gini ternormalisasi pada [0, 1].

    G = 1 - sum p_k^2 ;  G_norm = G * K / (K - 1).
    0 = semua expert satu kelas (konsensus penuh); 1 = heterogenitas maksimum.
    """
    if not votes:
        raise ValueError("Daftar votes kosong.")
    classes = SCALES[dimension]
    k = len(classes)
    counts = pd.Series(votes).value_counts()
    p = counts.to_numpy(dtype=float) / len(votes)
    gini = 1.0 - float(np.sum(p ** 2))
    if k <= 1:
        return 0.0
    return gini * k / (k - 1)


def consensus_multiplicity(gini_norm: float) -> int:
    """Jumlah huruf (1-3) berdasarkan indeks Gini ternormalisasi.

    G_norm <= 1/3  -> 3 huruf (AAA, konsensus tinggi)
    1/3 < G_norm <= 2/3 -> 2 huruf (AA)
    G_norm > 2/3   -> 1 huruf (A, konsensus rendah)
    """
    if gini_norm <= 1 / 3:
        return 3
    if gini_norm <= 2 / 3:
        return 2
    return 1


def rate_dimension(votes: list[str], dimension: str) -> dict:
    """Hitung rating satu dimensi (frequency/severity/control) untuk satu event."""
    rank = median_class_rank(votes, dimension)
    n_classes = len(SCALES[dimension])
    letter = rank_to_letter(rank, n_classes)
    gini = normalized_gini(votes, dimension)
    mult = consensus_multiplicity(gini)
    color = TRAFFIC_LIGHT.get(letter, "red")
    return {
        "median_class": SCALES[dimension][rank - 1],
        "rank": rank,
        "letter": letter,
        "rating": letter * mult,
        "gini": round(gini, 3),
        "color": color,
    }


def build_scorecard(expert_opinions: pd.DataFrame) -> pd.DataFrame:
    """Bangun scorecard per event type (analog Gambar 12.1).

    Args:
        expert_opinions: keluaran data_sources.generate_expert_opinions.

    Returns DataFrame satu baris per (business_line, event_type) dengan rating
    frekuensi, severity, kontrol; warna; serta perkiraan perceived loss.
    """
    rows: list[dict] = []
    group_cols = ["business_line", "event_type"]
    for (bl, et), g in expert_opinions.groupby(group_cols):
        freq = rate_dimension(g["frequency"].tolist(), "frequency")
        sev = rate_dimension(g["severity"].tolist(), "severity")
        ctrl = rate_dimension(g["control"].tolist(), "control")

        perceived = (
            FREQUENCY_PER_YEAR[freq["median_class"]]
            * SEVERITY_MIDPOINTS[sev["median_class"]]
            * CONTROL_LOSS_FACTOR[ctrl["median_class"]]
        )
        rows.append({
            "business_line": bl,
            "event_type": et,
            "frequency_rating": freq["rating"],
            "frequency_class": freq["median_class"],
            "frequency_color": freq["color"],
            "severity_rating": sev["rating"],
            "severity_class": sev["median_class"],
            "severity_color": sev["color"],
            "control_rating": ctrl["rating"],
            "control_class": ctrl["median_class"],
            "control_color": ctrl["color"],
            "perceived_loss": perceived,
            "priority_score": freq["rank"] * sev["rank"] * ctrl["rank"],
        })
    out = pd.DataFrame(rows)
    # Prioritas intervensi tertinggi di atas (frekuensi tinggi, kontrol buruk).
    return out.sort_values("priority_score", ascending=False).reset_index(drop=True)


def self_assessment_total_loss(scorecard: pd.DataFrame) -> float:
    """Estimasi kualitatif total perceived loss (jumlah seluruh kategori)."""
    return float(scorecard["perceived_loss"].sum())


if __name__ == "__main__":
    from orm.data_sources import generate_expert_opinions

    opinions = generate_expert_opinions()
    sc = build_scorecard(opinions)
    cols = [
        "event_type", "frequency_rating", "severity_rating",
        "control_rating", "perceived_loss", "priority_score",
    ]
    print("=== Scorecard self-assessment (urut prioritas intervensi) ===")
    print(sc[cols].to_string(index=False))
    print(f"\nTotal perceived loss (self-assessment): "
          f"{self_assessment_total_loss(sc):,.0f}")
