"""
orm/data_sources.py — Tiga aliran data risiko operasional (Bab 12.2).

Buku menggunakan tiga sumber informasi yang digabung lewat proses *scaling*:

  1. Internal loss data  — tabel kerugian internal (jumlah, tanggal, unit, dst).
                           Di sini diambil dari transaksi PaySim yang berlabel
                           fraud: `amount` = kerugian, `step` (jam) -> hari.
                           Sesuai buku, pemetaan difokuskan pada satu area:
                           business line "Payment and Settlement", event type
                           "External Fraud" (Giudici 2009, hlm. 230).
  2. Expert opinion      — kuesioner self-assessment: beberapa expert menilai
                           frekuensi, severity, dan kualitas kontrol tiap event
                           type (kelas ordinal). Bersifat sintetis namun setia
                           pada metodologi (data expert asli tidak tersedia).
  3. External loss data  — basis data konsorsium (mis. DIPO): kerugian gabungan
                           beberapa bank, biasanya lebih besar dan di atas ambang
                           tertentu (mis. EUR 5000). Dibangkitkan sintetis.

Proses *scaling* (hlm. 229): kerugian eksternal dibagi konstanta = rasio total
kerugian eksternal terhadap total kerugian internal, agar dapat dibandingkan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from orm.categories import (
    BUSINESS_LINES,
    CONTROL_CLASSES,
    EVENT_TYPES,
    FREQUENCY_CLASSES,
    SEVERITY_CLASSES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "paysim_processed.parquet"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Area fokus pemetaan (lihat buku: hanya area payment & settlement yang dipetakan)
INTERNAL_BUSINESS_LINE = "Payment and Settlement"
INTERNAL_EVENT_TYPE = "External Fraud"

# PaySim: 1 step = 1 jam. Rentang simulasi dipakai untuk annualisasi frekuensi.
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365.0


# ---------------------------------------------------------------------------
# 1. Internal loss data (dari PaySim)
# ---------------------------------------------------------------------------
def load_internal_losses(min_amount: float = 0.0) -> pd.DataFrame:
    """Muat kerugian internal dari transaksi fraud PaySim.

    Returns DataFrame kolom: business_line, event_type, day, amount.
    """
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} tidak ada. Jalankan dulu `python -m src.data` "
            "(atau `python -m src.make_sample_data` lalu `python -m src.data`)."
        )
    df = pd.read_parquet(PROCESSED_PATH, columns=["isFraud", "amount", "step"])
    loss = df[(df["isFraud"] == 1) & (df["amount"] > min_amount)].copy()
    loss["day"] = (loss["step"] // HOURS_PER_DAY).astype(int)
    loss["business_line"] = INTERNAL_BUSINESS_LINE
    loss["event_type"] = INTERNAL_EVENT_TYPE
    return loss[["business_line", "event_type", "day", "amount"]].reset_index(drop=True)


def internal_frequency_severity(losses: pd.DataFrame) -> dict:
    """Ringkas aliran internal menjadi parameter frekuensi & severity tahunan.

    Frekuensi diannualisasi dari rentang hari yang teramati. Severity adalah
    sampel nilai kerugian individual (dipakai untuk fit distribusi di modul
    aktuaria).
    """
    n_events = len(losses)
    span_days = max(1, int(losses["day"].max() - losses["day"].min() + 1))
    annual_frequency = n_events * (DAYS_PER_YEAR / span_days)
    return {
        "n_events": n_events,
        "span_days": span_days,
        "annual_frequency": float(annual_frequency),
        "severity_sample": losses["amount"].to_numpy(dtype=float),
        "total_internal": float(losses["amount"].sum()),
    }


# ---------------------------------------------------------------------------
# 2. Expert opinion (kuesioner self-assessment sintetis)
# ---------------------------------------------------------------------------
def generate_expert_opinions(
    n_experts: int = 8,
    business_lines: list[str] | None = None,
    event_types: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Bangkitkan kuesioner self-assessment untuk seluruh 56 kategori Basel.

    Tiap expert menilai, untuk tiap (business line, event type), satu kelas
    frekuensi, severity, dan kontrol. Penilaian dibangkitkan dari "profil risiko
    laten" tiap kategori ditambah ketidaksepakatan antar-expert, sehingga indeks
    Gini pada modul scorecard mencerminkan konsensus yang bervariasi.

    Returns DataFrame kolom: expert, business_line, event_type,
    frequency, severity, control.
    """
    rng = np.random.default_rng(seed)
    business_lines = business_lines or BUSINESS_LINES
    event_types = event_types or EVENT_TYPES

    rows: list[dict] = []
    for bl in business_lines:
        for et in event_types:
            # Profil laten (rank pusat) + tingkat konsensus per dimensi.
            freq_center = rng.integers(0, len(FREQUENCY_CLASSES))
            sev_center = rng.integers(0, len(SEVERITY_CLASSES))
            ctrl_center = rng.integers(0, len(CONTROL_CLASSES))
            # Sebaran (dispersi) kecil = expert sepakat, besar = beragam.
            freq_sd = rng.uniform(0.3, 1.6)
            sev_sd = rng.uniform(0.3, 1.6)
            ctrl_sd = rng.uniform(0.2, 1.0)

            for k in range(n_experts):
                rows.append({
                    "expert": k + 1,
                    "business_line": bl,
                    "event_type": et,
                    "frequency": _sample_class(rng, FREQUENCY_CLASSES, freq_center, freq_sd),
                    "severity": _sample_class(rng, SEVERITY_CLASSES, sev_center, sev_sd),
                    "control": _sample_class(rng, CONTROL_CLASSES, ctrl_center, ctrl_sd),
                })
    return pd.DataFrame(rows)


def _sample_class(rng: np.random.Generator, classes: list[str],
                  center: int, sd: float) -> str:
    """Ambil satu label kelas ordinal di sekitar `center` dengan sebaran `sd`."""
    idx = int(np.clip(round(rng.normal(center, sd)), 0, len(classes) - 1))
    return classes[idx]


# ---------------------------------------------------------------------------
# 3. External loss data (konsorsium, sintetis)
# ---------------------------------------------------------------------------
def generate_external_losses(
    n: int = 2_000,
    threshold: float = 5_000.0,
    scale_multiplier: float = 3.0,
    internal_severity: np.ndarray | None = None,
    seed: int = 7,
) -> np.ndarray:
    """Bangkitkan kerugian eksternal terpool (di atas `threshold`).

    Buku: kerugian eksternal (DIPO) cenderung LEBIH BESAR daripada kerugian
    internal (hlm. 229) dan hanya mencatat kerugian di atas ambang tertentu
    (mis. EUR 5000, hlm. 233). `scale_multiplier` mengatur seberapa besar
    relatif terhadap median kerugian internal.
    """
    rng = np.random.default_rng(seed)
    if internal_severity is not None and len(internal_severity) > 0:
        pos = internal_severity[internal_severity > 0]
        mu = float(np.log(np.median(pos))) + np.log(scale_multiplier)
        sigma = float(np.std(np.log(pos))) or 1.0
    else:
        mu, sigma = np.log(threshold * scale_multiplier), 1.2
    draws = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return draws[draws >= threshold]


# ---------------------------------------------------------------------------
# Scaling (Bab 12.2)
# ---------------------------------------------------------------------------
def scale_external_to_internal(external: np.ndarray, total_internal: float,
                               total_external: float | None = None) -> np.ndarray:
    """Skalakan kerugian eksternal agar sebanding dengan internal.

    Konstanta scaling = total kerugian eksternal / total kerugian internal
    (Giudici 2009, hlm. 229). Hasilnya dibagi konstanta tersebut.
    """
    total_external = total_external if total_external is not None else float(external.sum())
    if total_internal <= 0 or total_external <= 0:
        return external
    constant = total_external / total_internal
    return external / constant


if __name__ == "__main__":
    internal = load_internal_losses()
    fs = internal_frequency_severity(internal)
    print("=== Internal loss data (PaySim fraud) ===")
    print(f"  events={fs['n_events']:,}  span={fs['span_days']} hari  "
          f"freq_tahunan={fs['annual_frequency']:,.0f}")
    print(f"  total kerugian internal = {fs['total_internal']:,.0f}")

    experts = generate_expert_opinions()
    n_cat = experts.groupby(["business_line", "event_type"]).ngroups
    print(f"\n=== Expert opinion: {experts['expert'].nunique()} expert x "
          f"{n_cat} kategori (8 BL x 7 ET) = {len(experts)} baris ===")
    print(experts.head(7).to_string(index=False))

    ext = generate_external_losses(internal_severity=fs["severity_sample"])
    ext_scaled = scale_external_to_internal(ext, fs["total_internal"])
    print(f"\n=== External loss data (sintetis): {len(ext)} kerugian > ambang ===")
    print(f"  median mentah={np.median(ext):,.0f}  median ter-scale={np.median(ext_scaled):,.0f}")
