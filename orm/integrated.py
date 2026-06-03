"""
orm/integrated.py — Model terintegrasi (Bayesian) & perbandingan VaR (Gambar 12.2).

Pendekatan aktuaria hanya memakai data kerugian internal (backward-looking),
sedangkan scorecard hanya memakai opini self-assessment (forward-looking). Buku
mengusulkan menggabungkan TIGA aliran data — internal, self-assessment, eksternal
— menjadi satu distribusi kerugian terintegrasi, lalu menurunkan VaR darinya
(Giudici 2009, hlm. 233).

Untuk tiap event dipertimbangkan: seluruh kerugian masa lalu + ekspektasi
kerugian self-assessment periode berikut (dihitung sebagai SATU titik data,
biasanya lebih tinggi dari kerugian aktual) + kerugian eksternal terpool.

  * Bayes VaR (simple) : persentil dari distribusi kerugian terintegrasi.
  * Bayes MC VaR       : simulasi Monte Carlo agregat berbasis kerugian
                         terintegrasi (paralel dengan pendekatan aktuaria).

Sebagai tolok ukur top-down disertakan Basic Indicator Approach (BIA) Basel II:
VaR = 15% dari indikator relevan (gross income) (hlm. 235).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orm import actuarial
from orm.data_sources import (
    INTERNAL_BUSINESS_LINE,
    INTERNAL_EVENT_TYPE,
    generate_expert_opinions,
    generate_external_losses,
    internal_frequency_severity,
    load_internal_losses,
    scale_external_to_internal,
)
from orm.scorecard import build_scorecard

BIA_ALPHA = 0.15  # persentase Basic Indicator Approach (Basel II)


# ---------------------------------------------------------------------------
# Self-assessment
# ---------------------------------------------------------------------------
def self_assessment_point(
    scorecard: pd.DataFrame,
    business_line: str = INTERNAL_BUSINESS_LINE,
    event_type: str = INTERNAL_EVENT_TYPE,
) -> float:
    """Satu titik data forward-looking untuk kategori yang dimodelkan.

    Buku: untuk tiap event, ekspektasi kerugian self-assessment dihitung sebagai
    SATU titik data, "typically higher than actual losses" (hlm. 233). Dipilih
    perceived loss kategori yang sama dengan aliran internal (default Payment &
    Settlement / External Fraud). Bila kategori tidak ada, pakai perceived loss
    terbesar sebagai cadangan.
    """
    sel = scorecard[
        (scorecard["business_line"] == business_line)
        & (scorecard["event_type"] == event_type)
    ]
    if len(sel):
        return float(sel["perceived_loss"].iloc[0])
    return float(scorecard["perceived_loss"].max())


def self_assessment_var(scorecard: pd.DataFrame, q: float = actuarial.DEFAULT_VAR_Q) -> float:
    """VaR self-assessment murni: persentil tinggi dari perceived loss antar kategori."""
    return float(np.percentile(scorecard["perceived_loss"].to_numpy(), q * 100.0))


# ---------------------------------------------------------------------------
# Integrasi tiga aliran data
# ---------------------------------------------------------------------------
def integrated_severity(
    internal_severity: np.ndarray,
    sa_point: float,
    external_scaled: np.ndarray,
) -> np.ndarray:
    """Gabungkan kerugian internal + 1 titik self-assessment + eksternal ter-scale."""
    pos = np.asarray(internal_severity, dtype=float)
    pos = pos[pos > 0]
    return np.concatenate([pos, np.array([sa_point], dtype=float),
                           np.asarray(external_scaled, dtype=float)])


def bayes_var_simple(integrated_sev: np.ndarray, q: float = actuarial.DEFAULT_VAR_Q) -> float:
    """Bayes VaR sederhana: persentil dari distribusi kerugian terintegrasi."""
    return float(np.percentile(integrated_sev, q * 100.0))


def bayes_var_montecarlo(
    frequency: float,
    integrated_sev: np.ndarray,
    q: float = actuarial.DEFAULT_VAR_Q,
    n_sims: int = actuarial.DEFAULT_N_SIMS,
    seed: int = 42,
) -> float:
    """Bayes Monte Carlo VaR: agregat MC berbasis kerugian terintegrasi (empiris)."""
    losses = actuarial.simulate_aggregate(
        frequency, severity_sample=integrated_sev, empirical=True,
        n_sims=n_sims, seed=seed)
    return actuarial.value_at_risk(losses, q)


# ---------------------------------------------------------------------------
# Basic Indicator Approach (top-down)
# ---------------------------------------------------------------------------
def basic_indicator_approach(gross_income: float, alpha: float = BIA_ALPHA) -> float:
    """BIA Basel II: VaR = alpha * gross income (alpha = 15%)."""
    return float(alpha * gross_income)


# ---------------------------------------------------------------------------
# Perbandingan menyeluruh (analog Gambar 12.2)
# ---------------------------------------------------------------------------
def run_comparison(
    q: float = actuarial.DEFAULT_VAR_Q,
    n_sims: int = actuarial.DEFAULT_N_SIMS,
    gross_income: float | None = None,
    frequency: float | None = None,
    period_label: str = "harian",
    n_experts: int = 8,
    seed: int = 42,
) -> dict:
    """Hitung seluruh metrik VaR dan kembalikan tabel perbandingan (Gambar 12.2).

    SEMUA metrik memakai basis periode yang sama (default HARIAN) agar saling
    konsisten; yang dibandingkan adalah URUTAN/relasi antar-metode, bukan
    besaran absolut (skala mengikuti karakteristik data PaySim).

    Args:
        frequency: frekuensi kejadian internal per periode. Bila None, dipakai
            frekuensi harian (n_events / span_days).
        gross_income: indikator BIA pada basis periode yang sama. Bila None,
            ditetapkan ilustratif sehingga BIA setara Bayes VaR (mirip Gambar
            12.2 di mana BIA ~= Bayes VaR).
    """
    # --- Data internal (PaySim) ---
    internal = load_internal_losses()
    fs = internal_frequency_severity(internal)
    freq = frequency if frequency is not None else fs["n_events"] / fs["span_days"]
    sev = fs["severity_sample"]

    # --- Aktuaria ---
    act = actuarial.actuarial_summary(freq, sev, q=q, n_sims=n_sims, seed=seed)

    # --- Self-assessment (scorecard 56 kategori) ---
    opinions = generate_expert_opinions(n_experts=n_experts, seed=seed)
    scorecard = build_scorecard(opinions)
    sa_point = self_assessment_point(
        scorecard, INTERNAL_BUSINESS_LINE, INTERNAL_EVENT_TYPE)
    sa_var = self_assessment_var(scorecard, q=q)

    # --- Eksternal (sintetis) + scaling ---
    external = generate_external_losses(internal_severity=sev, seed=seed)
    external_scaled = scale_external_to_internal(external, fs["total_internal"])

    # --- Integrasi Bayesian ---
    integ = integrated_severity(sev, sa_point, external_scaled)
    bayes_simple = bayes_var_simple(integ, q=q)
    bayes_mc = bayes_var_montecarlo(freq, integ, q=q, n_sims=n_sims, seed=seed)

    # --- BIA ---
    if gross_income is None:
        gross_income = bayes_simple / BIA_ALPHA
    bia = basic_indicator_approach(gross_income)

    comparison = pd.Series({
        "Historical losses": act["historical_losses"],
        "Actuarial Montecarlo VaR": act["actuarial_montecarlo_var"],
        "Historical VaR": act["historical_var"],
        "Bayes Montecarlo VaR": bayes_mc,
        "Bayes VaR": bayes_simple,
        "Basic Indicator Approach VaR": bia,
        "Self-assessment VaR": sa_var,
    }, name="VaR")

    return {
        "comparison": comparison,
        "scorecard": scorecard,
        "actuarial": act,
        "frequency": freq,
        "period_label": period_label,
        "self_assessment_point": sa_point,
        "gross_income": gross_income,
        "internal_summary": {k: v for k, v in fs.items() if not k.startswith("severity")},
    }


if __name__ == "__main__":
    res = run_comparison()
    print(f"=== Perbandingan VaR (analog Gambar 12.2, basis {res['period_label']}) ===")
    print(res["comparison"].map(lambda x: f"{x:,.0f}").to_string())
    print(f"\nGross income (BIA, ilustratif) = {res['gross_income']:,.0f}")
    print(f"Frekuensi internal              = {res['frequency']:.1f} kejadian/{res['period_label']}")
    act = res["actuarial"]
    print("\nDiagnostik ekor severity (risiko-model):")
    print(f"  sigma lognormal            = {act['severity_sigma']:.3f}")
    print(f"  severity maks teramati     = {act['severity_max_observed']:,.0f}")
    print(f"  severity empiris  P99.9    = {act['severity_empirical_q999']:,.0f}")
    print(f"  severity lognormal P99.9   = {act['severity_lognormal_q999']:,.0f}")
