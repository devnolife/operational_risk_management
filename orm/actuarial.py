"""
orm/actuarial.py — Model aktuaria (Bab 12.3, Gambar 12.2).

Kejadian kerugian diasumsikan independen. Untuk event ke-j, total kerugian
dalam satu periode adalah penjumlahan dari sejumlah acak N impak X_i:

    L_j = sum_{i=1}^{N_j} X_ij                      (Giudici 2009, hlm. 231)

Distribusi L diperoleh dengan mengonvolusikan distribusi frekuensi N (Poisson)
dan distribusi severity S (lognormal), umumnya lewat estimasi Monte Carlo.
Dari distribusi L diturunkan fungsi yang diminati, mis. persentil 99.9 (VaR).

Catatan skala: severity diambil dari kerugian fraud PaySim (mean ~1.5 juta),
sehingga besaran VaR mengikuti karakteristik dataset tersebut. Periode default
adalah HARIAN (frekuensi = rata-rata kejadian per hari), agar simulasi tetap
ringan; perbandingan antar-metode bersifat konsisten karena memakai data sama.
"""
from __future__ import annotations

import numpy as np

DEFAULT_N_SIMS = 20_000
DEFAULT_VAR_Q = 0.999  # persentil 99.9 (value at risk)
_MAX_DRAWS = 50_000_000  # pagar memori untuk simulasi Monte Carlo


def fit_severity_lognormal(severity_sample: np.ndarray) -> tuple[float, float]:
    """Estimasi parameter (mu, sigma) lognormal dari sampel severity (>0)."""
    pos = np.asarray(severity_sample, dtype=float)
    pos = pos[pos > 0]
    if len(pos) == 0:
        raise ValueError("Sampel severity kosong/<=0.")
    logs = np.log(pos)
    return float(np.mean(logs)), float(np.std(logs) or 1.0)


def _safe_n_sims(frequency: float, n_sims: int) -> int:
    """Batasi n_sims agar ekspektasi total undian tidak melebihi pagar memori."""
    if frequency <= 0:
        return max(1, n_sims)
    max_sims = int(_MAX_DRAWS / frequency)
    return max(1, min(n_sims, max_sims))


def simulate_aggregate(
    frequency: float,
    severity_sample: np.ndarray | None = None,
    mu: float | None = None,
    sigma: float | None = None,
    empirical: bool = False,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = 42,
) -> np.ndarray:
    """Simulasikan distribusi kerugian agregat L (Monte Carlo).

    Tiap simulasi: N ~ Poisson(frequency), lalu L = sum dari N severity.

    Args:
        empirical: bila True, severity di-resample dari `severity_sample`
            (pendekatan historis/empiris); bila False memakai lognormal (mu,sigma).
    """
    rng = np.random.default_rng(seed)
    n_sims = _safe_n_sims(frequency, n_sims)
    counts = rng.poisson(frequency, size=n_sims)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(n_sims)

    if empirical:
        if severity_sample is None:
            raise ValueError("empirical=True memerlukan severity_sample.")
        pos = np.asarray(severity_sample, dtype=float)
        pos = pos[pos > 0]
        draws = rng.choice(pos, size=total, replace=True)
    else:
        if mu is None or sigma is None:
            mu, sigma = fit_severity_lognormal(severity_sample)
        draws = rng.lognormal(mean=mu, sigma=sigma, size=total)

    sim_index = np.repeat(np.arange(n_sims), counts)
    return np.bincount(sim_index, weights=draws, minlength=n_sims)


def value_at_risk(losses: np.ndarray, q: float = DEFAULT_VAR_Q) -> float:
    """VaR = persentil ke-q dari distribusi kerugian agregat."""
    return float(np.percentile(losses, q * 100.0))


def expected_loss(losses: np.ndarray) -> float:
    return float(np.mean(losses))


def actuarial_summary(
    frequency: float,
    severity_sample: np.ndarray,
    q: float = DEFAULT_VAR_Q,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = 42,
) -> dict:
    """Ringkas dua varian VaR aktuaria.

      * actuarial_montecarlo_var : severity parametrik (lognormal fit).
      * historical_var           : severity empiris (resample data historis).
      * historical_losses        : kerugian terealisasi (ekspektasi = freq x mean).
    """
    mu, sigma = fit_severity_lognormal(severity_sample)
    param = simulate_aggregate(
        frequency, mu=mu, sigma=sigma, empirical=False, n_sims=n_sims, seed=seed)
    emp = simulate_aggregate(
        frequency, severity_sample=severity_sample, empirical=True,
        n_sims=n_sims, seed=seed)

    pos = severity_sample[severity_sample > 0]
    realized = float(frequency * np.mean(pos))
    # Diagnostik risiko-model: ekor parametrik bisa jauh melampaui sampel empiris.
    sev_q = float(np.percentile(pos, q * 100.0))
    lognormal_sev_q = float(np.exp(mu + sigma * _z_quantile(q)))
    return {
        "frequency": float(frequency),
        "severity_mu": mu,
        "severity_sigma": sigma,
        "historical_losses": realized,
        "historical_var": value_at_risk(emp, q),
        "actuarial_montecarlo_var": value_at_risk(param, q),
        "expected_loss_montecarlo": expected_loss(param),
        "severity_empirical_q999": sev_q,
        "severity_lognormal_q999": lognormal_sev_q,
        "severity_max_observed": float(pos.max()),
        "_param_losses": param,
        "_empirical_losses": emp,
    }


def _z_quantile(q: float) -> float:
    """Kuantil normal baku z untuk probabilitas q (tanpa dependensi scipy)."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(q)


if __name__ == "__main__":
    from orm.data_sources import internal_frequency_severity, load_internal_losses

    internal = load_internal_losses()
    fs = internal_frequency_severity(internal)
    # Periode harian: frekuensi rata-rata kejadian per hari.
    freq_daily = fs["n_events"] / fs["span_days"]
    summary = actuarial_summary(freq_daily, fs["severity_sample"])

    print("=== Model aktuaria (basis harian) ===")
    print(f"  frekuensi harian (lambda) = {summary['frequency']:.1f} kejadian/hari")
    print(f"  severity lognormal: mu={summary['severity_mu']:.3f} "
          f"sigma={summary['severity_sigma']:.3f}")
    print(f"  kerugian terealisasi (E[L]) = {summary['historical_losses']:,.0f}")
    print(f"  Historical VaR (empiris) 99.9% = {summary['historical_var']:,.0f}")
    print(f"  Actuarial MC VaR (lognormal) 99.9% = "
          f"{summary['actuarial_montecarlo_var']:,.0f}")
