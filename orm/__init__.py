"""
orm — Operational Risk Management menurut Giudici (2009), Bab 12
"Applied Data Mining for Business and Industry".

Paket ini mengimplementasikan metodologi pengukuran risiko operasional dari buku:

  * categories   : kerangka Basel (8 business line x 7 event type = 56 kategori)
                   dan skala ordinal frekuensi/severity/kontrol.
  * data_sources : tiga aliran data (internal loss, expert opinion, external loss)
                   beserta proses scaling.
  * scorecard    : model scorecard self-assessment (median + indeks Gini ter-
                   normalisasi -> rating huruf A/AA/AAA dengan konvensi lampu lalu
                   lintas). Lihat Gambar 12.1.
  * actuarial    : model aktuaria L_j = sum X_ij; konvolusi frekuensi x severity
                   via Monte Carlo -> distribusi kerugian -> VaR (persentil 99.9).
  * integrated   : integrasi Bayesian tiga aliran data -> VaR terintegrasi, serta
                   Basic Indicator Approach (15% gross income). Lihat Gambar 12.2.
"""

__all__ = ["categories", "data_sources", "scorecard", "actuarial", "integrated"]
