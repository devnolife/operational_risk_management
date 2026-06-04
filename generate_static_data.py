"""
generate_static_data.py — Run all ORM models and save results to static/data.json.

Usage:
    python generate_static_data.py

Output:
    static/data.json — all pre-computed results for the static HTML dashboard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_PATH
from orm.data_sources import (
    generate_expert_opinions,
    generate_external_losses,
    internal_frequency_severity,
    load_internal_losses,
    scale_external_to_internal,
)
from orm.scorecard import (
    TRAFFIC_LIGHT_HEX,
    build_scorecard,
    self_assessment_total_loss,
)
from orm.actuarial import (
    fit_severity_lognormal,
    simulate_aggregate,
    value_at_risk,
)
from orm.integrated import run_comparison


def _ser(obj):
    """JSON-safe serialiser for numpy/pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Cannot serialise {type(obj)}")


def main():
    print("📦 Loading data …")

    # --- PaySim RAW metadata ---
    from src.data import find_raw_csv
    raw_csv = find_raw_csv()
    df_raw_sample = pd.read_csv(raw_csv, nrows=50)
    raw_cols = list(df_raw_sample.columns)
    raw_dtypes = {c: str(df_raw_sample[c].dtype) for c in raw_cols}

    # Count total rows in raw CSV (from header line count)
    df_raw_full = pd.read_csv(raw_csv)
    raw_total_rows = len(df_raw_full)
    raw_type_dist = df_raw_full["type"].value_counts().to_dict()
    raw_fraud_count = int(df_raw_full["isFraud"].sum())
    raw_flagged_count = int(df_raw_full["isFlaggedFraud"].sum())
    raw_n_unique = {c: int(df_raw_full[c].nunique()) for c in raw_cols}
    raw_nulls = {c: int(df_raw_full[c].isnull().sum()) for c in raw_cols}
    raw_preview = df_raw_full.head(30).to_dict(orient="records")
    del df_raw_full  # free memory

    # --- PaySim metadata ---
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PROCESSED_PATH)
    total_rows = int(pf.metadata.num_rows)
    all_cols = list(pf.schema_arrow.names)

    df_full = pd.read_parquet(PROCESSED_PATH, columns=["isFraud"])
    total_tx = int(len(df_full))
    total_fraud = int(df_full["isFraud"].sum())

    # fraud preview (first 50 rows, all columns)
    fraud_df = pd.read_parquet(PROCESSED_PATH)
    fraud_df = fraud_df[fraud_df["isFraud"] == 1].reset_index(drop=True)
    fraud_preview = fraud_df.head(50)

    # --- Internal losses ---
    losses = load_internal_losses()
    fs = internal_frequency_severity(losses)

    # --- Expert opinions ---
    expert_df = generate_expert_opinions(n_experts=8, seed=42)

    # --- External losses ---
    ext_raw = generate_external_losses(internal_severity=fs["severity_sample"])
    ext_scaled = scale_external_to_internal(ext_raw, fs["total_internal"])

    # --- Scorecard ---
    print("📋 Building scorecard …")
    sc = build_scorecard(expert_df)
    sa_total = self_assessment_total_loss(sc)

    sc_show = sc[[
        "business_line", "event_type",
        "frequency_rating", "frequency_color",
        "severity_rating", "severity_color",
        "control_rating", "control_color",
        "perceived_loss", "priority_score",
    ]].copy()

    # Heatmap pivot
    sc_pivot = sc.pivot_table(
        index="business_line", columns="event_type",
        values="perceived_loss", aggfunc="first",
    )

    # --- Run comparison (actuarial + integrated + BIA) ---
    print("📈 Running Monte Carlo (20 000 sims) …")
    res = run_comparison(q=0.999, n_sims=20_000)
    act = res["actuarial"]
    comp = res["comparison"]

    # Severity distribution for histogram (sample 2000 points from MC losses)
    mc_losses = act["_param_losses"]
    var_line = value_at_risk(mc_losses, 0.999)

    # Build histogram bins server-side
    counts, bin_edges = np.histogram(mc_losses, bins=80)
    hist_data = {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "var_line": float(var_line),
    }

    # --- VaR comparison bar chart data ---
    comp_sorted = comp.sort_values()
    comp_data = [
        {"method": str(k), "value": float(v)}
        for k, v in comp_sorted.items()
    ]

    # Focus area for executive summary
    focus_row = sc[
        (sc["business_line"] == "Payment and Settlement")
        & (sc["event_type"] == "External Fraud")
    ]
    focus_data = None
    if not focus_row.empty:
        r = focus_row.iloc[0]
        focus_data = {
            "freq_rating": str(r["frequency_rating"]),
            "sev_rating": str(r["severity_rating"]),
            "ctrl_rating": str(r["control_rating"]),
            "freq_color": TRAFFIC_LIGHT_HEX.get(r["frequency_color"], "#999"),
        }

    # --- Build final JSON ---
    data = {
        "raw": {
            "total_rows": raw_total_rows,
            "total_cols": len(raw_cols),
            "columns": raw_cols,
            "dtypes": raw_dtypes,
            "n_unique": raw_n_unique,
            "nulls": raw_nulls,
            "type_dist": raw_type_dist,
            "fraud_count": raw_fraud_count,
            "flagged_count": raw_flagged_count,
            "preview": raw_preview,
        },
        "paysim": {
            "total_tx": total_tx,
            "total_fraud": total_fraud,
            "total_rows": total_rows,
            "num_cols": len(all_cols),
            "columns": all_cols,
            "fraud_preview": fraud_preview.head(30).to_dict(orient="records"),
        },
        "internal": {
            "n_events": int(fs["n_events"]),
            "span_days": int(fs["span_days"]),
            "annual_frequency": float(fs["annual_frequency"]),
            "total_internal": float(fs["total_internal"]),
            "loss_preview": losses.head(20).to_dict(orient="records"),
            "severity_stats": {
                "mean": float(losses["amount"].mean()),
                "max": float(losses["amount"].max()),
                "median": float(losses["amount"].median()),
            },
        },
        "expert": {
            "n_experts": int(expert_df["expert"].nunique()),
            "n_rows": len(expert_df),
            "n_categories": int(expert_df.groupby(["business_line", "event_type"]).ngroups),
            "preview": expert_df.head(30).to_dict(orient="records"),
        },
        "external": {
            "n_rows": len(ext_raw),
            "raw": ext_raw[:30].tolist() if hasattr(ext_raw, "tolist") else list(ext_raw[:30]),
            "scaled": ext_scaled[:30].tolist() if hasattr(ext_scaled, "tolist") else list(ext_scaled[:30]),
            "median_raw": float(np.median(ext_raw)),
            "median_scaled": float(np.median(ext_scaled)),
        },
        "scorecard": {
            "total_categories": len(sc),
            "sa_total_loss": float(sa_total),
            "table": sc_show.to_dict(orient="records"),
            "top15": sc_show.head(15).to_dict(orient="records"),
            "heatmap": {
                "rows": sc_pivot.index.tolist(),
                "cols": sc_pivot.columns.tolist(),
                "values": sc_pivot.values.tolist(),
            },
            "traffic_light_hex": TRAFFIC_LIGHT_HEX,
        },
        "actuarial": {
            "expected_loss": float(act["expected_loss_montecarlo"]),
            "mc_var": float(act["actuarial_montecarlo_var"]),
            "hist_var": float(act["historical_var"]),
            "severity_sigma": float(act["severity_sigma"]),
            "severity_max": float(act["severity_max_observed"]),
            "histogram": hist_data,
        },
        "comparison": {
            "items": comp_data,
            "period_label": str(res["period_label"]),
            "self_assessment_point": float(res["self_assessment_point"]),
            "gross_income": float(res["gross_income"]),
            "bia": float(res["gross_income"] * 0.15),
        },
        "focus": focus_data,
        "meta": {
            "q": 0.999,
            "n_sims": 20_000,
            "n_experts": 8,
        },
    }

    out_path = ROOT / "static" / "data.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=_ser, ensure_ascii=False, indent=2)

    size_kb = out_path.stat().st_size / 1024
    print(f"✅ Saved to {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
