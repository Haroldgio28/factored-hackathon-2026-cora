"""CORA - Exploratory data analysis of the LATAM Bank sample.

Reads a local multi-day sample of call_center_interactions and complaints,
produces demand/resolution/channel charts and a JSON metrics summary used as
evidence for workflow selection. Charts/metrics are written to --outdir.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(base: str, sub: str):
    files = glob.glob(os.path.join(base, sub, "**", "*.csv"), recursive=True)
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(frames, ignore_index=True), len(files)


def savefig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=os.path.join(os.environ.get("KIROCREW_SCRATCH", "."), "cora_sample"))
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cci, ncci = load(args.sample, "cci")
    cmp, ncmp = load(args.sample, "complaints")

    metrics = {
        "sample": {
            "call_center_interactions_rows": int(len(cci)),
            "call_center_interactions_days": ncci,
            "complaints_rows": int(len(cmp)),
            "complaints_days": ncmp,
            "date_min": str(cci["interaction_date"].min()),
            "date_max": str(cci["interaction_date"].max()),
        }
    }

    # 1. Demand by reason_category
    demand = cci["reason_category"].value_counts()
    metrics["demand_by_reason_category_pct"] = (demand / len(cci) * 100).round(2).to_dict()
    fig, ax = plt.subplots(figsize=(7, 4))
    (demand / len(cci) * 100).sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("Call center demand by reason category (% of interactions)")
    ax.set_xlabel("% of interactions")
    savefig(fig, args.outdir, "01_demand_by_reason_category.png")

    # 2. FCR and escalation by reason_category
    fcr = (cci.groupby("reason_category")["was_resolved"].mean() * 100).round(1)
    esc = (cci.groupby("reason_category")["was_escalated"].mean() * 100).round(1)
    metrics["fcr_by_reason_category_pct"] = fcr.to_dict()
    metrics["escalation_by_reason_category_pct"] = esc.to_dict()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = fcr.sort_values().index
    x = range(len(idx))
    ax.bar([i - 0.2 for i in x], fcr[idx], width=0.4, label="FCR (resolved) %", color="#55A868")
    ax.bar([i + 0.2 for i in x], esc[idx], width=0.4, label="Escalation %", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(idx, rotation=20)
    ax.set_title("First-contact resolution vs escalation by reason category")
    ax.set_ylabel("%")
    ax.legend()
    savefig(fig, args.outdir, "02_fcr_vs_escalation.png")

    # 3. Channel mix
    ch = (cci["channel"].value_counts(normalize=True) * 100).round(1)
    metrics["channel_mix_pct"] = ch.to_dict()
    fig, ax = plt.subplots(figsize=(6, 4))
    ch.sort_values().plot.barh(ax=ax, color="#8172B3")
    ax.set_title("Interaction channel mix (%)")
    ax.set_xlabel("%")
    savefig(fig, args.outdir, "03_channel_mix.png")

    # 4. Accent / language proxy
    acc = (cci["customer_detected_accent"].value_counts(dropna=False, normalize=True) * 100).round(1)
    metrics["customer_accent_pct"] = {str(k): float(v) for k, v in acc.items()}
    fig, ax = plt.subplots(figsize=(6, 4))
    acc.sort_values().plot.barh(ax=ax, color="#CCB974")
    ax.set_title("Customer detected accent (% ; NaN = undetected)")
    ax.set_xlabel("%")
    savefig(fig, args.outdir, "04_customer_accent.png")

    # 5. Handling-time proxy by category (median duration)
    dur = cci.groupby("reason_category")["duration_seconds"].median().round(0).sort_values()
    metrics["median_duration_seconds_by_reason"] = dur.to_dict()

    # 6. Complaints structure
    case_type_share = cmp["case_type"].value_counts(normalize=True) * 100
    metrics["complaints_case_type_pct"] = case_type_share.round(1).to_dict()
    metrics["complaints_status_pct"] = (cmp["status"].value_counts(normalize=True) * 100).round(1).to_dict()
    origin_populated = float(cmp["origin_interaction_id"].notna().mean() * 100)
    metrics["complaints_origin_interaction_id_populated_pct"] = round(origin_populated, 2)
    metrics["complaints_sla_breached_pct"] = round(float(cmp["sla_breached"].mean() * 100), 2)

    # Automatable volume estimate: Transaccional + Producto share with high FCR
    auto_cats = [c for c in ["Transaccional", "Producto"] if c in demand.index]
    auto_share = float(demand[auto_cats].sum() / len(cci) * 100)
    metrics["high_fcr_automation_candidate_share_pct"] = round(auto_share, 2)
    metrics["high_fcr_automation_candidate_categories"] = auto_cats

    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
