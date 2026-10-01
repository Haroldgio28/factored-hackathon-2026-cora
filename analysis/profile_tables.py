"""CORA - Profile all 13 LATAM Bank tables against the data dictionary.

Dimension/reference tables are read in full; fact tables from a local sample
(see documentation/reports/data_profile.md for the sample windows). For every table it
reports rows, schema drift vs the dictionary, null rates, PK duplicates, key enums,
date ranges and orphan foreign keys, and writes a markdown report + JSON metrics.

Usage:
    python analysis/profile_tables.py --sample <sample_dir> --out documentation/reports
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

# Expected columns per the data dictionary v1.0.0 (order not enforced).
DICTIONARY = {
    "customers": "customer_id document_number document_type first_name last_name date_of_birth gender email mobile_phone landline_phone address city state country postal_code detected_accent segment credit_score estimated_monthly_income occupation marital_status education_level registration_date registration_branch_id customer_status last_updated accepts_marketing",
    "products": "product_id customer_id product_type product_number currency current_balance credit_limit interest_rate opening_date expiration_date opening_branch_id product_status opening_channel has_linked_app days_past_due last_transaction_date last_updated",
    "branches": "branch_id branch_code branch_name branch_type address city state country postal_code geographic_zone phone email opening_time closing_time has_atms atm_count has_teller_windows teller_window_count latitude longitude branch_opening_date branch_status",
    "service_agents": "agent_id employee_code first_name last_name email phone native_accent country_of_origin assigned_branch_id agent_type experience_level languages specialty hire_date avg_csat total_monthly_interactions agent_status work_shift",
    "marketing_campaigns": "campaign_id campaign_name description campaign_type campaign_objective promoted_product target_segment target_country start_date end_date budget campaign_status expected_conversion_rate",
    "daily_exchange_rates": "date source_currency target_currency exchange_rate buy_rate sell_rate source",
    "transactions": "transaction_id transaction_date process_date product_id customer_id transaction_type transaction_category amount currency amount_usd channel branch_id merchant_name merchant_category transaction_country transaction_city transaction_status response_code is_fraud fraud_score latitude longitude",
    "call_center_interactions": "interaction_id interaction_date process_date customer_id agent_id interaction_type channel contact_reason reason_category duration_seconds wait_time_seconds was_resolved requires_followup detected_sentiment sentiment_score customer_detected_accent agent_used_accent was_escalated mentioned_products has_transcript has_recording",
    "call_transcripts": "transcript_id interaction_id process_date customer_id agent_id full_text customer_text agent_text detected_language detected_accent accent_confidence detected_keywords mentioned_entities detected_intents main_topics transcription_model audio_quality duration_seconds",
    "satisfaction_surveys": "survey_id survey_date process_date interaction_id customer_id agent_id survey_type send_channel main_score nps_category question_1_text question_1_response question_2_text question_2_response question_3_text question_3_response open_comments comment_sentiment response_time_hours campaign_response_rate",
    "digital_events": "event_id event_date process_date customer_id session_id event_type event_category channel platform browser app_version page_url page_title action element_id product_id event_value duration_seconds ip_address ip_country ip_city is_mobile referrer utm_source utm_medium utm_campaign",
    "complaints": "complaint_id creation_date process_date customer_id case_type category subcategory reception_channel affected_product_id related_branch_id origin_interaction_id description claimed_amount currency priority status assigned_agent_id assignment_date first_response_date resolution_date closing_date sla_breached resolution_days resolution compensation_granted resolution_satisfaction is_repeat_complainer",
    "campaign_sends": "send_id send_date process_date campaign_id customer_id send_channel template_used subject send_status was_delivered was_opened open_date was_clicked click_date click_count had_conversion conversion_date conversion_value open_device open_country failure_reason send_cost",
}
PK = {
    "customers": ["customer_id"], "products": ["product_id"], "branches": ["branch_id"],
    "service_agents": ["agent_id"], "marketing_campaigns": ["campaign_id"],
    "daily_exchange_rates": ["date", "source_currency", "target_currency"],
    "transactions": ["transaction_id"], "call_center_interactions": ["interaction_id"],
    "call_transcripts": ["transcript_id"], "satisfaction_surveys": ["survey_id"],
    "digital_events": ["event_id"], "complaints": ["complaint_id"], "campaign_sends": ["send_id"],
}
DATE_COL = {
    "transactions": "transaction_date", "call_center_interactions": "interaction_date",
    "call_transcripts": "process_date", "satisfaction_surveys": "survey_date",
    "digital_events": "event_date", "complaints": "creation_date", "campaign_sends": "send_date",
    "daily_exchange_rates": "date", "customers": "registration_date", "products": "opening_date",
}
ENUMS = {
    "customers": ["country", "segment", "customer_status", "document_type", "detected_accent"],
    "products": ["product_type", "currency", "product_status"],
    "transactions": ["transaction_type", "transaction_status", "currency", "channel", "is_fraud"],
    "call_transcripts": ["detected_language", "detected_accent", "transcription_model", "audio_quality"],
    "satisfaction_surveys": ["survey_type", "nps_category", "comment_sentiment"],
    "digital_events": ["event_type", "event_category", "channel"],
    "campaign_sends": ["send_channel", "send_status"],
    "daily_exchange_rates": ["source_currency", "target_currency"],
    "service_agents": ["native_accent", "agent_type", "agent_status"],
    "branches": ["country", "branch_type", "branch_status"],
    "marketing_campaigns": ["campaign_type", "campaign_objective", "campaign_status"],
}
SAMPLE_DIR = {"call_center_interactions": "cci"}
DIMENSIONS = ["customers", "products", "branches", "service_agents", "marketing_campaigns", "daily_exchange_rates"]


def load(sample: str, table: str) -> tuple[pd.DataFrame, int]:
    if table in DIMENSIONS:
        return pd.read_csv(os.path.join(sample, "dim", f"{table}.csv"), low_memory=False), 1
    files = glob.glob(os.path.join(sample, SAMPLE_DIR.get(table, table), "**", "*.csv"), recursive=True)
    return pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True), len(files)


def profile(df: pd.DataFrame, table: str, files: int, ids: dict) -> dict:
    expected = set(DICTIONARY[table].split())
    actual = set(df.columns)
    pk = PK[table]
    null_pct = (df.isna().mean() * 100).round(2)
    res = {
        "rows": int(len(df)), "files": files,
        "missing_vs_dictionary": sorted(expected - actual),
        "extra_vs_dictionary": sorted(actual - expected),
        "pk_duplicate_pct": round(float(df.duplicated(subset=pk).mean() * 100), 3) if set(pk) <= actual else None,
        "full_row_duplicate_pct": round(float(df.duplicated().mean() * 100), 3),
        "top_null_columns_pct": null_pct[null_pct > 0].sort_values(ascending=False).head(8).to_dict(),
        "enums": {c: df[c].astype(str).value_counts().head(10).to_dict() for c in ENUMS.get(table, []) if c in actual},
    }
    dc = DATE_COL.get(table)
    if dc and dc in actual:
        d = pd.to_datetime(df[dc], errors="coerce")
        res["date_range"] = [str(d.min()), str(d.max())]
        res["unparseable_dates_pct"] = round(float(d.isna().mean() * 100), 3)
    for fk, ref in (("customer_id", "customers"), ("product_id", "products"), ("agent_id", "service_agents")):
        if fk in actual and table != ref and ref in ids:
            vals = df[fk].dropna()
            res[f"orphan_{fk}_pct"] = round(float((~vals.isin(ids[ref])).mean() * 100), 3) if len(vals) else None
    return res


def to_markdown(results: dict, total_bytes: dict) -> str:
    out = ["# Data Profile - all 13 tables", "",
           "Generated by `analysis/profile_tables.py`. Dimensions read in full; facts from the local sample (see `files`).",
           "Evidence type: **offline measurement on the organizer's synthetic data**.", ""]
    out += ["## Summary", "", "| Table | Rows profiled | Files | Missing vs dict | Extra vs dict | PK dup % | Row dup % | Date range |",
            "|---|---|---|---|---|---|---|---|"]
    for t, r in results.items():
        dr = " -> ".join(x[:10] for x in r.get("date_range", ["", ""]))
        out.append(f"| `{t}` | {r['rows']:,} | {r['files']} | {', '.join(r['missing_vs_dictionary']) or '-'} | "
                   f"{', '.join(r['extra_vs_dictionary']) or '-'} | {r['pk_duplicate_pct']} | {r['full_row_duplicate_pct']} | {dr} |")
    out += ["", "## Per-table detail", ""]
    for t, r in results.items():
        out += [f"### `{t}`", "", "```json", json.dumps({k: v for k, v in r.items() if k not in ('rows', 'files')},
                                                          ensure_ascii=False, indent=2, default=str), "```", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids, results = {}, {}
    for t in DIMENSIONS:
        df, n = load(args.sample, t)
        if t in ("customers", "products", "service_agents"):
            ids[t] = set(df[PK[t][0]].dropna())
        results[t] = profile(df, t, n, ids)
        del df
    for t in ["transactions", "call_center_interactions", "call_transcripts", "satisfaction_surveys",
              "digital_events", "complaints", "campaign_sends"]:
        df, n = load(args.sample, t)
        results[t] = profile(df, t, n, ids)
        del df

    with open(os.path.join(args.out, "data_profile.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(args.out, "data_profile.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(results, {}))
    print(json.dumps(results, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
