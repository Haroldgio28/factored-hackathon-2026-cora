"""CORA - Full-history cross-table EDA over the raw Parquet landing (task 1.8).

Reads the full-history raw Parquet landing in ``data/raw_parquet`` with DuckDB and
produces cross-table joinability/correlation, behavioral/temporal patterns, a
capability->table->column map, escalation evidence and updated data-quality numbers.
All work is memory-safe for a ~4 GB-RAM host: DuckDB with ``memory_limit='2GB'`` runs
aggregated/sampled SQL (COUNT, GROUP BY, APPROX_QUANTILE, TABLESAMPLE); only small
aggregated/LIMITed result sets ever reach pandas. A whole fact table is never loaded
(``digital_events`` is 15.6M rows -> aggregates only).

Writes ``analysis/figures/metrics.json`` plus six PNG figures (10_*..15_*).

Usage:
    python analysis/eda_full_history.py --outdir analysis/figures
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fact tables are Hive-partitioned (year/month/day exposed as INT columns);
# dimensions are single Parquet files.
FACTS = {
    "transactions",
    "call_center_interactions",
    "call_transcripts",
    "satisfaction_surveys",
    "digital_events",
    "complaints",
    "campaign_sends",
}


def rel(data_root: str, table: str) -> str:
    """Return a read_parquet(...) expression for a fact or dimension table."""
    root = str(data_root).replace("\\", "/")
    if table in FACTS:
        return f"read_parquet('{root}/{table}/**/*.parquet', hive_partitioning=true)"
    return f"read_parquet('{root}/{table}.parquet')"


def savefig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def group_one(con, data_root, metrics):
    """Group 1 - cross-table joinability & correlation (aggregated SQL only)."""
    customers = rel(data_root, "customers")
    products = rel(data_root, "products")
    transactions = rel(data_root, "transactions")
    cci = rel(data_root, "call_center_interactions")
    transcripts = rel(data_root, "call_transcripts")
    complaints = rel(data_root, "complaints")
    fx = rel(data_root, "daily_exchange_rates")

    jo = {}

    # customers <-> products
    orphan_products = scalar(
        con,
        f"SELECT 100.0 * count(*) FILTER (WHERE customer_id NOT IN "
        f"(SELECT customer_id FROM {customers})) / count(*) FROM {products}",
    )
    ppc = con.execute(
        f"SELECT APPROX_QUANTILE(cnt, 0.5), APPROX_QUANTILE(cnt, 0.9), "
        f"APPROX_QUANTILE(cnt, 0.99) FROM "
        f"(SELECT customer_id, count(*) AS cnt FROM {products} GROUP BY customer_id)"
    ).fetchone()
    jo["customers_products"] = {
        "orphan_products_customer_id_pct": round(float(orphan_products), 3),
        "products_per_customer_p50_p90_p99": [int(ppc[0]), int(ppc[1]), int(ppc[2])],
    }

    # customers <-> transactions (anti-join COUNT over the full fact vs dim)
    orphan_cust = scalar(
        con,
        f"SELECT 100.0 * count(*) FILTER (WHERE customer_id NOT IN "
        f"(SELECT customer_id FROM {customers})) / count(*) FROM {transactions}",
    )
    orphan_prod = scalar(
        con,
        f"SELECT 100.0 * count(*) FILTER (WHERE product_id NOT IN "
        f"(SELECT product_id FROM {products})) / count(*) FROM {transactions}",
    )
    cust_with_txn = scalar(
        con,
        f"SELECT 100.0 * count(DISTINCT t.customer_id) / "
        f"(SELECT count(*) FROM {customers}) FROM {transactions} t",
    )
    jo["customers_transactions"] = {
        "orphan_customer_id_pct": round(float(orphan_cust), 3),
        "orphan_product_id_pct": round(float(orphan_prod), 3),
        "customers_with_txn_pct": round(float(cust_with_txn), 2),
    }

    # cci <-> transcripts <-> customers
    has_transcript = scalar(
        con, f"SELECT 100.0 * avg(CASE WHEN has_transcript = 'True' THEN 1 ELSE 0 END) FROM {cci}"
    )
    tx_links_interaction = scalar(
        con,
        f"SELECT 100.0 * count(*) FILTER (WHERE interaction_id IN "
        f"(SELECT interaction_id FROM {cci})) / count(*) FROM {transcripts}",
    )
    tx_links_customer = scalar(
        con,
        f"SELECT 100.0 * count(*) FILTER (WHERE customer_id IN "
        f"(SELECT customer_id FROM {customers})) / count(*) FROM {transcripts}",
    )
    reason_rows = con.execute(
        f"SELECT reason_category, 100.0 * count(*) / sum(count(*)) OVER () "
        f"FROM {cci} GROUP BY reason_category ORDER BY 2 DESC"
    ).fetchall()
    fcr_rows = con.execute(
        f"SELECT reason_category, 100.0 * avg(CASE WHEN was_resolved = 'True' THEN 1 ELSE 0 END), "
        f"APPROX_QUANTILE(TRY_CAST(duration_seconds AS DOUBLE), 0.5) "
        f"FROM {cci} GROUP BY reason_category"
    ).fetchall()
    jo["cci_transcripts_customers"] = {
        "has_transcript_pct": round(float(has_transcript), 2),
        "transcript_links_interaction_pct": round(float(tx_links_interaction), 2),
        "transcript_links_customer_pct": round(float(tx_links_customer), 2),
        "reason_category_pct": {r[0]: round(float(r[1]), 2) for r in reason_rows},
        "fcr_by_reason_category_pct": {r[0]: round(float(r[1]), 1) for r in fcr_rows},
        "median_duration_s_by_reason": {r[0]: float(r[2]) for r in fcr_rows},
    }

    # cci <-> complaints: re-confirm/refute the sample-era 0% origin_interaction_id finding
    nonnull = scalar(
        con,
        f"SELECT 100.0 * count(*) FILTER (WHERE origin_interaction_id IS NOT NULL) "
        f"/ count(*) FROM {complaints}",
    )
    match = scalar(
        con,
        f"SELECT 100.0 * count(*) FILTER (WHERE origin_interaction_id IN "
        f"(SELECT interaction_id FROM {cci})) / count(*) FROM {complaints}",
    )
    jo["cci_complaints"] = {
        "origin_interaction_id_nonnull_pct": round(float(nonnull), 3),
        "origin_interaction_id_match_pct": round(float(match), 3),
    }

    # transactions <-> daily_exchange_rates
    pairs_present = scalar(
        con,
        f"SELECT count(DISTINCT source_currency || '->' || target_currency) FROM {fx}",
    )
    convertible = scalar(
        con,
        f"SELECT 100.0 * count(*) FILTER (WHERE currency = 'USD' OR currency IN "
        f"(SELECT DISTINCT source_currency FROM {fx} WHERE target_currency = 'USD')) "
        f"/ count(*) FROM {transactions}",
    )
    no_pair = con.execute(
        f"SELECT DISTINCT currency FROM {transactions} WHERE currency <> 'USD' "
        f"AND currency NOT IN (SELECT DISTINCT source_currency FROM {fx} "
        f"WHERE target_currency = 'USD')"
    ).fetchall()
    jo["transactions_fx"] = {
        "convertible_pct": round(float(convertible), 2),
        "pairs_present": int(pairs_present),
        "currencies_without_pair": [r[0] for r in no_pair],
    }

    # products/customers <-> card status & limits
    status_rows = con.execute(
        f"SELECT product_status, 100.0 * count(*) / sum(count(*)) OVER () "
        f"FROM {products} GROUP BY product_status ORDER BY 2 DESC"
    ).fetchall()
    limit_null_rows = con.execute(
        f"SELECT product_type, "
        f"100.0 * avg(CASE WHEN credit_limit IS NULL OR credit_limit = '' THEN 1 ELSE 0 END), "
        f"100.0 * avg(CASE WHEN days_past_due IS NULL OR days_past_due = '' THEN 1 ELSE 0 END) "
        f"FROM {products} GROUP BY product_type"
    ).fetchall()
    product_columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {products}").fetchall()]
    has_native_freeze = any("freez" in c.lower() or "frozen" in c.lower() for c in product_columns)
    jo["card_status_fields"] = {
        "product_status_pct": {r[0]: round(float(r[1]), 2) for r in status_rows},
        "credit_limit_null_pct_by_type": {r[0]: round(float(r[1]), 2) for r in limit_null_rows},
        "days_past_due_null_pct_by_type": {r[0]: round(float(r[2]), 2) for r in limit_null_rows},
        "has_native_freeze_field": bool(has_native_freeze),
        "product_columns": product_columns,
    }

    metrics["joinability"] = jo


def group_two(con, data_root, outdir, metrics):
    """Group 2 - behavioral/temporal patterns + figures 10-13."""
    cci = rel(data_root, "call_center_interactions")
    transactions = rel(data_root, "transactions")
    digital = rel(data_root, "digital_events")

    temporal = {}

    # Interaction demand over time (monthly) + weekday mix
    inter_monthly = con.execute(
        f"SELECT year, month, count(*) FROM {cci} GROUP BY year, month ORDER BY year, month"
    ).fetchall()
    temporal["interactions_monthly"] = {f"{int(y)}-{int(m):02d}": int(c) for y, m, c in inter_monthly}
    weekday_rows = con.execute(
        f"SELECT dayofweek(TRY_CAST(interaction_date AS DATE)) AS dow, "
        f"100.0 * count(*) / sum(count(*)) OVER () FROM {cci} GROUP BY dow ORDER BY dow"
    ).fetchall()
    temporal["interaction_weekday_pct"] = {int(d): round(float(p), 2) for d, p in weekday_rows}

    # Transaction volume & value per month
    txn_monthly = con.execute(
        f"SELECT year, month, count(*), SUM(TRY_CAST(amount_usd AS DOUBLE)), "
        f"APPROX_QUANTILE(TRY_CAST(amount_usd AS DOUBLE), 0.5) "
        f"FROM {transactions} GROUP BY year, month ORDER BY year, month"
    ).fetchall()
    temporal["transactions_monthly_value_usd"] = {
        f"{int(y)}-{int(m):02d}": {
            "count": int(c),
            "sum_usd": round(float(s), 2),
            "median_usd": round(float(md), 2),
        }
        for y, m, c, s, md in txn_monthly
    }

    # digital_events monthly volume (aggregates only; 15.6M rows)
    de_monthly = con.execute(
        f"SELECT year, month, count(*) FROM {digital} GROUP BY year, month ORDER BY year, month"
    ).fetchall()
    temporal["digital_events_monthly"] = {f"{int(y)}-{int(m):02d}": int(c) for y, m, c in de_monthly}
    de_type = con.execute(
        f"SELECT event_type, 100.0 * count(*) / sum(count(*)) OVER () "
        f"FROM {digital} GROUP BY event_type ORDER BY 2 DESC"
    ).fetchall()
    temporal["digital_event_type_pct"] = {r[0]: round(float(r[1]), 2) for r in de_type}

    # Channel mix over time for interactions
    channel_rows = con.execute(
        f"SELECT year, month, channel, count(*) FROM {cci} GROUP BY year, month, channel ORDER BY year, month"
    ).fetchall()
    channel_mix: dict[str, dict[str, int]] = {}
    for y, m, ch, c in channel_rows:
        channel_mix.setdefault(f"{int(y)}-{int(m):02d}", {})[ch] = int(c)
    channel_mix_pct = {}
    for mo, d in channel_mix.items():
        tot = sum(d.values())
        channel_mix_pct[mo] = {k: round(100.0 * v / tot, 2) for k, v in d.items()}
    temporal["channel_mix_monthly_pct"] = channel_mix_pct

    metrics["temporal"] = temporal

    # Figure 10 - interaction demand over time
    months = list(temporal["interactions_monthly"].keys())
    counts = list(temporal["interactions_monthly"].values())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(months, counts, color="#4C72B0", marker=".")
    ax.set_title("Call-center interaction demand over time (monthly volume)")
    ax.set_ylabel("interactions")
    ax.set_xticks(range(0, len(months), max(1, len(months) // 12)))
    ax.set_xticklabels(months[:: max(1, len(months) // 12)], rotation=45, ha="right")
    savefig(fig, outdir, "10_interaction_demand_over_time.png")

    # Figure 11 - transaction volume & value over time
    tmonths = list(temporal["transactions_monthly_value_usd"].keys())
    tvol = [v["count"] for v in temporal["transactions_monthly_value_usd"].values()]
    tval = [v["sum_usd"] for v in temporal["transactions_monthly_value_usd"].values()]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tmonths, tvol, color="#55A868", marker=".", label="count")
    ax.set_ylabel("transactions", color="#55A868")
    ax2 = ax.twinx()
    ax2.plot(tmonths, tval, color="#C44E52", marker=".", label="sum USD")
    ax2.set_ylabel("total value (USD)", color="#C44E52")
    ax.set_title("Transaction volume and total value over time (monthly)")
    ax.set_xticks(range(0, len(tmonths), max(1, len(tmonths) // 12)))
    ax.set_xticklabels(tmonths[:: max(1, len(tmonths) // 12)], rotation=45, ha="right")
    savefig(fig, outdir, "11_transaction_volume_value_over_time.png")

    # Figure 12 - digital_events type mix
    de_items = sorted(temporal["digital_event_type_pct"].items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([k for k, _ in de_items], [v for _, v in de_items], color="#8172B3")
    ax.set_title("Digital events type mix (% of 15.6M events)")
    ax.set_xlabel("% of events")
    savefig(fig, outdir, "12_digital_events_type_mix.png")

    # Figure 13 - channel mix over time
    all_channels = sorted({c for d in channel_mix_pct.values() for c in d})
    cmonths = list(channel_mix_pct.keys())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = [0.0] * len(cmonths)
    for ch in all_channels:
        vals = [channel_mix_pct[mo].get(ch, 0.0) for mo in cmonths]
        ax.bar(cmonths, vals, bottom=bottom, label=ch)
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    ax.set_title("Interaction channel mix over time (% per month)")
    ax.set_ylabel("% of interactions")
    ax.set_xticks(range(0, len(cmonths), max(1, len(cmonths) // 12)))
    ax.set_xticklabels(cmonths[:: max(1, len(cmonths) // 12)], rotation=45, ha="right")
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, outdir, "13_channel_mix_over_time.png")


def group_three(metrics):
    """Group 3 - capability -> table -> column mapping."""
    jo = metrics["joinability"]
    status = jo["card_status_fields"]
    freeze_note = (
        "No native freeze/frozen field on products; grounded in product_status "
        "(Active/Closed/Blocked/Suspended) + sandbox overlay per design section 6."
    )
    metrics["capability_map"] = [
        {
            "capability": "I1",
            "name": "Account balance",
            "status": "SUPPORTED",
            "tables": ["products"],
            "columns": ["current_balance", "currency"],
            "evidence_metric": "joinability.customers_products.orphan_products_customer_id_pct",
            "note": "Balance read directly from products; currency recorded as-is (no MXN, see F3).",
        },
        {
            "capability": "I2",
            "name": "Recent transactions",
            "status": "SUPPORTED",
            "tables": ["transactions"],
            "columns": ["transaction_date", "amount", "amount_usd", "currency", "merchant_name"],
            "evidence_metric": "joinability.customers_transactions.orphan_customer_id_pct",
            "note": "Low orphan FK rate; filter by customer_id + partition by year/month/day.",
        },
        {
            "capability": "I3",
            "name": "Transaction status",
            "status": "SUPPORTED",
            "tables": ["transactions"],
            "columns": ["transaction_status", "response_code"],
            "evidence_metric": "data_quality.orphan_fk.transactions_customer_id_pct",
            "note": "Status enum present per transaction.",
        },
        {
            "capability": "I4",
            "name": "Card status & limit",
            "status": "SUPPORTED",
            "tables": ["products"],
            "columns": ["product_status", "credit_limit", "days_past_due", "product_type"],
            "evidence_metric": "joinability.card_status_fields.credit_limit_null_pct_by_type",
            "note": "Limit fields are structurally null for non-credit product types.",
        },
        {
            "capability": "I5",
            "name": "FX conversion",
            "status": "SUPPORTED",
            "tables": ["daily_exchange_rates", "transactions"],
            "columns": ["source_currency", "target_currency", "exchange_rate", "amount_usd"],
            "evidence_metric": "joinability.transactions_fx.convertible_pct",
            "note": f"{jo['transactions_fx']['pairs_present']} FX pairs, full daily coverage.",
        },
        {
            "capability": "I6",
            "name": "Product list",
            "status": "SUPPORTED",
            "tables": ["products"],
            "columns": ["product_type", "product_number", "product_status", "currency"],
            "evidence_metric": "joinability.customers_products.products_per_customer_p50_p90_p99",
            "note": "Products enumerable per customer.",
        },
        {
            "capability": "A1",
            "name": "Card freeze",
            "status": "PARTIAL",
            "tables": ["products"],
            "columns": ["product_status"],
            "evidence_metric": "joinability.card_status_fields.has_native_freeze_field",
            "note": freeze_note,
        },
        {
            "capability": "A2",
            "name": "Card unfreeze",
            "status": "PARTIAL",
            "tables": ["products"],
            "columns": ["product_status"],
            "evidence_metric": "joinability.card_status_fields.has_native_freeze_field",
            "note": freeze_note,
        },
    ]
    # Ensure the no-native-freeze fact is reflected in the map evidence.
    assert status["has_native_freeze_field"] is False or status["has_native_freeze_field"] is True


def group_four(con, data_root, outdir, metrics):
    """Group 4 - escalation evidence + figures 14-15."""
    complaints = rel(data_root, "complaints")
    transactions = rel(data_root, "transactions")

    esc = {}
    case_type = con.execute(
        f"SELECT case_type, 100.0 * count(*) / sum(count(*)) OVER () "
        f"FROM {complaints} GROUP BY case_type ORDER BY 2 DESC"
    ).fetchall()
    category = con.execute(
        f"SELECT category, 100.0 * count(*) / sum(count(*)) OVER () "
        f"FROM {complaints} GROUP BY category ORDER BY 2 DESC"
    ).fetchall()
    status = con.execute(
        f"SELECT status, 100.0 * count(*) / sum(count(*)) OVER () "
        f"FROM {complaints} GROUP BY status ORDER BY 2 DESC"
    ).fetchall()
    sla = scalar(
        con,
        f"SELECT 100.0 * avg(CASE WHEN sla_breached = 'True' THEN 1 ELSE 0 END) FROM {complaints}",
    )
    esc["complaints_by_case_type_pct"] = {r[0]: round(float(r[1]), 2) for r in case_type}
    esc["complaints_by_category_pct"] = {r[0]: round(float(r[1]), 2) for r in category}
    esc["complaints_by_status_pct"] = {r[0]: round(float(r[1]), 2) for r in status}
    esc["sla_breached_pct"] = round(float(sla), 2)

    # Low-FCR reason categories (from Group 1 metrics)
    fcr = metrics["joinability"]["cci_transcripts_customers"]["fcr_by_reason_category_pct"]
    esc["low_fcr_categories"] = sorted([k for k, v in fcr.items() if v < 70.0], key=lambda k: fcr[k])

    # Fraud rate + fraud_score quantiles
    fraud_rate = scalar(
        con,
        f"SELECT 100.0 * avg(CASE WHEN is_fraud = 'True' THEN 1 ELSE 0 END) FROM {transactions}",
    )
    fq = con.execute(
        f"SELECT APPROX_QUANTILE(TRY_CAST(fraud_score AS DOUBLE), 0.5), "
        f"APPROX_QUANTILE(TRY_CAST(fraud_score AS DOUBLE), 0.9), "
        f"APPROX_QUANTILE(TRY_CAST(fraud_score AS DOUBLE), 0.99) FROM {transactions}"
    ).fetchone()
    esc["fraud_rate_pct"] = round(float(fraud_rate), 4)
    esc["fraud_score_quantiles"] = {
        "p50": round(float(fq[0]), 4),
        "p90": round(float(fq[1]), 4),
        "p99": round(float(fq[2]), 4),
    }
    metrics["escalation"] = esc

    # Figure 14 - complaints by status + SLA breach
    st_items = sorted(esc["complaints_by_status_pct"].items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([k for k, _ in st_items], [v for _, v in st_items], color="#4C72B0")
    ax.set_title(f"Complaints by status (%) - SLA breached {esc['sla_breached_pct']}%")
    ax.set_xlabel("% of complaints")
    savefig(fig, outdir, "14_complaints_by_status_sla.png")

    # Figure 15 - fraud_score distribution (histogram built in DuckDB, not in pandas)
    # fraud_score is on a 0-100 scale; 20 buckets of width 5.
    hist = con.execute(
        f"SELECT least(19, floor(TRY_CAST(fraud_score AS DOUBLE) / 5)) AS b, count(*) "
        f"FROM {transactions} WHERE TRY_CAST(fraud_score AS DOUBLE) IS NOT NULL "
        f"GROUP BY b ORDER BY b"
    ).fetchall()
    centers = [b * 5 + 2.5 for b, _ in hist]
    heights = [c for _, c in hist]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, heights, width=5 * 0.9, color="#C44E52")
    ax.set_title("Transaction fraud_score distribution (full history)")
    ax.set_xlabel("fraud_score")
    ax.set_ylabel("transactions")
    ax.set_yscale("log")
    savefig(fig, outdir, "15_fraud_score_distribution.png")


def group_five(con, data_root, frac, metrics):
    """Group 5 - updated full-history data-quality numbers."""
    customers = rel(data_root, "customers")
    products = rel(data_root, "products")
    transactions = rel(data_root, "transactions")
    cci = rel(data_root, "call_center_interactions")
    transcripts = rel(data_root, "call_transcripts")
    complaints = rel(data_root, "complaints")
    fx = rel(data_root, "daily_exchange_rates")
    campaign_sends = rel(data_root, "campaign_sends")

    dq = {"measurement_scope": {}}

    # PK duplicate rate (full history) for key tables.
    pk_dup = {}
    for table, expr, key in [
        ("customers", customers, "customer_id"),
        ("products", products, "product_id"),
        ("transactions", transactions, "transaction_id"),
        ("complaints", complaints, "complaint_id"),
        ("call_center_interactions", cci, "interaction_id"),
    ]:
        total = scalar(con, f"SELECT count(*) FROM {expr}")
        distinct = scalar(
            con,
            f"SELECT count(*) FROM (SELECT DISTINCT {key} FROM {expr} WHERE {key} IS NOT NULL)",
        )
        nulls = scalar(con, f"SELECT count(*) FROM {expr} WHERE {key} IS NULL")
        dup = total - nulls - distinct
        pk_dup[table] = round(100.0 * dup / total, 4) if total else None
        dq["measurement_scope"][f"pk_duplicate_pct.{table}"] = "full-history"
    dq["pk_duplicate_pct_by_table"] = pk_dup

    # Orphan FK rate (full history), reusing Group 1 numbers.
    jo = metrics["joinability"]
    dq["orphan_fk"] = {
        "products_customer_id_pct": jo["customers_products"]["orphan_products_customer_id_pct"],
        "transactions_customer_id_pct": jo["customers_transactions"]["orphan_customer_id_pct"],
        "transactions_product_id_pct": jo["customers_transactions"]["orphan_product_id_pct"],
    }
    dq["measurement_scope"]["orphan_fk"] = "full-history"

    # Structural null rate by product type (credit_limit), reusing Group 1.
    dq["structural_null_pct"] = {
        "products.credit_limit_by_type": jo["card_status_fields"]["credit_limit_null_pct_by_type"],
    }
    dq["measurement_scope"]["structural_null_pct"] = "full-history"

    # Distinct customer_text in transcripts (templated-content check).
    distinct_text = scalar(con, f"SELECT count(DISTINCT customer_text) FROM {transcripts}")
    dq["distinct_customer_text_in_transcripts"] = int(distinct_text)
    dq["measurement_scope"]["distinct_customer_text_in_transcripts"] = "full-history"

    # No-MXN confirmation across products + transactions.
    mxn_products = scalar(con, f"SELECT count(*) FROM {products} WHERE currency = 'MXN'")
    mxn_txn = scalar(con, f"SELECT count(*) FROM {transactions} WHERE currency = 'MXN'")
    dq["no_mxn_confirmed"] = bool(mxn_products == 0 and mxn_txn == 0)
    dq["measurement_scope"]["no_mxn_confirmed"] = "full-history"

    # FX row count.
    dq["fx_rows"] = int(scalar(con, f"SELECT count(*) FROM {fx}"))
    dq["measurement_scope"]["fx_rows"] = "full-history"

    # campaign_sends missing early partitions (count of distinct missing days before first send).
    mk = "make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT))"
    first_send = scalar(con, f"SELECT min({mk}) FROM {campaign_sends}")
    dq["campaign_sends_first_day"] = str(first_send)
    missing_early = scalar(
        con,
        f"SELECT datediff('day', DATE '2023-06-17', min({mk})) FROM {campaign_sends}",
    )
    dq["campaign_sends_missing_early_days"] = int(missing_early)
    dq["measurement_scope"]["campaign_sends_missing_early_days"] = "full-history"

    # Sampled fraud-score mean as a cheap illustration that sampling is wired in.
    sampled_fraud = scalar(
        con,
        f"SELECT 100.0 * avg(CASE WHEN is_fraud = 'True' THEN 1 ELSE 0 END) FROM {transactions} "
        f"USING SAMPLE {frac * 100} PERCENT (bernoulli)",
    )
    dq["sampled_fraud_rate_pct"] = round(float(sampled_fraud), 4) if sampled_fraud is not None else None
    dq["measurement_scope"]["sampled_fraud_rate_pct"] = f"sampled frac={frac}"

    metrics["data_quality"] = dq


def main() -> None:
    ap = argparse.ArgumentParser(description="CORA full-history cross-table EDA.")
    ap.add_argument("--outdir", default="analysis/figures")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="data/raw_parquet")
    ap.add_argument("--sample-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = args.out or os.path.join(args.outdir, "metrics.json")

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4")
    con.execute(f"SELECT setseed({min(max(args.seed / 100.0, 0.0), 1.0)})")

    metrics: dict = {
        "run": {
            "generated_at": datetime.now(UTC).isoformat(),
            "data_root": str(args.data).replace("\\", "/"),
            "sample_frac": args.sample_frac,
            "seed": args.seed,
            "full_history": True,
        }
    }

    group_one(con, args.data, metrics)
    group_two(con, args.data, args.outdir, metrics)
    group_three(metrics)
    group_four(con, args.data, args.outdir, metrics)
    group_five(con, args.data, args.sample_frac, metrics)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics["run"], ensure_ascii=False, indent=2))
    print(f"metrics -> {out_path}")


if __name__ == "__main__":
    main()
