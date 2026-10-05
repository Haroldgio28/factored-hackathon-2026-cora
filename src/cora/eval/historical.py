"""Historical baseline B0 from the curated call-center data (task 6.8, REQ-41/47).

B0 is the pre-CORA human-agent operation measured straight from the dataset: first-contact
resolution (FCR) and average handle time (AHT) by contact reason. It is the "what the bank does
today" reference REQ-41 asks for, but it is NOT comparable 1:1 with CORA's SAR - B0 covers the full
range of human-handled contacts (including ones CORA never attempts), over a different population,
measured from historical records. Every number therefore carries that caveat and the REQ-47
`offline measurement` label; the report (6.9) must repeat it next to any B0 vs CORA figure.

Signal choice (owner EDA finding, rephrased for compliance):

- **FCR** = share of interactions with `was_resolved = true`, by `reason_category`. This is the
  genuine first-contact-resolution signal.
- **AHT** = mean `duration_seconds`, by `reason_category`.
- `was_escalated` is intentionally NOT used as an escalation-truth signal: in this dataset it is a
  near-flat ~10% synthetic artefact, so treating it as ground truth would be dishonest. B0 reports
  resolution and handle time only.

The computation is one bounded DuckDB `GROUP BY` aggregate through the `DataSource` (the
`quality/checks.py` pattern): only the per-category summary (a handful of rows) is pulled into
Python, never the fact table itself (host RAM ~4 GB).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cora.data.datasource import DataSource

__all__ = ["B0_LABEL", "B0_NOT_COMPARABLE_NOTE", "compute_b0"]

# The honest labels every B0 figure must carry (REQ-41/47). Kept verbatim so the report and the
# tests reference one string, not a paraphrase.
B0_LABEL = "offline measurement"
B0_NOT_COMPARABLE_NOTE = (
    "Historical human-agent baseline, not comparable 1:1 with CORA's SAR: different population, "
    "full range of human-handled contacts, measured from historical records."
)

_TABLE = "call_center_interactions"
_CATEGORY = "reason_category"


def compute_b0(source: DataSource) -> dict[str, object]:
    """Compute FCR and AHT by contact reason from the call-center interactions.

    Returns a JSON-serialisable dict carrying the per-category rows plus the honesty labels. Each
    row has the category, interaction count, FCR (resolved count / count), and AHT seconds (mean
    `duration_seconds` over the non-null rows). FCR is `None` when a category has no non-null
    `was_resolved`, and AHT is `None` when it has no non-null `duration_seconds` - undefined is
    reported as `None`, never 0.0.
    """
    con = source._connect()
    rel = source._source(_TABLE)
    # One bounded GROUP BY: resolved/total for FCR, mean duration for AHT, per category. The result
    # is one row per reason_category (a tiny table), so nothing large reaches Python.
    # The raw landing stores every column as VARCHAR (typing happens in the curated/contracts
    # layer), so was_resolved ('True'/'False') and duration_seconds ('603.0') are cast here.
    # TRY_CAST turns any malformed value into NULL rather than raising, matching the landing's
    # fail-soft typing; a NULL is then excluded by count()/avg() exactly like a missing value.
    sql = (  # noqa: S608 - no user input; table/columns are literals from the contract
        f"SELECT {_CATEGORY} AS category, "
        "count(*) AS n, "
        "count(*) FILTER (WHERE TRY_CAST(was_resolved AS BOOLEAN)) AS resolved, "
        "count(TRY_CAST(was_resolved AS BOOLEAN)) AS resolved_known, "
        "avg(TRY_CAST(duration_seconds AS DOUBLE)) AS aht_seconds "
        f"FROM {rel} "
        f"WHERE {_CATEGORY} IS NOT NULL "
        f"GROUP BY {_CATEGORY} ORDER BY {_CATEGORY}"
    )
    rows: list[dict[str, object]] = []
    for category, n, resolved, resolved_known, aht in con.execute(sql).fetchall():
        rows.append(
            {
                "category": category,
                "n": int(n),
                "fcr": (int(resolved) / int(resolved_known)) if resolved_known else None,
                "fcr_numerator": int(resolved),
                "fcr_denominator": int(resolved_known),
                "aht_seconds": float(aht) if aht is not None else None,
            }
        )
    return {
        "label": B0_LABEL,
        "note": B0_NOT_COMPARABLE_NOTE,
        "escalation_signal": "not used (was_escalated is a ~flat synthetic artefact, not ground truth)",
        "by_category": rows,
    }
