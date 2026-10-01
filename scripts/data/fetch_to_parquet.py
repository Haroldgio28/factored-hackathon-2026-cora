"""Fetch the LATAM Bank dataset from the organizer S3 bucket and land it as Parquet.

Raw landing zone (design §7): a lossless, immutable Parquet copy of every source CSV.
- All columns are stored as VARCHAR (``all_varchar``) so no value is altered by type
  inference; typing and contracts are applied later in the curated layer (tasks 1.2-1.4).
- Partition layout mirrors the source: ``<table>/year=YYYY/month=MM/day=DD/<table>_YYYYMMDD.parquet``;
  dimension/reference tables land as ``<table>.parquet``.
- Every file is verified (row and column counts equal to the CSV) before its CSV is deleted.
- A lineage manifest (source key, CSV bytes, sha256, rows, columns, parquet bytes) is written per table.

Usage:
    python scripts/data/fetch_to_parquet.py --dest data/raw_parquet --staging <tmp_dir> [--tables ...]
Credentials: AWS profile from CORA_DATATHON_PROFILE (default ``cora-datathon``); nothing is stored in code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

BUCKET = os.getenv("CORA_DATATHON_BUCKET", "factored-datathon-2026-s3-157725502942-us-east-2-an")
REGION = os.getenv("CORA_DATATHON_REGION", "us-east-2")
PROFILE = os.getenv("CORA_DATATHON_PROFILE", "cora-datathon")
DIMENSIONS = [
    "customers",
    "products",
    "branches",
    "service_agents",
    "marketing_campaigns",
    "daily_exchange_rates",
]
FACTS = [
    "transactions",
    "call_center_interactions",
    "call_transcripts",
    "satisfaction_surveys",
    "digital_events",
    "complaints",
    "campaign_sends",
]


def sync(src: str, dst: Path, single_file: bool) -> None:
    cmd = [
        "aws",
        "s3",
        "cp" if single_file else "sync",
        src,
        str(dst),
        "--profile",
        PROFILE,
        "--region",
        REGION,
        "--only-show-errors",
    ]
    subprocess.run(cmd, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(csv: Path, parquet: Path, source_key: str) -> dict:
    parquet.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='512MB'; SET threads=2")
    c, p = str(csv).replace("\\", "/"), str(parquet).replace("\\", "/")
    # all_varchar makes type inference irrelevant; the default sniffer sample only detects the dialect.
    rel = f"read_csv('{c}', all_varchar=true, header=true)"
    csv_rows = con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0]
    csv_cols = len(con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall())
    con.execute(f"COPY (SELECT * FROM {rel}) TO '{p}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    pq_rows = con.execute(f"SELECT count(*) FROM '{p}'").fetchone()[0]
    pq_cols = len(con.execute(f"DESCRIBE SELECT * FROM '{p}'").fetchall())
    con.close()
    if (csv_rows, csv_cols) != (pq_rows, pq_cols):
        raise ValueError(
            f"verification failed for {source_key}: csv={csv_rows}x{csv_cols} parquet={pq_rows}x{pq_cols}"
        )
    return {
        "source_key": source_key,
        "csv_bytes": csv.stat().st_size,
        "csv_sha256": sha256(csv),
        "rows": pq_rows,
        "columns": pq_cols,
        "parquet_bytes": parquet.stat().st_size,
    }


def land_table(table: str, dest: Path, staging: Path, workers: int, do_sync: bool = True) -> dict:
    t0 = time.time()
    stage = staging / table
    stage.mkdir(parents=True, exist_ok=True)
    resumed = []
    if table in DIMENSIONS:
        if do_sync:
            sync(f"s3://{BUCKET}/data/{table}.csv", stage / f"{table}.csv", single_file=True)
        jobs = [(stage / f"{table}.csv", dest / f"{table}.parquet", f"data/{table}.csv")]
    else:
        if do_sync:
            sync(f"s3://{BUCKET}/data/{table}/", stage, single_file=False)
        jobs = []
        for csv in sorted(stage.rglob("*.csv")):
            rel = csv.relative_to(stage)
            jobs.append((csv, dest / table / rel.with_suffix(".parquet"), f"data/{table}/{rel.as_posix()}"))
        # Resume: Parquet whose CSV was already verified and deleted in an interrupted run.
        pending = {pq for _, pq, _ in jobs}
        for pq in sorted((dest / table).rglob("*.parquet")) if (dest / table).exists() else []:
            if pq not in pending:
                con = duckdb.connect()
                p = str(pq).replace("\\", "/")
                rows = con.execute(f"SELECT count(*) FROM '{p}'").fetchone()[0]
                cols = len(con.execute(f"DESCRIBE SELECT * FROM '{p}'").fetchall())
                con.close()
                key = f"data/{table}/{pq.relative_to(dest / table).with_suffix('.csv').as_posix()}"
                resumed.append(
                    {
                        "source_key": key,
                        "csv_bytes": None,
                        "csv_sha256": None,
                        "rows": rows,
                        "columns": cols,
                        "parquet_bytes": pq.stat().st_size,
                        "note": "landed and verified in an interrupted earlier run; CSV hash not retained",
                    }
                )

    entries, failures = list(resumed), []

    def work(job):
        csv, pq, key = job
        try:
            entry = convert(csv, pq, key)
            csv.unlink()  # delete only after successful verification
            return entry, None
        except Exception as exc:  # keep the CSV for inspection
            return None, f"{key}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for entry, err in pool.map(work, jobs):
            (entries.append(entry) if entry else failures.append(err))

    manifest = {
        "table": table,
        "source": f"s3://{BUCKET}/data/",
        "landing": "raw (lossless, all VARCHAR)",
        "pipeline_version": "fetch_to_parquet/1.0",
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": len(entries),
        "failures": failures,
        "rows": sum(e["rows"] for e in entries),
        "csv_bytes": sum(e["csv_bytes"] or 0 for e in entries),
        "resumed_files": len(resumed),
        "parquet_bytes": sum(e["parquet_bytes"] for e in entries),
        "elapsed_s": round(time.time() - t0, 1),
        "entries": entries,
    }
    (dest / "_manifests").mkdir(parents=True, exist_ok=True)
    (dest / "_manifests" / f"{table}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if not failures:
        shutil.rmtree(stage, ignore_errors=True)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--staging", required=True, type=Path)
    ap.add_argument("--tables", nargs="*", default=DIMENSIONS + FACTS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--no-sync", action="store_true", help="resume from CSVs already in staging, do not download"
    )
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    total_fail = 0
    for table in args.tables:
        m = land_table(table, args.dest, args.staging, args.workers, do_sync=not args.no_sync)
        total_fail += len(m["failures"])
        ratio = m["csv_bytes"] / m["parquet_bytes"] if m["parquet_bytes"] else 0
        print(
            f"{table}: files={m['files']} rows={m['rows']:,} csv={m['csv_bytes'] / 1e6:.1f}MB "
            f"parquet={m['parquet_bytes'] / 1e6:.1f}MB ratio={ratio:.1f}x failures={len(m['failures'])} "
            f"time={m['elapsed_s']}s",
            flush=True,
        )
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
