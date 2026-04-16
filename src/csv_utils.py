"""
Benchmark CSV utility — append evaluation results to results/benchmark_results.csv.

Usage:
    from csv_utils import append_result
    append_result(
        model="Qwen/Qwen3-0.6B", method="doml", dataset="wikitext2",
        metric="perplexity", value=30.55, bpw=2.09, seed=0,
        blocksize=128, salient_metric="magnitude",
        extra_params={}, quantization_time_s=120.5, notes=""
    )
"""

import csv
import fcntl
import json
import os
from datetime import datetime

_DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "results", "benchmark_results.csv")


def _csv_path():
    """Return active CSV path, honoring BILLM_BENCH_CSV env var if set."""
    return os.environ.get("BILLM_BENCH_CSV", _DEFAULT_CSV_PATH)


# Backwards-compat alias for any callers that import CSV_PATH directly.
CSV_PATH = _csv_path()

COLUMNS = [
    "timestamp", "model", "method", "dataset", "metric", "value",
    "bpw", "seed", "blocksize", "salient_metric", "extra_params",
    "quantization_time_s", "notes",
]


def _ensure_csv():
    """Create CSV with header if it doesn't exist."""
    path = _csv_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)


def append_result(model, method, dataset, metric, value,
                  bpw="", seed="", blocksize="", salient_metric="",
                  extra_params=None, quantization_time_s="", notes=""):
    """Append a single result row to the benchmark CSV (thread/process safe)."""
    _ensure_csv()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if job_id:
        notes = f"slurm_job_id={job_id}" + (f" | {notes}" if notes else "")
    row = [
        datetime.now().isoformat(timespec="seconds"),
        model,
        method,
        dataset,
        metric,
        value,
        bpw,
        seed,
        blocksize,
        salient_metric,
        json.dumps(extra_params) if extra_params else "",
        quantization_time_s,
        notes,
    ]
    path = _csv_path()
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def result_exists(model, method, dataset, metric, seed=""):
    """Check if a result already exists in the CSV (for skip-if-done logic)."""
    path = _csv_path()
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["model"] == model and row["method"] == method
                    and row["dataset"] == dataset and row["metric"] == metric):
                if seed != "" and row.get("seed", "") != str(seed):
                    continue
                return True
    return False
