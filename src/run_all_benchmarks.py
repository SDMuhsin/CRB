#!/usr/bin/env python3
"""
Master benchmark runner: all methods × all models × all datasets.
Runs each combination as a subprocess, skips already-completed runs.
Logs progress to llmdocs/trackers/benchmark_progress.md.

Usage:
    source env/bin/activate
    python3 -u src/run_all_benchmarks.py [--dry-run] [--method METHOD] [--model MODEL] [--skip-downstream]
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csv_utils import CSV_PATH, result_exists

TRACKER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "llmdocs", "trackers", "benchmark_progress.md")

# ── Models ──────────────────────────────────────────────────────────────────
MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]

# ── PPL datasets ────────────────────────────────────────────────────────────
DATASETS = ["wikitext2", "c4"]

# ── Method definitions ──────────────────────────────────────────────────────
# Each method: (name, script_type, base_args, bpw_approx)
# script_type: "run.py", "sinq", "tesseraq", "pbllm"
METHODS = [
    {
        "name": "fp16",
        "script": "run.py",
        "method_arg": "fp16",
        "extra_args": [],
        "bpw": 16,
    },
    {
        "name": "rtn",
        "script": "run.py",
        "method_arg": "rtn",
        "extra_args": ["--blocksize", "128", "--salient_metric", "magnitude"],
        "bpw": 1.07,
    },
    {
        "name": "gptq_2bit",
        "script": "run.py",
        "method_arg": "2bit",
        "extra_args": ["--blocksize", "128", "--salient_metric", "magnitude"],
        "bpw": 2.0,
    },
    {
        "name": "braq",
        "script": "run.py",
        "method_arg": "braq",
        "extra_args": ["--blocksize", "128", "--salient_metric", "magnitude"],
        "bpw": 1.07,
    },
    {
        "name": "crbog",
        "script": "run.py",
        "method_arg": "crbog",
        "extra_args": ["--blocksize", "128", "--salient_metric", "magnitude",
                       "--corr_damp", "0.1", "--lam", "1e-5"],
        "bpw": 1.07,
    },
    {
        "name": "doml",
        "script": "run.py",
        "method_arg": "doml",
        "extra_args": ["--blocksize", "128", "--salient_metric", "magnitude"],
        "bpw": 2.09,
    },
    {
        "name": "sinq",
        "script": "sinq",
        "method_arg": None,
        "extra_args": ["--nbits", "2", "--group_size", "64"],
        "bpw": 2.51,
    },
    {
        "name": "tesseraq",
        "script": "tesseraq",
        "method_arg": None,
        "extra_args": ["--bit", "2", "--group_size", "128", "--iterations", "100"],
        "bpw": 2.25,
    },
    {
        "name": "pbllm",
        "script": "pbllm",
        "method_arg": "xnor",
        "extra_args": ["--low_frac", "0.7", "--high_bit", "8",
                       "--blocksize", "128", "--salient_metric", "magnitude"],
        "bpw": None,  # depends on low_frac
    },
]


def build_command(method, model, dataset, seed=0, downstream=True):
    """Build the subprocess command for a given method/model/dataset."""
    m = method
    base_device = ["--device", "cuda:0"]
    base_seed = ["--seed", str(seed)]

    if m["script"] == "run.py":
        cmd = ["python3", "-u", "run.py", model, dataset, m["method_arg"]]
        cmd += m["extra_args"] + base_device + base_seed
        if downstream:
            cmd += ["--eval_arc", "--eval_mmlu", "--eval_hellaswag"]
        return cmd

    elif m["script"] == "sinq":
        cmd = ["python3", "-u", "src/run_sinq.py", model, dataset]
        cmd += m["extra_args"] + base_device + base_seed
        if downstream:
            cmd += ["--eval_arc", "--eval_mmlu", "--eval_hellaswag"]
        return cmd

    elif m["script"] == "tesseraq":
        cmd = ["python3", "-u", "src/run_tesseraq.py", model, dataset]
        cmd += m["extra_args"] + base_device + base_seed
        if downstream:
            cmd += ["--eval_arc", "--eval_mmlu", "--eval_hellaswag"]
        return cmd

    elif m["script"] == "pbllm":
        cmd = ["python3", "-u", "PB-LLM/gptq_pb/run.py", model, dataset, m["method_arg"]]
        cmd += m["extra_args"] + base_seed
        if downstream:
            cmd += ["--eval_arc", "--eval_mmlu", "--eval_hellaswag"]
        return cmd

    raise ValueError(f"Unknown script type: {m['script']}")


def is_done(method_name, model, dataset, seed=0):
    """Check if this method/model/dataset PPL already exists in CSV."""
    return result_exists(model, method_name, dataset, "perplexity", seed)


def update_tracker(status_lines):
    """Write progress tracker."""
    os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)
    with open(TRACKER_PATH, "w") as f:
        f.write("# Benchmark Progress\n\n")
        f.write(f"Last updated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("| # | Method | Model | Dataset | Status | Time |\n")
        f.write("|---|--------|-------|---------|--------|------|\n")
        for i, line in enumerate(status_lines, 1):
            f.write(f"| {i} | {line} |\n")


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--method", type=str, default=None,
                        help="Only run this method (e.g., 'doml', 'fp16')")
    parser.add_argument("--model", type=str, default=None,
                        help="Only run this model (e.g., 'Qwen/Qwen3-0.6B')")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Only run this dataset (e.g., 'wikitext2')")
    parser.add_argument("--skip-downstream", action="store_true",
                        help="Skip downstream evals (ARC, MMLU, HellaSwag)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    methods = METHODS
    models = MODELS
    datasets = DATASETS

    if args.method:
        methods = [m for m in methods if m["name"] == args.method]
        if not methods:
            print(f"Unknown method: {args.method}")
            print(f"Available: {[m['name'] for m in METHODS]}")
            sys.exit(1)

    if args.model:
        models = [m for m in models if m == args.model]
        if not models:
            print(f"Unknown model: {args.model}")
            sys.exit(1)

    if args.dataset:
        datasets = [d for d in datasets if d == args.dataset]

    # Build run list
    runs = []
    for method in methods:
        for model in models:
            for dataset in datasets:
                runs.append((method, model, dataset))

    total = len(runs)
    print(f"Total benchmark runs: {total}")
    print(f"Methods: {[m['name'] for m in methods]}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Downstream evals: {'NO' if args.skip_downstream else 'YES (ARC, MMLU, HellaSwag)'}")
    print()

    status_lines = []
    completed = 0
    skipped = 0
    failed = 0

    for idx, (method, model, dataset) in enumerate(runs):
        mname = method["name"]
        model_short = model.split("/")[-1]

        # Check if already done
        if is_done(mname, model, dataset, args.seed):
            print(f"[{idx+1}/{total}] SKIP (done): {mname} | {model_short} | {dataset}")
            status_lines.append(f"{mname} | {model_short} | {dataset} | SKIP (done) | - ")
            skipped += 1
            continue

        cmd = build_command(method, model, dataset, args.seed,
                            downstream=not args.skip_downstream)
        cmd_str = " ".join(cmd)

        if args.dry_run:
            print(f"[{idx+1}/{total}] DRY-RUN: {cmd_str}")
            status_lines.append(f"{mname} | {model_short} | {dataset} | DRY-RUN | - ")
            continue

        print(f"\n{'='*70}")
        print(f"[{idx+1}/{total}] RUNNING: {mname} | {model_short} | {dataset}")
        print(f"CMD: {cmd_str}")
        print(f"{'='*70}\n")

        tick = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd="/workspace/BiLLM2",
                timeout=7200,  # 2 hour timeout per run
                capture_output=False,  # stream output live
            )
            elapsed = time.time() - tick
            elapsed_str = f"{elapsed/60:.1f}m"

            if result.returncode == 0:
                print(f"\n  OK ({elapsed_str})")
                status_lines.append(f"{mname} | {model_short} | {dataset} | OK | {elapsed_str}")
                completed += 1
            else:
                print(f"\n  FAILED (exit {result.returncode}, {elapsed_str})")
                status_lines.append(f"{mname} | {model_short} | {dataset} | FAILED ({result.returncode}) | {elapsed_str}")
                failed += 1

        except subprocess.TimeoutExpired:
            elapsed = time.time() - tick
            print(f"\n  TIMEOUT after {elapsed/60:.1f}m")
            status_lines.append(f"{mname} | {model_short} | {dataset} | TIMEOUT | {elapsed/60:.1f}m")
            failed += 1

        except Exception as e:
            elapsed = time.time() - tick
            print(f"\n  ERROR: {e} ({elapsed/60:.1f}m)")
            status_lines.append(f"{mname} | {model_short} | {dataset} | ERROR | {elapsed/60:.1f}m")
            failed += 1

        # Update tracker after each run
        update_tracker(status_lines)

    # Final summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY")
    print(f"  Completed: {completed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {total}")
    print(f"  CSV: {CSV_PATH}")
    print(f"{'='*70}")

    update_tracker(status_lines)


if __name__ == "__main__":
    main()
