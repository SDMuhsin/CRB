#!/usr/bin/env python3
"""Produce plain-text wikitext-2-raw-v1 files for llama.cpp imatrix / perplexity.

Provenance (how each output file is produced):
  * Source: HF `wikitext` / `wikitext-2-raw-v1`, loaded via `datasets.load_dataset`
    from the repo's pre-warmed cache (HF_HOME=<ROOT>/downloads; Arrow cache at
    downloads/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625a
    ee71d232d685c3/). No network needed.
  * wt2_test.raw  = "\n\n".join(test_split['text'])  — the FULL test split, joined
    exactly like the repo's datautils.get_wikitext2 testenc (and like llama.cpp's
    canonical wiki.test.raw usage). Used with `llama-perplexity -f ... -c 2048`.
  * wt2_train_calib.txt = "\n\n".join(first K rows of the TRAIN split), where K is
    the smallest row count whose join reaches >= 100_000 Qwen3-0.6B tokens
    (counted with the HF AutoTokenizer from the local Qwen3-0.6B snapshot,
    add_special_tokens=False). Rows are taken in original order (no shuffle).
    Used as llama-imatrix calibration text.

Run: source <ROOT>/env/bin/activate && python3 make_eval_text.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "downloads"))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

QWEN3_SNAPSHOT = os.path.join(
    ROOT,
    "downloads/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
)
OUT_DIR = os.path.join(ROOT, "downloads/cpu_baselines/calib")
TARGET_CALIB_TOKENS = 100_000

os.makedirs(OUT_DIR, exist_ok=True)

train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# ---- full test split ----
test_text = "\n\n".join(test["text"])
test_path = os.path.join(OUT_DIR, "wt2_test.raw")
with open(test_path, "w", encoding="utf-8") as f:
    f.write(test_text)

# ---- ~100k-token calibration slice of the train split ----
tok = AutoTokenizer.from_pretrained(QWEN3_SNAPSHOT)

rows = train["text"]
lo, hi = 1, len(rows)


def ntok(k):
    return len(tok("\n\n".join(rows[:k]), add_special_tokens=False).input_ids)


# binary search for smallest prefix reaching the target
while lo < hi:
    mid = (lo + hi) // 2
    if ntok(mid) >= TARGET_CALIB_TOKENS:
        hi = mid
    else:
        lo = mid + 1

calib_text = "\n\n".join(rows[:lo])
calib_tokens = ntok(lo)
calib_path = os.path.join(OUT_DIR, "wt2_train_calib.txt")
with open(calib_path, "w", encoding="utf-8") as f:
    f.write(calib_text)

test_tokens = len(tok(test_text, add_special_tokens=False).input_ids)
print(f"wrote {test_path}: {len(test_text)} chars, {test_tokens} qwen3 tokens "
      f"({len(rows and test['text'])} rows, full test split)")
print(f"wrote {calib_path}: first {lo} train rows, {len(calib_text)} chars, "
      f"{calib_tokens} qwen3 tokens (target {TARGET_CALIB_TOKENS})")
sys.stdout.flush()
