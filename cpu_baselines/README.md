# CPU Baselines — SOTA multicore CPU kernels for low-bit Qwen3

Campaign home for CPU-baseline benchmarking (see the binding protocol in
`llmdocs/cpu_kernel/CPU_BASELINES_TRACKER.md`). Goal: measure reference SOTA
CPU inference engines (llama.cpp first, ik_llama.cpp next) running Qwen3-0.6B
at low bit-widths, as the speed/quality bar for future DOML/SDOML CPU kernels.

## Layout

- `scripts/` — re-runnable bash/python pipeline (all paths derived from a
  `ROOT` var = repo root; activate `env/` first where noted):
  - `build_llamacpp.sh` — CPU-only Release build (cmake+ninja from the venv,
    `-DGGML_CUDA=OFF -DGGML_NATIVE=ON`) of llama-bench / llama-perplexity /
    llama-quantize / llama-imatrix / llama-cli in `temp/llama.cpp/build/bin/`.
  - `make_eval_text.py` — writes `downloads/cpu_baselines/calib/wt2_train_calib.txt`
    (~100k Qwen3 tokens of wikitext-2-raw-v1 TRAIN, "\n\n"-joined) and
    `wt2_test.raw` (FULL test split, same joining) from the repo HF cache
    (`HF_HOME=downloads/`, offline). Provenance in the script docstring.
  - `convert_llamacpp.sh` — HF snapshot → `qwen3-0.6b-bf16.gguf`
    (convert_hf_to_gguf.py, worked fine on transformers 5.3.0) → `qwen3-0.6b-f16.gguf`
    (llama-quantize F16). **F16 is the PPL/speed anchor.**
  - `imatrix_llamacpp.sh` — llama-imatrix on the bf16 GGUF with the calib text
    (-c 2048, t=24) → `qwen3-0.6b-wt2train.imatrix` (GGUF-format imatrix).
  - `quantize_llamacpp.sh` — ladder Q4_K_M, Q4_0 (no imatrix) + Q3_K_M, Q2_K,
    IQ2_M, IQ2_XS, IQ2_XXS, IQ1_S (with imatrix).
  - `ppl_llamacpp.sh` — sequential `llama-perplexity -f wt2_test.raw -c 2048 -t 24`
    over anchor + all 8 quants (full test set, ~2h total).
  - `bench_llamacpp.sh` — sequential `llama-bench -p 512 -n 128 -r 3 -t 1,6,12,24,48`
    per model (single pass emits JSON to stdout + md to stderr via `-o json -oe md`),
    plus a `numactl --cpunodebind=0 ... -t 12` node0 point (`--membind` is EPERM in
    this container, so it is CPU-binding only — CSV numa column says `node0-cpubind`).
    Resumable (skips models whose .json already exists). Machine must be otherwise idle.
  - `append_csv_llamacpp.py` — parses the bench JSON / PPL / quantize logs and
    appends schema-conformant rows to the CSV (append-only; don't run twice).
- `results/cpu_baseline_results.csv` — all measurements, schema in the tracker.
  (Re-included in git via `!cpu_baselines/results/` in `.gitignore`.)

## Artifact locations (big files, not in git)

- GGUFs + imatrix: `downloads/cpu_baselines/llama.cpp/`
- Calibration/eval text: `downloads/cpu_baselines/calib/`
- Raw run logs (every CSV row traces to one): `llmdocs/cpu_kernel/verify/cpu_baseline_logs/`
- Framework clones: `temp/llama.cpp` (commit recorded per CSV row), etc.

## How to rerun end-to-end (llama.cpp)

```bash
cd <repo-root>
source env/bin/activate
bash cpu_baselines/scripts/build_llamacpp.sh
python3 cpu_baselines/scripts/make_eval_text.py
bash cpu_baselines/scripts/convert_llamacpp.sh
bash cpu_baselines/scripts/imatrix_llamacpp.sh          # ~7 min
bash cpu_baselines/scripts/quantize_llamacpp.sh         # ~5 min
nohup bash cpu_baselines/scripts/ppl_llamacpp.sh &      # hours; poll llamacpp_ppl_progress.log
nohup bash cpu_baselines/scripts/bench_llamacpp.sh &    # idle machine only; poll llamacpp_bench_progress.log
python3 cpu_baselines/scripts/append_csv_llamacpp.py
```

## ik_llama.cpp (Phase 4)

Same protocol, same calib/test text, same machine — numbers directly comparable
to the llama.cpp rows in the CSV (framework column distinguishes them). Clone at
`temp/ik_llama.cpp` (commit recorded per CSV row). Key differences vs mainline:

- **No re-conversion needed:** ik's binaries load the Phase-2 mainline
  `qwen3-0.6b-bf16.gguf` / `-f16.gguf` directly (verified with a coherent
  llama-cli generation). Quants are made FROM the mainline bf16 GGUF.
- **Own imatrix:** mainline (2026) writes GGUF-format imatrix files; ik uses its
  legacy binary format, so `imatrix_ikllamacpp.sh` generates ik's own
  `downloads/cpu_baselines/ik_llama.cpp/qwen3-0.6b-wt2train.ik.imatrix`
  from the same calibration text.
- **Ladder:** mainline-comparable Q4_K_M, Q4_0, Q2_K, IQ2_XS, IQ2_XXS +
  ik-native IQ4_KS, IQ3_K, IQ2_KL, IQ2_K, IQ2_KS, IQ2_KT
  (`quantize_ikllamacpp.sh`; imatrix for everything ≤ ~3.5 bpw, 4-bit types without).
- **BPW parsing:** ik's llama-quantize does not print a whole-file BPW; the CSV
  bpw comes from the model-load line `model size = ... (X.XXX BPW)` in the PPL
  logs (same bytes×8/elements definition as mainline's quantize print).
- **Anchor:** the F16 anchor PPL is re-measured with IK'S OWN llama-perplexity
  (never reuse the mainline anchor number across harnesses).
- **Run-time repack (`-rtr 1`):** `bench_ikllamacpp.sh` adds one extra
  llama-bench pass at `-t 24` with `-rtr 1` per model (ik repacks tensors into
  interleaved `_R4` layouts at load time) — CSV rows with `notes=rtr`.

```bash
cd <repo-root>
source env/bin/activate
bash cpu_baselines/scripts/build_ikllamacpp.sh
bash cpu_baselines/scripts/imatrix_ikllamacpp.sh          # ~4 min
bash cpu_baselines/scripts/quantize_ikllamacpp.sh         # ~10 min
nohup bash cpu_baselines/scripts/ppl_ikllamacpp.sh &      # ~1.5 h; poll ikllamacpp_ppl_progress.log
nohup bash cpu_baselines/scripts/bench_ikllamacpp.sh &    # idle machine only; poll ikllamacpp_bench_progress.log
python3 cpu_baselines/scripts/append_csv_ikllamacpp.py
```

## Caveats

- llama.cpp PPL is NOT comparable to the repo's `run.py` PPL (different
  windowing) — only compare ratios against the same-tool F16 anchor.
- Benchmarks are meaningful only on an otherwise-idle machine; keep the GPUs
  and other CPU jobs quiet during `bench_llamacpp.sh`.
- i-quants (IQ2*/IQ1*) are known to decode slower on CPU than K-quants —
  slower tg128 there is expected, not a harness bug.
