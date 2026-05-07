This repository houses code for the paper titled "Distribution-Optimal Multi-Level Quantization and Joint Sparsification for 2-Bit LLMs", currently under review at NeurIPS 2026.


# Abstract

We introduce DOML (*Distribution-Optimal Multi-Level* quantization), a 2-bit post-training quantization (PTQ) method that pairs per-row Lloyd-Max optimal codebooks with a three-way salience-based column partition: three four-level codebooks per row, yielding twelve effective reconstruction levels at 0.05-0.19 bpw of codebook overhead. Across seven Qwen3 and Llama-3 model points, DOML ties or wins against every 2-bpw scalar PTQ baseline on at least 6 of 7 in geometric-mean perplexity and 5 of 7 in mean zero-shot accuracy, with average perplexity reductions of 38%-72% and accuracy gains of 3.6-10.0 pp over the leading non-uniform scalar competitors (TesseraQ, GuidedQuant + LNQ, LeanQuant). The construction extends to K = 2^b levels per partition for b in {3, 4, 5}, reaching within 0.7% of FP16 on the narrow-hidden-dimension model points at 5 bits. We further introduce SDOML, a joint sparsity-and-quantization extension that co-optimizes a per-row keep mask with the Lloyd-Max codebook in a single Lloyd-style alternation; an SDOML+partition variant masking 50% of the inactive column partition outperforms SparseGPT (joint sparse + 2-bit) by three to four orders of magnitude on Qwen3-1.7B and Llama-3.2-1B.


# Dependencies

```
torch>=2.3
transformers>=4.51
tokenizers>=0.21
huggingface_hub>=0.25
safetensors>=0.4
accelerate>=0.30
datasets>=2.18,<3
numpy>=1.24,<2
sentencepiece>=0.2
numba>=0.58
flash1dkmeans>=0.1
```

See `requirements.txt` for the full pinned list.


# Steps to reproduce work

The base command to run quantization and the full seven-task evaluation suite is

```
python3 run.py $model $dataset $technique --blocksize 128 --salient_metric magnitude --device="cuda:0" --full_eval
```

Models supported (verified): `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`, `NousResearch/Meta-Llama-3.1-8B` (ungated mirror), `NousResearch/Llama-2-7b-hf`.

Techniques implemented in this repository:

- `doml`: DOML (per-row Lloyd-Max with three-way structural partition).
- `sdoml`: Plain SDOML (single K=4 codebook with joint sparse mask).
- `sdoml_partition` with `--sdoml_asymmetric`: asymmetric SDOML+partition (mask only on the inactive partition).
- `magfit`: decoupled prune-then-Lloyd-Max ablation.
- `2bit` with `--partition 1 --global_scale`: paper-faithful per-row GPTQ.
- `2bit` with `--disable_gptq`: round-to-nearest 2-bit.
- `braq`: BiLLM baseline.

Higher bit widths (b in {3, 4, 5}): pass `--codebook_bits {3,4,5}` to the `doml`/`sdoml`/`sdoml_partition`/`magfit` techniques.

Joint sparsity sweep: pass `--sparsity {0.05, 0.20, 0.25, 0.50, 0.75}` to the `sdoml`/`sdoml_partition`/`magfit` techniques.

Additional baselines have separate runners:

- TesseraQ: `src/run_tesseraq.py` (with `--use_awq_init` by default).
- LeanQuant: `src/run_leanquant.py` (with `--nbits` in `{2,3,4,5}`).
- GuidedQuant + LNQ: `src/run_lnq.py --full_pipeline`.
- SparseGPT: `src/run_sparsegpt.py` (with `--sparsity` and optional `--nbits`).
- SINQ: `src/run_sinq.py`.
- PB-LLM: `PB-LLM/gptq_pb/run.py`.

Datasets:

- Calibration: WikiText-2 (default), C4, Penn Treebank.
- Perplexity evaluation: WikiText-2, C4, Penn Treebank.
- Downstream evaluation: MMLU (5-shot), HellaSwag (0-shot), ARC-Easy (0-shot), ARC-Challenge (0-shot).

The `sbatch/` directory contains seven driver scripts that fan out the full method matrix on a SLURM cluster (one script per model point). Per-method walltime, MIG-slice, and host-RAM budgets are encoded in those scripts.


# Related projects

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

[TesseraQ: Ultra Low-Bit LLM Post-Training Quantization with Block Reconstruction](https://github.com/Intelligent-Computing-Lab-Yale/TesseraQ)

[LeanQuant: Accurate and Scalable Large Language Model Quantization with Loss-Error-Aware Grid](https://github.com/LeanQuant/LeanQuant)

[GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance](https://github.com/snu-mllab/GuidedQuant)

[SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://github.com/IST-DASLab/sparsegpt)

[BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://github.com/Aaronhuang-778/BiLLM)


# Citation

A preprint is not yet available. Reviewers may obtain additional information through the venue's reviewer-author communication channel.
