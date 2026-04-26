"""
Shared evaluation pipeline for the BiLLM2 quantization runners.

After a runner finishes quantizing the model in-place, it calls
`evaluate_and_log_all()` to produce the standard evaluation suite:

    - PPL on WikiText-2 / C4 / PTB
    - 5-shot MMLU
    - 0-shot HellaSwag
    - 0-shot ARC-Easy + ARC-Challenge

Each evaluation writes its own row to the active CSV (the file pointed at
by `BILLM_BENCH_CSV`, defaulting to `results/benchmark_results.csv`). Rows
are tagged with `(dataset, metric)`:

    | dataset         | metric     |
    |-----------------|------------|
    | wikitext2 / c4 / ptb        | perplexity |
    | mmlu / hellaswag             | accuracy   |
    | arc-easy / arc-challenge     | accuracy   |

A single failed eval does not abort the others — exceptions are caught,
logged, and recorded as `value="FAILED:<ExceptionName>"` so partial CSV
state is still useful.

Public API:
    evaluate_and_log_all(model, model_name, dev, method, ..., flags...)
    parse_eval_args(parser) -> adds the standard --full_eval / --eval_*
                              / --eval_extra_ppl / --ppl_eval_seqlen flags.

Each runner uses these in two lines instead of duplicating ~80 lines of
per-eval boilerplate.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

# csv_utils lives next to this file under src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
# eval_*.py + datautils + eval_ppl_utils live at the repo root.
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from csv_utils import append_result  # noqa: E402


# ---------------------------------------------------------------------------
# Architecture dispatch
# ---------------------------------------------------------------------------

def get_ppl_eval_fn(model_name: str):
    """Return the architecture-specific PPL eval function.

    Mirrors the dispatch logic that was previously duplicated across
    run.py / src/run_*.py / PB-LLM/gptq_pb/run.py. All eval functions have
    signature `(model, testenc, dev, dataset, log_wandb, save_title, save)`
    and return a Python float (perplexity).
    """
    name = model_name.lower()
    if "opt" in name:
        from eval_ppl_utils import opt_eval
        return opt_eval
    if "llama" in name or "danube" in name:
        from eval_ppl_utils import llama_eval
        return llama_eval
    if "qwen" in name or "smollm" in name:
        from eval_ppl_utils import qwen_eval
        return qwen_eval
    if "granite" in name:
        from eval_ppl_utils import granite_eval
        return granite_eval
    if "pythia" in name:
        from eval_ppl_utils import pythia_eval
        return pythia_eval
    if "bloom" in name:
        from eval_ppl_utils import bloom_eval
        return bloom_eval
    raise ValueError(f"No PPL eval function known for model: {model_name}")


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def add_eval_cli(parser):
    """Add the standard evaluation flags to an argparse parser.

    Idempotent: if a flag is already registered (e.g. by an existing runner
    that defined `--eval_mmlu` directly), the call is skipped for that flag.
    Returns the parser for chaining.
    """
    def _add(name, **kwargs):
        if any(name in a.option_strings for a in parser._actions):
            return
        parser.add_argument(name, **kwargs)

    _add("--full_eval", action="store_true",
         help="Run the full eval suite: PPL on (wikitext2,c4,ptb) + MMLU + "
              "HellaSwag + ARC-Easy + ARC-Challenge. Equivalent to passing "
              "--eval_extra_ppl --eval_mmlu --eval_hellaswag --eval_arc.")
    _add("--eval_extra_ppl", action="store_true",
         help="Also evaluate PPL on C4 and PTB (in addition to args.dataset).")
    _add("--ppl_eval_seqlen", type=int, default=None,
         help="Override eval seqlen for PPL passes (default: model.seqlen).")
    _add("--eval_arc", action="store_true",
         help="Also evaluate on ARC-Easy + ARC-Challenge (0-shot).")
    _add("--eval_mmlu", action="store_true",
         help="Also evaluate on MMLU (5-shot, 57 subjects).")
    _add("--eval_hellaswag", action="store_true",
         help="Also evaluate on HellaSwag (0-shot).")
    return parser


def resolve_eval_flags(args, primary_dataset=None):
    """Map argparse Namespace into a normalized dict of which evals to run.

    `--full_eval` turns on all extras. Otherwise the individual flags are
    respected. `primary_dataset` is the calibration / always-eval PPL
    dataset (typically `args.dataset`); it is included in `ppl_datasets`.
    """
    full = bool(getattr(args, "full_eval", False))
    extra_ppl = full or bool(getattr(args, "eval_extra_ppl", False))
    eval_mmlu = full or bool(getattr(args, "eval_mmlu", False))
    eval_hellaswag = full or bool(getattr(args, "eval_hellaswag", False))
    eval_arc = full or bool(getattr(args, "eval_arc", False))

    ppl_datasets = []
    if primary_dataset:
        ppl_datasets.append(primary_dataset)
    if extra_ppl:
        for ds in ("wikitext2", "c4", "ptb"):
            if ds not in ppl_datasets:
                ppl_datasets.append(ds)

    return {
        "ppl_datasets": tuple(ppl_datasets),
        "eval_mmlu": eval_mmlu,
        "eval_hellaswag": eval_hellaswag,
        "eval_arc": eval_arc,
        "ppl_eval_seqlen": getattr(args, "ppl_eval_seqlen", None),
    }


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------

def _csv_row(model_name, method, dataset, metric, value, *, bpw, seed,
             blocksize, salient_metric, extra_params, quantization_time_s, notes):
    """Thin wrapper around csv_utils.append_result with consistent kwargs."""
    append_result(
        model=model_name,
        method=method,
        dataset=dataset,
        metric=metric,
        value=value,
        bpw=bpw if bpw is not None else "",
        seed=seed if seed is not None else "",
        blocksize=blocksize if blocksize is not None else "",
        salient_metric=salient_metric or "",
        extra_params=extra_params,
        quantization_time_s=quantization_time_s if quantization_time_s is not None else "",
        notes=notes or "",
    )


def _record_failure(model_name, method, dataset, metric, exc, *, bpw, seed,
                    blocksize, salient_metric, extra_params, quantization_time_s, notes):
    msg = f"{type(exc).__name__}: {str(exc)[:160]}"
    print(f"  *** {dataset}/{metric} FAILED: {msg}")
    traceback.print_exc()
    note = f"FAILED: {msg}" if not notes else f"{notes} | FAILED: {msg}"
    _csv_row(model_name, method, dataset, metric,
             value=f"FAILED:{type(exc).__name__}",
             bpw=bpw, seed=seed, blocksize=blocksize,
             salient_metric=salient_metric, extra_params=extra_params,
             quantization_time_s=quantization_time_s, notes=note)


# ---------------------------------------------------------------------------
# Eval runners
# ---------------------------------------------------------------------------

def _safe_to_cpu(model):
    """Best-effort move model to CPU and free GPU memory between evals."""
    try:
        import torch
        try:
            model.cpu()
        except Exception:
            pass
        torch.cuda.empty_cache()
    except Exception:
        pass


def _eval_ppl_one(model, model_name, dev, dataset_name, seed, eval_seqlen, save_title):
    """Run PPL on one dataset using the architecture-specific eval fn."""
    import torch  # local import keeps eval_utils import-cheap
    from datautils import get_loaders

    eval_fn = get_ppl_eval_fn(model_name)

    orig_seqlen = model.seqlen
    if eval_seqlen is not None:
        model.seqlen = int(eval_seqlen)

    try:
        _, testenc = get_loaders(
            dataset_name, seed=seed, seqlen=model.seqlen, model=model_name,
        )
        ppl = eval_fn(model, testenc, dev, dataset_name,
                      log_wandb=False, save_title=save_title, save=False)
        return float(ppl)
    finally:
        model.seqlen = orig_seqlen


def evaluate_and_log_all(
    model,
    model_name,
    dev,
    method,
    *,
    bpw=None,
    seed=0,
    blocksize=None,
    salient_metric=None,
    extra_params=None,
    quantization_time_s=None,
    notes=None,
    ppl_datasets=("wikitext2",),
    eval_mmlu=False,
    eval_hellaswag=False,
    eval_arc=False,
    ppl_eval_seqlen=None,
    save_title_prefix=None,
):
    """Run the configured eval suite and append one CSV row per (dataset, metric).

    Designed to be called once per quantization, after the model is in its final
    quantized state. Failures of individual evals do not abort the suite — each
    failure is logged and a placeholder CSV row is written.

    Args:
        model: the (already-quantized) HF causal-LM model. Must have `.seqlen`.
        model_name: HF model id (e.g. "Qwen/Qwen3-0.6B"). Drives both PPL eval
            dispatch and the cache_dir for downstream eval datasets.
        dev: torch device string ("cuda:0").
        method: short method tag for the CSV `method` column.
        bpw, seed, blocksize, salient_metric, extra_params,
        quantization_time_s, notes: identifying columns for each CSV row.
        ppl_datasets: tuple of PPL dataset names to evaluate (de-duplicated).
        eval_mmlu, eval_hellaswag, eval_arc: whether to run each downstream eval.
        ppl_eval_seqlen: override for PPL eval `model.seqlen` (default: keep).
        save_title_prefix: tag used in the legacy ./output/GLOBAL_*.json files.
    """
    if save_title_prefix is None:
        save_title_prefix = f"{model_name.replace('/', '_')}_{method}_seed{seed}"

    # De-duplicate ppl_datasets while preserving order.
    seen = set()
    ppl_list = []
    for d in ppl_datasets or ():
        if d and d not in seen:
            seen.add(d)
            ppl_list.append(d)

    def _common_kwargs():
        return dict(
            bpw=bpw, seed=seed, blocksize=blocksize,
            salient_metric=salient_metric, extra_params=extra_params,
            quantization_time_s=quantization_time_s, notes=notes,
        )

    # ----- PPL evals --------------------------------------------------------
    for ds in ppl_list:
        print(f"\n=== PPL eval on {ds} ===")
        t0 = time.time()
        try:
            ppl = _eval_ppl_one(
                model, model_name, dev, ds, seed,
                ppl_eval_seqlen, f"{save_title_prefix}_{ds}",
            )
            elapsed = time.time() - t0
            print(f"  {ds} PPL: {ppl:.4f}  ({elapsed:.1f}s)")
            _csv_row(model_name, method, ds, "perplexity", ppl, **_common_kwargs())
        except Exception as e:  # noqa: BLE001
            _record_failure(model_name, method, ds, "perplexity", e, **_common_kwargs())
        finally:
            _safe_to_cpu(model)

    # ----- Downstream evals -------------------------------------------------
    if eval_mmlu:
        print(f"\n=== MMLU (5-shot) ===")
        try:
            from eval_mmlu import eval_mmlu as _eval_mmlu
            acc = _eval_mmlu(model, model_name, dev,
                             save_title=f"{save_title_prefix}_MMLU")
            _csv_row(model_name, method, "mmlu", "accuracy", float(acc), **_common_kwargs())
        except Exception as e:  # noqa: BLE001
            _record_failure(model_name, method, "mmlu", "accuracy", e, **_common_kwargs())
        finally:
            _safe_to_cpu(model)

    if eval_hellaswag:
        print(f"\n=== HellaSwag (0-shot) ===")
        try:
            from eval_hellaswag import eval_hellaswag as _eval_hs
            acc = _eval_hs(model, model_name, dev,
                           save_title=f"{save_title_prefix}_HELLASWAG")
            _csv_row(model_name, method, "hellaswag", "accuracy", float(acc),
                     **_common_kwargs())
        except Exception as e:  # noqa: BLE001
            _record_failure(model_name, method, "hellaswag", "accuracy", e,
                            **_common_kwargs())
        finally:
            _safe_to_cpu(model)

    if eval_arc:
        print(f"\n=== ARC (0-shot) ===")
        try:
            from eval_arc import eval_arc as _eval_arc
            arc_results = _eval_arc(model, model_name, dev,
                                    save_title=f"{save_title_prefix}_ARC")
            for split, ds_label in [
                ("ARC-Easy", "arc-easy"),
                ("ARC-Challenge", "arc-challenge"),
            ]:
                if split in arc_results:
                    _csv_row(model_name, method, ds_label, "accuracy",
                             float(arc_results[split]["accuracy"]),
                             **_common_kwargs())
                else:
                    _record_failure(model_name, method, ds_label, "accuracy",
                                    KeyError(split), **_common_kwargs())
        except Exception as e:  # noqa: BLE001
            _record_failure(model_name, method, "arc-easy", "accuracy", e,
                            **_common_kwargs())
            _record_failure(model_name, method, "arc-challenge", "accuracy", e,
                            **_common_kwargs())
        finally:
            _safe_to_cpu(model)
