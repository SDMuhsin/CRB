import torch
import torch.nn as nn
import json
import os
import fcntl
import tempfile
import subprocess
from datasets import load_dataset


EVAL_SAVE_FILE = "./output/GLOBAL_HUMANEVAL.json"


def save_result(save_title, results):
    """Save HumanEval results to JSON file with file locking."""
    os.makedirs(os.path.dirname(EVAL_SAVE_FILE), exist_ok=True)

    if not os.path.exists(EVAL_SAVE_FILE):
        with open(EVAL_SAVE_FILE, "w") as f:
            json.dump({}, f)

    with open(EVAL_SAVE_FILE, "r+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            data = json.load(f)
            data[save_title] = results
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def truncate_at_stop_patterns(text):
    """Truncate generated code at common function-end patterns."""
    stop_patterns = ['\nclass ', '\ndef ', '\n# ', '\nif __name__', '\nprint(']
    min_idx = len(text)
    for pattern in stop_patterns:
        idx = text.find(pattern)
        if idx != -1 and idx < min_idx:
            min_idx = idx
    return text[:min_idx]


def check_correctness(prompt, completion, test, entry_point, timeout=10):
    """
    Execute generated code with test cases and check correctness.

    Returns True if all test cases pass, False otherwise.
    """
    full_code = prompt + completion + "\n" + test
    # The test string should call check(entry_point), but add it if missing
    if f"check({entry_point})" not in full_code:
        full_code += f"\ncheck({entry_point})\n"

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            f.flush()
            tmp_file = f.name

        result = subprocess.run(
            ['python3', tmp_file],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)


@torch.no_grad()
def eval_humaneval(model, model_name, dev, save_title='UNNAMED_HUMANEVAL'):
    """
    Evaluate a model on HumanEval (code generation).

    Greedy generation (pass@1). For each problem, generates a function body,
    concatenates with the prompt, and executes test cases to check correctness.

    Note: General language models (e.g., OPT) that are not trained for code
    will typically score 0% — this is expected, not anomalous.
    """
    print("Evaluating on HumanEval (pass@1, greedy) ...")

    from datautils import get_tokenizer
    tokenizer = get_tokenizer(model_name)

    # Load HumanEval dataset
    dataset = load_dataset("openai_humaneval", split="test",
                           cache_dir="./downloads/datasets")

    # Enable cache for efficient autoregressive generation
    use_cache = model.config.use_cache
    model.config.use_cache = True
    model.to(dev)

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    passed = 0
    total = 0
    per_problem = []

    for i, example in enumerate(dataset):
        task_id = example['task_id']
        prompt = example['prompt']
        test = example['test']
        entry_point = example['entry_point']

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(dev)

        # Truncate prompt if too long (leave room for generation)
        max_prompt_len = model.seqlen - 256
        if input_ids.shape[1] > max_prompt_len:
            input_ids = input_ids[:, -max_prompt_len:]

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode only the generated tokens
        generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                     skip_special_tokens=True)

        # Truncate at stop patterns (end of function)
        generated = truncate_at_stop_patterns(generated)

        # Check correctness
        is_correct = check_correctness(prompt, generated, test, entry_point,
                                       timeout=10)

        if is_correct:
            passed += 1
        total += 1

        per_problem.append({
            "task_id": task_id,
            "passed": is_correct,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(dataset)}] Running pass@1: {passed/total:.4f}")

    pass_at_1 = passed / total

    print(f"\nHumanEval Results:")
    print(f"  pass@1: {pass_at_1:.4f} ({passed}/{total})")

    results = {
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": total,
        "per_problem": per_problem,
    }

    save_result(save_title, results)
    print(f"  Results saved to {EVAL_SAVE_FILE}")

    model.cpu()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return pass_at_1
