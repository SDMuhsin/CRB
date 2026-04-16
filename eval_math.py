import torch
import torch.nn as nn
import json
import os
import re
import fcntl
from datasets import load_dataset


EVAL_SAVE_FILE = "./output/GLOBAL_MATH.json"


def save_result(save_title, results):
    """Save MATH results to JSON file with file locking."""
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


def extract_boxed_answer(text):
    """
    Extract the last \\boxed{...} answer from text.
    Handles nested braces.
    """
    # Find the last occurrence of \boxed{
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return None

    # Extract content handling nested braces
    start = idx + len('\\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i-1].strip()
    return None


def extract_last_number(text):
    """Extract the last number from text as a fallback."""
    # Find all numbers (including decimals, negatives, fractions)
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer):
    """Normalize a math answer for comparison."""
    if answer is None:
        return None

    answer = answer.strip()

    # Remove surrounding $ signs
    answer = answer.strip('$')

    # Remove trailing period
    if answer.endswith('.'):
        answer = answer[:-1]

    # Remove \text{} wrappers
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)

    # Remove spaces
    answer = answer.replace(' ', '')

    # Try fraction conversion
    frac_match = re.match(r'^(-?\d+)/(\d+)$', answer)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            answer = str(num / den)

    # Try \\frac{a}{b} conversion
    frac_match = re.match(r'^\\frac\{(-?\d+)\}\{(\d+)\}$', answer)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            answer = str(num / den)

    # Try float conversion for numeric comparison
    try:
        val = float(answer)
        # Round to avoid floating point issues
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}"
    except (ValueError, OverflowError):
        return answer.lower()


def answers_match(predicted, ground_truth):
    """Check if predicted and ground truth answers match."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm is None or gt_norm is None:
        return False

    # Exact string match after normalization
    if pred_norm == gt_norm:
        return True

    # Try numeric comparison with tolerance
    try:
        pred_val = float(pred_norm)
        gt_val = float(gt_norm)
        return abs(pred_val - gt_val) < 1e-4
    except (ValueError, OverflowError):
        return False


@torch.no_grad()
def eval_math(model, model_name, dev, save_title='UNNAMED_MATH'):
    """
    Evaluate a model on the MATH benchmark (competition mathematics).

    Greedy generation. For each problem, generates a solution, extracts
    the answer (from \\boxed{} notation or last number), and compares
    with ground truth.

    Note: General language models not trained on math will typically
    score near 0% — this is expected, not anomalous.
    """
    print("Evaluating on MATH (greedy generation) ...")

    from datautils import get_tokenizer
    tokenizer = get_tokenizer(model_name)

    # Load MATH dataset (all subjects combined)
    math_configs = ['algebra', 'counting_and_probability', 'geometry',
                    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    all_examples = []
    for cfg in math_configs:
        split_data = load_dataset("EleutherAI/hendrycks_math", cfg, split="test",
                                  cache_dir="./downloads/datasets")
        for ex in split_data:
            ex['type'] = cfg
            all_examples.append(ex)
    dataset = all_examples
    print(f"  MATH test set: {len(dataset)} problems across {len(math_configs)} categories")

    # Enable cache for efficient generation
    use_cache = model.config.use_cache
    model.config.use_cache = True
    model.to(dev)

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    correct = 0
    total = 0
    by_level = {}
    by_type = {}

    for i, example in enumerate(dataset):
        problem = example['problem']
        solution = example['solution']
        level = example.get('level', 'Unknown')
        prob_type = example.get('type', 'Unknown')

        # Extract ground truth answer from solution
        gt_answer = extract_boxed_answer(solution)
        if gt_answer is None:
            gt_answer = extract_last_number(solution)
        if gt_answer is None:
            continue

        # Format prompt
        prompt = f"Problem: {problem}\nSolution:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(dev)

        # Truncate prompt if too long (leave room for generation)
        max_prompt_len = model.seqlen - 512
        if max_prompt_len < 1:
            max_prompt_len = model.seqlen // 2
        if input_ids.shape[1] > max_prompt_len:
            input_ids = input_ids[:, -max_prompt_len:]

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode only the generated tokens
        generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                     skip_special_tokens=True)

        # Extract predicted answer
        pred_answer = extract_boxed_answer(generated)
        if pred_answer is None:
            pred_answer = extract_last_number(generated)

        is_correct = answers_match(pred_answer, gt_answer)

        if is_correct:
            correct += 1
        total += 1

        # Track by level and type
        for grouping, key in [(by_level, level), (by_type, prob_type)]:
            if key not in grouping:
                grouping[key] = {"correct": 0, "total": 0}
            grouping[key]["total"] += 1
            if is_correct:
                grouping[key]["correct"] += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(dataset)}] Running accuracy: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0.0

    # Compute breakdowns
    level_accuracies = {}
    for key in sorted(by_level.keys()):
        d = by_level[key]
        level_accuracies[key] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    type_accuracies = {}
    for key in sorted(by_type.keys()):
        d = by_type[key]
        type_accuracies[key] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    print(f"\nMATH Results:")
    print(f"  Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    if level_accuracies:
        print(f"  By level:")
        for lvl, acc in level_accuracies.items():
            print(f"    {lvl}: {acc:.4f} ({by_level[lvl]['correct']}/{by_level[lvl]['total']})")
    if type_accuracies:
        print(f"  By type:")
        for typ, acc in type_accuracies.items():
            print(f"    {typ}: {acc:.4f} ({by_type[typ]['correct']}/{by_type[typ]['total']})")

    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_level": {k: {"accuracy": level_accuracies[k], **v} for k, v in by_level.items()},
        "by_type": {k: {"accuracy": type_accuracies[k], **v} for k, v in by_type.items()},
    }

    save_result(save_title, results)
    print(f"  Results saved to {EVAL_SAVE_FILE}")

    model.cpu()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return accuracy
