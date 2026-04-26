import torch
import torch.nn as nn
import json
import os
import re
import fcntl
from datasets import load_dataset


EVAL_SAVE_FILE = "./output/GLOBAL_HELLASWAG.json"

# Honor BILLM_DOWNLOADS_DIR so cluster jobs route HF dataset cache to scratch.
_DATASETS_CACHE = os.path.join(os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads"), "datasets")


def save_result(save_title, results):
    """Save HellaSwag results to JSON file with file locking."""
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


def preprocess_hellaswag(text):
    """Clean up HellaSwag text artifacts."""
    text = text.strip()
    text = re.sub(r'\[header\]\s*', '', text)
    text = re.sub(r'\[.*?\]\s*', '', text)
    text = text.strip()
    return text


def compute_completion_ll(model, input_ids, prompt_len):
    """
    Compute total log-likelihood and token count for completion tokens.

    Args:
        model: language model
        input_ids: [1, total_len] tensor on device
        prompt_len: number of prompt tokens (completion starts here)

    Returns:
        (total_log_likelihood, num_completion_tokens)
    """
    n_completion = input_ids.shape[1] - prompt_len
    if n_completion <= 0:
        return 0.0, 0

    outputs = model(input_ids)
    logits = outputs.logits[0]  # [total_len, vocab_size]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # For completion token at position j, the predicting logit is at position j-1
    completion_ids = input_ids[0, prompt_len:]  # [n_completion]
    prediction_log_probs = log_probs[prompt_len - 1:-1]  # [n_completion, vocab_size]
    token_log_probs = prediction_log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)

    total_ll = token_log_probs.sum().item()
    return total_ll, n_completion


@torch.no_grad()
def eval_hellaswag(model, model_name, dev, save_title='UNNAMED_HELLASWAG'):
    """
    Evaluate a model on HellaSwag (commonsense sentence completion).

    0-shot evaluation. For each example, computes the length-normalized
    log-likelihood of each of 4 ending options given the context, and
    picks the highest-scoring ending.
    """
    print("Evaluating on HellaSwag (0-shot) ...")

    from datautils import get_tokenizer
    tokenizer = get_tokenizer(model_name)

    # Load HellaSwag — use validation split (test labels are hidden)
    dataset = load_dataset("Rowan/hellaswag", split="validation",
                           cache_dir=_DATASETS_CACHE)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(dev)

    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        ctx = preprocess_hellaswag(example['ctx'])
        endings = [preprocess_hellaswag(e) for e in example['endings']]
        label = int(example['label'])

        # Tokenize context to find prompt length
        ctx_ids = tokenizer(ctx, return_tensors='pt')
        prompt_len = ctx_ids.input_ids.shape[1]

        best_score = float('-inf')
        best_idx = 0

        for j, ending in enumerate(endings):
            full_text = ctx + " " + ending
            full_ids = tokenizer(full_text, return_tensors='pt').input_ids.to(dev)

            # Truncate if too long
            if full_ids.shape[1] > model.seqlen:
                # Truncate from the left but keep at least some completion tokens
                overshoot = full_ids.shape[1] - model.seqlen
                full_ids = full_ids[:, overshoot:]
                adj_prompt_len = max(1, prompt_len - overshoot)
            else:
                adj_prompt_len = prompt_len

            total_ll, n_tokens = compute_completion_ll(model, full_ids, adj_prompt_len)

            # Length-normalize
            score = total_ll / n_tokens if n_tokens > 0 else float('-inf')

            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx == label:
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(dataset)}] Running accuracy: {correct/total:.4f}")

    accuracy = correct / total

    print(f"\nHellaSwag Results:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    save_result(save_title, results)
    print(f"  Results saved to {EVAL_SAVE_FILE}")

    model.cpu()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return accuracy
