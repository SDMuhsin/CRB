import torch
import torch.nn as nn
import json
import os
import fcntl
from datasets import load_dataset


EVAL_SAVE_FILE = "./output/GLOBAL_LAMBADA.json"


def save_lambada_result(save_title, accuracy):
    """Save LAMBADA accuracy result to JSON file with file locking."""
    os.makedirs(os.path.dirname(EVAL_SAVE_FILE), exist_ok=True)

    if not os.path.exists(EVAL_SAVE_FILE):
        with open(EVAL_SAVE_FILE, "w") as f:
            json.dump({}, f)

    with open(EVAL_SAVE_FILE, "r+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            data = json.load(f)
            data[save_title] = accuracy
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@torch.no_grad()
def opt_eval_lambada(model, model_name, dev, save_title='UNNAMED_OPT_LAMBADA'):
    """
    Evaluate an OPT model on LAMBADA last-word prediction accuracy.

    For each passage, the model must correctly predict all tokens of the
    final word (greedy argmax) given the preceding context.
    """
    print("Evaluating on LAMBADA ...")

    from datautils import get_tokenizer
    tokenizer = get_tokenizer(model_name)

    dataset = load_dataset("lambada", split="test")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(dev)

    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        text = example["text"]

        # Split into context and target (last word)
        last_space = text.rfind(" ")
        if last_space == -1:
            continue
        context = text[:last_space]

        # Tokenize full text and context to find boundary
        full_ids = tokenizer(text, return_tensors="pt").input_ids.to(dev)
        ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(dev)

        ctx_len = ctx_ids.shape[1]
        full_len = full_ids.shape[1]
        n_target = full_len - ctx_len

        if n_target <= 0:
            continue

        # Forward pass
        logits = model(full_ids).logits

        # logits[0, t] predicts token at position t+1
        target_tokens = full_ids[0, ctx_len:full_len]
        pred_tokens = logits[0, ctx_len - 1 : full_len - 1].argmax(dim=-1)

        if torch.all(pred_tokens == target_tokens):
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(dataset)}] Running accuracy: {correct}/{total} = {correct/total*100:.2f}%")

    accuracy = correct / total * 100
    print(f"LAMBADA Accuracy: {accuracy:.2f}% ({correct}/{total})")

    model.cpu()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache

    save_lambada_result(save_title, accuracy)

    return accuracy
