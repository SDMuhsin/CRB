import torch
import torch.nn as nn
import json
import os
import fcntl
from datasets import load_dataset


EVAL_SAVE_FILE = "./output/GLOBAL_ARC.json"


def save_result(save_title, results):
    """Save ARC results to JSON file with file locking."""
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


def format_arc_question(question, choices_text):
    """Format an ARC question with its choices."""
    prompt = f"Question: {question}\nAnswer:"
    return prompt


@torch.no_grad()
def eval_arc(model, model_name, dev, save_title='UNNAMED_ARC'):
    """
    Evaluate a model on ARC (AI2 Reasoning Challenge).

    0-shot evaluation on both ARC-Easy and ARC-Challenge. For each question,
    computes the length-normalized log-likelihood of each answer choice and
    picks the highest.
    """
    print("Evaluating on ARC (0-shot) ...")

    from datautils import get_tokenizer
    tokenizer = get_tokenizer(model_name)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(dev)

    all_results = {}

    for split_name in ["ARC-Easy", "ARC-Challenge"]:
        print(f"\n  Evaluating {split_name} ...")
        dataset = load_dataset("allenai/ai2_arc", split_name, split="test",
                               cache_dir="./downloads/datasets")

        correct = 0
        total = 0

        for i, example in enumerate(dataset):
            question = example['question']
            choices_text = example['choices']['text']
            choices_labels = example['choices']['label']
            answer_key = example['answerKey']

            # Find ground truth index
            try:
                gt_idx = choices_labels.index(answer_key)
            except ValueError:
                # answerKey might be numeric (1,2,3,4) instead of letter (A,B,C,D)
                letter_map = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}
                mapped = letter_map.get(answer_key, answer_key)
                try:
                    gt_idx = choices_labels.index(mapped)
                except ValueError:
                    continue

            # Format prompt
            prompt = f"Question: {question}\nAnswer:"
            prompt_ids = tokenizer(prompt, return_tensors='pt')
            prompt_len = prompt_ids.input_ids.shape[1]

            best_score = float('-inf')
            best_idx = 0

            for j, choice_text in enumerate(choices_text):
                full_text = prompt + " " + choice_text
                full_ids = tokenizer(full_text, return_tensors='pt').input_ids.to(dev)

                # Truncate if too long
                if full_ids.shape[1] > model.seqlen:
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

            if best_idx == gt_idx:
                correct += 1
            total += 1

            if (i + 1) % 500 == 0:
                print(f"    [{i+1}/{len(dataset)}] Running accuracy: {correct/total:.4f}")

        accuracy = correct / total
        print(f"  {split_name} accuracy: {accuracy:.4f} ({correct}/{total})")
        all_results[split_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    print(f"\nARC Results:")
    for split_name, res in all_results.items():
        print(f"  {split_name}: {res['accuracy']:.4f}")

    save_result(save_title, all_results)
    print(f"  Results saved to {EVAL_SAVE_FILE}")

    model.cpu()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return all_results
