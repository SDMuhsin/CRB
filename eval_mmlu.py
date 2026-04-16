import torch
import torch.nn as nn
import json
import os
import fcntl
from datasets import load_dataset


EVAL_SAVE_FILE = "./output/GLOBAL_MMLU.json"


def save_result(save_title, results):
    """Save MMLU results to JSON file with file locking."""
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


def format_mmlu_question(question, choices):
    """Format a single MMLU question with lettered choices."""
    letters = ['A', 'B', 'C', 'D']
    formatted = question + "\n"
    for letter, choice in zip(letters, choices):
        formatted += f"{letter}. {choice}\n"
    formatted += "Answer:"
    return formatted


def format_mmlu_prompt(subject, few_shot_examples, question, choices):
    """Format full MMLU prompt with few-shot examples."""
    subject_name = subject.replace('_', ' ')
    prompt = f"The following are multiple choice questions (with answers) about {subject_name}.\n\n"

    letters = ['A', 'B', 'C', 'D']
    for ex in few_shot_examples:
        prompt += format_mmlu_question(ex['question'], ex['choices'])
        prompt += f" {letters[ex['answer']]}\n\n"

    prompt += format_mmlu_question(question, choices)
    return prompt


@torch.no_grad()
def eval_mmlu(model, model_name, dev, save_title='UNNAMED_MMLU'):
    """
    Evaluate a model on MMLU (Massive Multitask Language Understanding).

    5-shot evaluation across all 57 subjects. For each question, computes
    the log-probability of each answer letter (A/B/C/D) at the last position
    and picks the highest.
    """
    print("Evaluating on MMLU (5-shot) ...")

    from datautils import get_tokenizer
    tokenizer = get_tokenizer(model_name)

    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", "all", cache_dir="./downloads/datasets")
    test_data = dataset['test']
    # Few-shot examples come from validation split
    try:
        dev_data = dataset['validation']
    except KeyError:
        dev_data = dataset['dev']

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(dev)

    # Get token IDs for answer letters (with space prefix)
    answer_tokens = []
    for letter in ['A', 'B', 'C', 'D']:
        token_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        answer_tokens.append(token_ids[-1])

    # Group dev examples by subject for few-shot
    dev_by_subject = {}
    for ex in dev_data:
        subj = ex['subject']
        if subj not in dev_by_subject:
            dev_by_subject[subj] = []
        dev_by_subject[subj].append(ex)

    # Evaluate
    subject_correct = {}
    subject_total = {}
    total_correct = 0
    total = 0

    for i, example in enumerate(test_data):
        subject = example['subject']
        question = example['question']
        choices = example['choices']
        answer = example['answer']  # int 0-3

        # Try with 5-shot, reduce if prompt is too long
        few_shot_pool = dev_by_subject.get(subject, [])
        input_ids = None
        for n_shots in range(min(5, len(few_shot_pool)), -1, -1):
            few_shot = few_shot_pool[:n_shots]
            prompt = format_mmlu_prompt(subject, few_shot, question, choices)
            ids = tokenizer.encode(prompt, return_tensors='pt')
            if ids.shape[1] <= model.seqlen:
                input_ids = ids.to(dev)
                break

        if input_ids is None:
            # Even 0-shot is too long; truncate from the left
            prompt = format_mmlu_prompt(subject, [], question, choices)
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            input_ids = input_ids[:, -model.seqlen:].to(dev)

        # Forward pass
        logits = model(input_ids).logits[0, -1, :]  # [vocab_size]

        # Score answer letters
        scores = logits[answer_tokens]
        prediction = scores.argmax().item()

        correct = (prediction == answer)
        total_correct += correct
        total += 1

        if subject not in subject_correct:
            subject_correct[subject] = 0
            subject_total[subject] = 0
        subject_correct[subject] += correct
        subject_total[subject] += 1

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(test_data)}] Running accuracy: {total_correct/total:.4f}")

    overall_accuracy = total_correct / total

    # Per-subject accuracy
    subject_accuracies = {}
    for subj in sorted(subject_correct.keys()):
        subject_accuracies[subj] = subject_correct[subj] / subject_total[subj]

    print(f"\nMMLU Results:")
    print(f"  Overall accuracy: {overall_accuracy:.4f} ({total_correct}/{total})")
    print(f"  Subjects evaluated: {len(subject_accuracies)}")

    results = {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total": total,
        "subject_accuracies": subject_accuracies,
    }

    save_result(save_title, results)
    print(f"  Results saved to {EVAL_SAVE_FILE}")

    model.cpu()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return overall_accuracy
