import torch
import torch.nn as nn
import json
import os
import fcntl
from datasets import load_dataset


EVAL_SAVE_FILE = "./output/GLOBAL_MRR.json"


def save_mrr_result(save_title, results):
    """Save MRR results to JSON file with file locking."""
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


@torch.no_grad()
def opt_eval_mrr(model, model_name, dev, save_title='UNNAMED_OPT_MRR'):
    """
    Evaluate an OPT model using Mean Reciprocal Rank (MRR) on PTB test set.

    For each next-token prediction position, computes the reciprocal of the rank
    of the correct token in the model's output distribution. MRR is the mean
    of these reciprocal ranks across all positions.

    MRR captures how close the model is to the correct answer even when
    the top-1 prediction is wrong -- unlike accuracy (binary) or PPL (log-prob).
    """
    print("Evaluating MRR on PTB test set ...")

    from datautils import get_tokenizer
    tokenizer = get_tokenizer(model_name)

    # Load PTB test set
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
    test_ids = testenc.input_ids  # [1, total_tokens]

    seqlen = model.seqlen
    nsamples = test_ids.numel() // seqlen

    print(f"  PTB test set: {test_ids.numel()} tokens, {nsamples} chunks of {seqlen}")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(dev)

    reciprocal_ranks_sum = 0.0
    total_positions = 0

    for i in range(nsamples):
        input_ids = test_ids[:, i * seqlen : (i + 1) * seqlen].to(dev)

        logits = model(input_ids).logits  # [1, seqlen, vocab_size]

        # shift: logits[0, :-1] predicts tokens at positions [1:]
        shift_logits = logits[0, :-1, :]  # [seqlen-1, vocab_size]
        shift_labels = input_ids[0, 1:]   # [seqlen-1]

        # Get the logit value for each correct token
        correct_logits = shift_logits[
            torch.arange(shift_logits.size(0), device=dev), shift_labels
        ]  # [seqlen-1]

        # Rank = number of tokens with strictly higher logit + 1
        ranks = (shift_logits > correct_logits.unsqueeze(1)).sum(dim=1).float() + 1.0

        reciprocal_ranks = 1.0 / ranks
        reciprocal_ranks_sum += reciprocal_ranks.sum().item()
        total_positions += ranks.numel()

        if (i + 1) % 5 == 0 or i == 0:
            running_mrr = reciprocal_ranks_sum / total_positions
            print(f"  [chunk {i+1}/{nsamples}] Running MRR: {running_mrr:.6f}")

    mrr = reciprocal_ranks_sum / total_positions
    print(f"MRR on PTB: {mrr:.6f} ({total_positions} positions)")

    model.cpu()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache

    save_mrr_result(save_title, mrr)

    return mrr
