import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
import os


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

downloads_dir = os.environ.get("BILLM_DOWNLOADS_DIR", "./downloads")

'''
Generate tokenizer and return it to preload datasets by converting them to embedded vectors instead of natural words
'''
def get_tokenizer(model_name):
    # Load from HuggingFace's own cache under $BILLM_DOWNLOADS_DIR.
    # Do NOT pickle the tokenizer — pickled tokenizers are bound to the
    # exact transformers version that wrote them and break cryptically
    # after any upgrade (e.g. wheelhouse transformers 4.42 -> PyPI 4.51).
    if "llama" in model_name.lower() or "danube" in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name, use_fast=False, cache_dir=downloads_dir,
        )
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    elif any(k in model_name.lower() for k in ("smollm", "pythia", "qwen", "bloom", "granite")):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=downloads_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, cache_dir=downloads_dir,
        )
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset(
        'c4', 'en', split='train', streaming=True
    )
    valdata = load_dataset(
        'c4', 'en', split='validation', streaming=True
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            sample = next(iter(traindata))
            trainenc = tokenizer(sample['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join([next(iter(valdata))['text'] for _ in range(1100)]), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_c4_old(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_redpajama(nsamples, seed, seqlen, model, tokenizer):
    # RedPajama-Data-1T-Sample was removed from HuggingFace. Use C4 (same type
    # of web crawl data) with concatenation, following GuidedQuant's approach.
    print("RedPajama unavailable on HF; using C4 web corpus as substitute...")
    data = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train',
    )

    target_tokens = nsamples * seqlen * 3
    max_docs = min(len(data), target_tokens // 150)
    print(f"  Tokenizing {max_docs} C4 documents...")
    trainenc = tokenizer("\n\n".join(data['text'][:max_docs]), return_tensors='pt')
    total_tokens = trainenc.input_ids.shape[1]
    print(f"  {total_tokens:,} tokens from {max_docs} documents")

    if total_tokens < nsamples * seqlen:
        raise RuntimeError(
            f"Not enough tokens: have {total_tokens:,}, need {nsamples * seqlen:,}"
        )

    random.seed(seed)
    trainloader = []
    selected = set()
    while len(trainloader) < nsamples:
        idx = random.randint(0, total_tokens - seqlen - 1)
        if selected:
            closest = min(selected, key=lambda x: abs(x - idx))
            if abs(idx - closest) < seqlen:
                continue
        inp = trainenc.input_ids[:, idx:idx + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        selected.add(idx)
        trainloader.append((inp, tar))
        if len(trainloader) % 100 == 0:
            print(f"  Sampled {len(trainloader)}/{nsamples}")

    print(f"  Done: {nsamples} non-overlapping samples of seqlen {seqlen}")
    return trainloader, None


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    cache_file=f'{downloads_dir}/DOWNLOAD_{name}_{nsamples}_{seed}_{seqlen}_{model}.pt'
    try:
        return torch.load(cache_file)
    except:
        pass

    # Concurrent benchmark jobs hitting the same cache miss would each spend
    # GBs tokenizing in parallel and racing to torch.save the same file.
    # Serialise on a sibling .lock so only one job builds the cache; the rest
    # block, then load the freshly written cache when their turn comes.
    import fcntl
    directory = os.path.dirname(cache_file) or '.'
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    lock_path = cache_file + '.lock'
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        # Re-check after acquiring lock — another job may have just finished.
        try:
            return torch.load(cache_file)
        except Exception:
            pass

        tokenizer = get_tokenizer(model)
        if 'wikitext2' in name:
            loaders = get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
        if 'ptb' in name:
            loaders = get_ptb(nsamples, seed, seqlen, model, tokenizer)
        if 'c4' in name:
            loaders = get_c4(nsamples, seed, seqlen, model, tokenizer)
        if 'redpajama' in name:
            loaders = get_redpajama(nsamples, seed, seqlen, model, tokenizer)

        torch.save(loaders, cache_file)
        return loaders
