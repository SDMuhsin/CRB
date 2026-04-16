#!/bin/bash
# Run all 4 quantized configs for Yelp MRR benchmark (sequential on GPU 0)
set -e

source env/bin/activate
export TRANSFORMERS_CACHE="./downloads" HF_HOME="./downloads" PYTHONPATH="$PYTHONPATH:$(pwd)"

echo "=== RUN 1/4: CRB + magnitude ==="
python3 -u run.py facebook/opt-1.3b wikitext2 crb --blocksize 128 --salient_metric magnitude --device="cuda:0" --skip_ppl_save --eval_mrr_yelp

echo "=== RUN 2/4: BRAQ + magnitude ==="
python3 -u run.py facebook/opt-1.3b wikitext2 braq --blocksize 128 --salient_metric magnitude --device="cuda:0" --skip_ppl_save --eval_mrr_yelp

echo "=== RUN 3/4: CRB + hessian ==="
python3 -u run.py facebook/opt-1.3b wikitext2 crb --blocksize 128 --salient_metric hessian --device="cuda:0" --skip_ppl_save --eval_mrr_yelp

echo "=== RUN 4/4: BRAQ + hessian ==="
python3 -u run.py facebook/opt-1.3b wikitext2 braq --blocksize 128 --salient_metric hessian --device="cuda:0" --skip_ppl_save --eval_mrr_yelp

echo "=== ALL RUNS COMPLETE ==="
cat ./output/GLOBAL_MRR_YELP.json
