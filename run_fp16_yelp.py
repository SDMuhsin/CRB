"""Quick fp16 baseline: load model and run Yelp MRR eval directly (no quantization)."""
import torch
import os

os.environ.setdefault("TRANSFORMERS_CACHE", "./downloads")
os.environ.setdefault("HF_HOME", "./downloads")

from run import get_model
from eval_mrr_yelp import opt_eval_mrr_yelp

model_name = "facebook/opt-1.3b"
device = "cuda:0"

model = get_model(model_name)
model.eval()

save_title = f"{model_name}_wikitext2_fp16_128_magnitude_MRR_YELP"
opt_eval_mrr_yelp(model, model_name, device, save_title=save_title)
