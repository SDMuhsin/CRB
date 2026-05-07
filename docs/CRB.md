This repository houses code for the paper titled "Coupled Residual Binarization with Two-Way Sign Refinement : Advancing LLM compression", currently under review at TNNLS.


# Abstract
iWe introduce \emph{Coupled Residual Binarization} (CRB), a post‑training quantization technique that pushes large language models (LLMs) to a near 1-bit precision while preserving perplexity competitive with 3‑bit methods. Starting from the BiLLM framework, CRB couples the two binary residual expansions found in salient weight regions and solves for their scale factors in closed form via a Tikhonov‑regularized coordinate‑descent scheme. A correlation‑damping term and a lightweight sign‑refinement loop further reduce reconstruction error with negligible calibration cost. On six OPT models ranging from 1.3B to 66B parameters, CRB lowers perplexity on WikiText‑2 and Penn Treebank (PTB) by up to \textbf{23\%} over BiLLM at identical storage cost, and narrows the gap to 3‑bit SOTA PTQ schemes to within \textbf{5\%} while requiring \(2.7\times\) less memory. These results establish a new state of the art for binary post‑training quantization of LLMs and highlight the practical value of joint, closed‑form optimization in ultra‑low‑bit regimes.


# Dependencies

```
torch~=2.5.1
transformers==4.35.0
datasets==2.14.6
numpy
sentencepiece
exceptiongroup
matplotlib
```


# Steps to reproduce work

The commands to run quantization is 
``` 
python3 run.py $model $dataset $technique --blocksize 128 --salient_metric hessian --device=\"cuda\" 
```

Models supported currently : facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b facebook/opt-30b facebook/opt-66b
Techniques
- CRB : Our complete technique
- CRBOG : Base version of our technique without two way sign refinement and stabilization
- braq : BiLLM (baseline)
For remaining baselines, refer to the GPTQ and PB-LLM directories.
Datasets
- PTB
- wikitext2

# Related projects

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[PB-LLM: Partially Binarized Large Language Models](https://github.com/hahnyuan/PB-LLM)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

[BiLLM : Pushing the Limit of Post-Training Quantization for LLMs](https://github.com/Aaronhuang-778/BiLLM)


# Citation

A preprint is not yet available. Reviewers may obtain additional information through the venue's reviewer-author communication channel.

