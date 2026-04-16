#!/bin/bash
# Phase 5 experiments: per-sublayer order optimization
# Partition search uses original (1,1,2) evaluation (reverted order-aware search)
# Run sequentially on GPU 0

source env/bin/activate
MODEL="Qwen/Qwen3-0.6B"
BASE_ARGS="$MODEL wikitext2 mixed --blocksize 128 --salient_metric magnitude --device=cuda:0 --seed 0"

# 1. Uniform (2,2,2) with ORIGINAL partition — most promising MLP boost
echo "=== Exp 4b: uniform (2,2,2) — original partition (2.0 bpw) ==="
python3 -u run.py $BASE_ARGS --attn_order 2 --mlp_orders 2,2,2 2>&1 | tee /tmp/exp_uniform2_origpart.log
echo ""

# 2. Gate/up boost: attn=3, gate/up=(2,2,2), down=(1,1,2) — SwiGLU source fix (~2.02 bpw)
echo "=== Exp 6: attn=3, gate/up (2,2,2), down (1,1,2) (~2.02 bpw) ==="
python3 -u run.py $BASE_ARGS --attn_order 3 --gate_up_orders 2,2,2 2>&1 | tee /tmp/exp_gateup2.log
echo ""

# 3. Order=4 — test if regression was truly partition-related
echo "=== Exp 2: order=4 attn, MLP (1,1,2) (~1.80 bpw) ==="
python3 -u run.py $BASE_ARGS --attn_order 4 2>&1 | tee /tmp/exp_order4_origpart.log
echo ""

# 4. Order=5 — max attn within budget (~2.05 bpw)
echo "=== Exp 3: order=5 attn, MLP (1,1,2) (~2.05 bpw) ==="
python3 -u run.py $BASE_ARGS --attn_order 5 2>&1 | tee /tmp/exp_order5_origpart.log
echo ""

# 5. KV=5, QO=3 + gate/up=(2,2,2) — combined optimization (~2.02 bpw)
echo "=== Exp 7: KV=5 QO=3 gate/up=(2,2,2) (~2.02 bpw) ==="
python3 -u run.py $BASE_ARGS --attn_order 3 --kv_order 5 --gate_up_orders 2,2,2 2>&1 | tee /tmp/exp_combined.log
echo ""

echo "=== All experiments complete ==="
