#!/bin/bash
# Phase 2: Empirical Landscape — Run all baselines
# 2 models × 4 methods × 3 seeds = 24 quantization runs
set -e

cd /workspace/BiLLM2
source env/bin/activate

RESULTS_DIR="results/phase2_landscape"
mkdir -p "$RESULTS_DIR"

COMMON_ARGS="--blocksize 128 --salient_metric magnitude --device=cuda:0"

MODELS=("Qwen/Qwen3-0.6B" "meta-llama/Llama-3.2-1B")
MODEL_TAGS=("qwen3_0.6b" "llama3.2_1b")
METHODS=("braq" "2bit" "4bit" "rtn")
SEEDS=(0 1 2)

TOTAL=$((${#MODELS[@]} * ${#METHODS[@]} * ${#SEEDS[@]}))
COUNT=0

for m_idx in "${!MODELS[@]}"; do
    MODEL="${MODELS[$m_idx]}"
    TAG="${MODEL_TAGS[$m_idx]}"
    for METHOD in "${METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            COUNT=$((COUNT + 1))
            LOGFILE="$RESULTS_DIR/${TAG}_${METHOD}_s${SEED}.log"

            # Skip if log already exists and contains a Perplexity line
            if grep -q "Perplexity:" "$LOGFILE" 2>/dev/null; then
                echo "[$COUNT/$TOTAL] SKIP (already done): $TAG $METHOD seed=$SEED"
                continue
            fi

            echo "========================================"
            echo "[$COUNT/$TOTAL] Running: $TAG $METHOD seed=$SEED"
            echo "  Model: $MODEL"
            echo "  Log: $LOGFILE"
            echo "  Started: $(date)"
            echo "========================================"

            python3 -u run.py "$MODEL" wikitext2 "$METHOD" \
                $COMMON_ARGS \
                --seed "$SEED" \
                2>&1 | tee "$LOGFILE"

            echo "  Finished: $(date)"
            echo ""
        done
    done
done

echo "========================================"
echo "ALL PHASE 2 BASELINES COMPLETE"
echo "========================================"
