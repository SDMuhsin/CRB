#!/bin/bash
# ============================================================================
# LLaMA-7B FlashFFN Benchmark Suite - SLURM Submission Script
# ============================================================================
#
# Submits Mo5 (median-of-five, seeds 41-45) benchmarks for PEFT methods with
# and without FlashFFN on LLaMA-7B (SwiGLU, FlashFFN-compatible).
#
# 8 PEFT baselines + 1 Full FT + 7 FlashFFN variants = 16 techniques
# PEFT targets attention only (q_proj, v_proj) → MLP frozen →
# FlashFFN uses activations mode (always wins, no breakeven).
# Full FT + FlashFFN excluded: MLP weights trainable → recompute mode → below breakeven.
#
# NOTE: LLaMA-7B requires significantly more GPU memory than TinyLlama.
# Gradient checkpointing is enabled by default. Batch sizes are reduced.
#
# Usage:
#   ./sbatch/run_llama7b.sh
#   ./sbatch/run_llama7b.sh --account def-myprof
#   ./sbatch/run_llama7b.sh --local    # Run locally (no SLURM)
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT=""
LOCAL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT] [--local]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL="huggyllama/llama-7b"
MODEL_SHORT="llama7b"
DTYPE="bfloat16"
# Attention-only PEFT targeting: q_proj,v_proj only.
# MLP layers (gate/up/down_proj) stay frozen → FlashFFN activations mode (always wins).
TARGET_MODULES="q_proj,v_proj"

# Techniques to benchmark (comment/uncomment as needed)
techniques=(
    "base"
    "lora"
    "dora"
    "adalora"
    "dylora"
    "vera"
    "fourierft"
    "spectral"
    # FlashFFN variants — all use activations mode (MLP frozen, no breakeven)
    # No base_flash: Full FT trains MLP weights → recompute mode → below breakeven
    "lora_flash"
    "dora_flash"
    "adalora_flash"
    "dylora_flash"
    "vera_flash"
    "fourierft_flash"
    "spectral_flash"
)

# Tasks to evaluate
tasks=(
    "cola"
    "mrpc"
    "sst2"
    "rte"
    "qnli"
    "stsb"
    "boolq"
    "cb"
    "wikitext2"
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
#
# LLaMA-7B: 32 layers, hidden_dim=4096, intermediate=11008
# 64 target modules (32 layers x {q_proj, v_proj})
# Gradient checkpointing enabled by default for memory.
#
# Hyperparameters extrapolated from TinyLlama validated configs.
# ============================================================================

# --- Shared across all techniques ---
BATCH_SIZE=4
EVAL_BATCH_SIZE=8
TOTAL_BATCH_SIZE=64
WEIGHT_DECAY=0.01
LR_SCHEDULER="linear"
GRAD_CLIP=1.0
MAX_LENGTH=128
PAD_TO_MAX="--pad_to_max_length"
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# --- WikiText-2 overrides ---
WT2_BATCH_SIZE=2
WT2_MAX_LENGTH=512

# --- Full fine-tuning ---
BASE_LR="2e-5"
BASE_EPOCHS=3

# --- PEFT epochs ---
PEFT_EPOCHS=10

# --- WikiText-2 epochs (all methods) ---
WT2_EPOCHS=1

# --- LoRA (PEFT library, r=8) ---
LORA_LR="5e-4"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.0

# --- DoRA (PEFT library, r=8) ---
DORA_LR="5e-4"
DORA_R=8
DORA_ALPHA=16
DORA_DROPOUT=0.05

# --- AdaLoRA (PEFT library) ---
ADALORA_LR="5e-4"
ADALORA_INIT_R=8
ADALORA_TARGET_R=4
ADALORA_ALPHA=16

# --- DyLoRA (custom) ---
DYLORA_LR="5e-4"
DYLORA_R=8
DYLORA_ALPHA=16

# --- VeRA (PEFT library, r=256) ---
VERA_LR="1e-3"
VERA_R=256
VERA_D_INITIAL=0.1
VERA_DROPOUT=0.0

# --- FourierFT (PEFT library) ---
# NOTE: Not yet tuned for LLaMA-7B. BERT defaults as starting point.
# Effective scale = scaling / d^2. For 4096-dim: may need scaling ~4000.
FOURIERFT_LR="5e-2"
FOURIERFT_N=256
FOURIERFT_SCALING=150.0

# --- Spectral Adapter ---
# NOTE: Not yet tuned for LLaMA-7B. Using BERT defaults as starting point.
SPECTRAL_LR="2e-2"
SPECTRAL_SCALING=1.0
SPECTRAL_DROPOUT=0.0
SPECTRAL_FREQ_MODE="contiguous"
SPECTRAL_DENSE_P=16
SPECTRAL_DENSE_Q=16
SPECTRAL_DENSE_D_INITIAL=0.01
SPECTRAL_RTE_D_INITIAL=0.07
SPECTRAL_RTE_SCALING=2.0
SPECTRAL_FACTORED_P=32
SPECTRAL_FACTORED_Q=32
SPECTRAL_FACTORED_D_INITIAL=0.07
SPECTRAL_FACTORED_RANK=4
SPECTRAL_FACTORED_LEARN_TASKS="cola cb"
SPECTRAL_FACTORED_TASKS="sst2 qnli"

# --- FlashFFN ---
FLASH_FFN_K_FRACTION=0.3

# ============================================================================
# END CONFIGURATION
# ============================================================================

job_count=0
mkdir -p ./logs ./results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

is_wikitext_task() {
    case $1 in
        wikitext2|wikitext103) return 0 ;;
        *) return 1 ;;
    esac
}

is_flash_technique() {
    [[ "$1" == *"_flash" ]]
}

get_base_technique() {
    echo "${1%_flash}"
}

get_epochs() {
    local technique=$1
    local task=$2
    if is_wikitext_task "$task"; then
        echo "$WT2_EPOCHS"
    elif [[ "$(get_base_technique "$technique")" == "base" ]]; then
        echo "$BASE_EPOCHS"
    else
        echo "$PEFT_EPOCHS"
    fi
}

get_job_resources() {
    # Sets: gpu_type, gpu_mem
    # LLaMA-7B (14GB model weight in bf16) needs a full GPU.
    local technique=$1
    local base_tech=$(get_base_technique "$technique")

    gpu_type="h100:1"
    if [[ "$base_tech" == "base" ]]; then
        gpu_mem="64000M"
    else
        gpu_mem="40000M"
    fi
}

get_time_limit() {
    # Returns SLURM time string.
    # Mo5 (5 seeds) on full H100 (80GB).
    # LLaMA-7B is ~6x bigger than TinyLlama but runs on full GPU (~3x more
    # compute than MIG 3g.40gb). Net: ~2x TinyLlama's calibrated estimates.
    # Same effective batch size (64) via batch=4, grad_accum=16.
    local technique=$1
    local task=$2
    local minutes=0

    local base_tech=$(get_base_technique "$technique")

    if is_wikitext_task "$task"; then
        if [[ "$base_tech" == "base" ]]; then
            minutes=960
        else
            minutes=480
        fi
    elif [[ "$base_tech" == "base" ]]; then
        # Full FT: 3 epochs, 5 seeds
        case $task in
            cb)                minutes=60    ;;
            rte)               minutes=240   ;;
            mrpc)              minutes=360   ;;
            stsb)              minutes=600   ;;
            cola)              minutes=720   ;;
            boolq)             minutes=960   ;;
            sst2)              minutes=5760  ;;
            qnli)              minutes=8640  ;;
        esac
    else
        # PEFT: 10 epochs, 5 seeds (attention-only, faster backward)
        case $task in
            cb)            minutes=60    ;;
            rte)           minutes=480   ;;
            mrpc)          minutes=720   ;;
            stsb)          minutes=1080  ;;
            cola)          minutes=1440  ;;
            boolq)         minutes=1680  ;;
            sst2)          minutes=10080 ;;
            qnli)          minutes=10080 ;;
        esac
    fi

    # FlashFFN activations mode has minimal overhead (no top-K)
    # No time adjustment needed for LLaMA-7B flash techniques

    # Format as D-HH:MM:SS or H:MM:SS
    local hours=$((minutes / 60))
    local mins=$((minutes % 60))
    if [[ $hours -ge 24 ]]; then
        local days=$((hours / 24))
        hours=$((hours % 24))
        printf "%d-%02d:%02d:00" "$days" "$hours" "$mins"
    else
        printf "%d:%02d:00" "$hours" "$mins"
    fi
}

build_python_cmd() {
    local technique=$1
    local task=$2
    local epochs=$3
    local run_name=$4

    local base_tech=$(get_base_technique "$technique")
    local flash=""
    if is_flash_technique "$technique"; then
        flash=" --flash_ffn --flash_ffn_k_fraction $FLASH_FFN_K_FRACTION"
    fi

    # Base command
    local common="python src/train_glue.py"
    common+=" --model_name_or_path $MODEL"
    common+=" --task_name $task"
    common+=" --num_train_epochs $epochs"
    common+=" --total_batch_size $TOTAL_BATCH_SIZE"
    common+=" --weight_decay $WEIGHT_DECAY"
    common+=" --lr_scheduler_type $LR_SCHEDULER"
    common+=" --grad_clipping $GRAD_CLIP"
    common+=" --dtype $DTYPE"
    common+=" --name $run_name"
    common+=" $GRADIENT_CHECKPOINTING"

    # Attention-only PEFT targeting (skip for Full FT which has no adapters)
    if [[ "$base_tech" != "base" ]]; then
        common+=" --adapter_target_modules $TARGET_MODULES"
    fi

    # Task-specific overrides
    if is_wikitext_task "$task"; then
        common+=" --per_device_train_batch_size $WT2_BATCH_SIZE"
        common+=" --per_device_eval_batch_size $WT2_BATCH_SIZE"
        common+=" --max_length $WT2_MAX_LENGTH"
    else
        common+=" --per_device_train_batch_size $BATCH_SIZE"
        common+=" --per_device_eval_batch_size $EVAL_BATCH_SIZE"
        common+=" --max_length $MAX_LENGTH"
        common+=" $PAD_TO_MAX"
    fi

    case $base_tech in
        base)
            echo "$common --optimizer adamw --learning_rate $BASE_LR$flash"
            ;;
        lora)
            echo "$common --optimizer adamw-lora --learning_rate $LORA_LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT$flash"
            ;;
        dora)
            echo "$common --optimizer adamw-dora --learning_rate $DORA_LR --dora_r $DORA_R --dora_alpha $DORA_ALPHA --dora_dropout $DORA_DROPOUT$flash"
            ;;
        adalora)
            echo "$common --optimizer adamw-adalora --learning_rate $ADALORA_LR --adalora_init_r $ADALORA_INIT_R --adalora_target_r $ADALORA_TARGET_R --adalora_alpha $ADALORA_ALPHA$flash"
            ;;
        dylora)
            echo "$common --optimizer adamw-dylora --learning_rate $DYLORA_LR --dylora_r $DYLORA_R --dylora_alpha $DYLORA_ALPHA$flash"
            ;;
        vera)
            echo "$common --optimizer adamw-vera --learning_rate $VERA_LR --vera_r $VERA_R --vera_d_initial $VERA_D_INITIAL --vera_dropout $VERA_DROPOUT$flash"
            ;;
        fourierft)
            echo "$common --optimizer adamw-fourierft --learning_rate $FOURIERFT_LR --fourierft_n_frequency $FOURIERFT_N --fourierft_scaling $FOURIERFT_SCALING$flash"
            ;;
        spectral)
            local cmd="$common --optimizer adamw-spectral --learning_rate $SPECTRAL_LR --spectral_dropout $SPECTRAL_DROPOUT --spectral_freq_mode $SPECTRAL_FREQ_MODE"
            if [[ " $SPECTRAL_FACTORED_LEARN_TASKS " == *" $task "* ]]; then
                cmd+=" --spectral_p $SPECTRAL_FACTORED_P --spectral_q $SPECTRAL_FACTORED_Q"
                cmd+=" --spectral_d_initial $SPECTRAL_FACTORED_D_INITIAL"
                cmd+=" --spectral_factored_rank $SPECTRAL_FACTORED_RANK"
                cmd+=" --spectral_scaling $SPECTRAL_SCALING"
                cmd+=" --spectral_learn_scaling"
            elif [[ " $SPECTRAL_FACTORED_TASKS " == *" $task "* ]]; then
                cmd+=" --spectral_p $SPECTRAL_FACTORED_P --spectral_q $SPECTRAL_FACTORED_Q"
                cmd+=" --spectral_d_initial $SPECTRAL_FACTORED_D_INITIAL"
                cmd+=" --spectral_factored_rank $SPECTRAL_FACTORED_RANK"
                cmd+=" --spectral_scaling $SPECTRAL_SCALING"
            elif [[ "$task" == "rte" ]]; then
                cmd+=" --spectral_p $SPECTRAL_DENSE_P --spectral_q $SPECTRAL_DENSE_Q"
                cmd+=" --spectral_d_initial $SPECTRAL_RTE_D_INITIAL"
                cmd+=" --spectral_scaling $SPECTRAL_RTE_SCALING"
            else
                cmd+=" --spectral_p $SPECTRAL_DENSE_P --spectral_q $SPECTRAL_DENSE_Q"
                cmd+=" --spectral_d_initial $SPECTRAL_DENSE_D_INITIAL"
                cmd+=" --spectral_scaling $SPECTRAL_SCALING"
            fi
            echo "$cmd$flash"
            ;;
    esac
}

get_technique_desc() {
    local tech=$1
    local base_tech=$(get_base_technique "$tech")
    local flash_suffix=""
    if is_flash_technique "$tech"; then
        flash_suffix=" + FlashFFN(k=$FLASH_FFN_K_FRACTION)"
    fi
    case $base_tech in
        base)      echo "Full FT (lr=$BASE_LR)${flash_suffix}" ;;
        lora)      echo "LoRA (r=$LORA_R, a=$LORA_ALPHA, lr=$LORA_LR)${flash_suffix}" ;;
        dora)      echo "DoRA (r=$DORA_R, a=$DORA_ALPHA, lr=$DORA_LR)${flash_suffix}" ;;
        adalora)   echo "AdaLoRA (r=$ADALORA_INIT_R→$ADALORA_TARGET_R, a=$ADALORA_ALPHA, lr=$ADALORA_LR)${flash_suffix}" ;;
        dylora)    echo "DyLoRA (r=$DYLORA_R, a=$DYLORA_ALPHA, lr=$DYLORA_LR)${flash_suffix}" ;;
        vera)      echo "VeRA (r=$VERA_R, d=$VERA_D_INITIAL, lr=$VERA_LR)${flash_suffix}" ;;
        fourierft) echo "FourierFT (n=$FOURIERFT_N, s=$FOURIERFT_SCALING, lr=$FOURIERFT_LR)${flash_suffix}" ;;
        spectral)  echo "Spectral (per-task config, lr=$SPECTRAL_LR)${flash_suffix}" ;;
    esac
}

# ============================================================================
# MAIN LOOP
# ============================================================================

echo "============================================"
echo "LLaMA-7B FlashFFN Benchmark Suite"
echo "============================================"
echo "Model:      $MODEL"
echo "Techniques: ${techniques[*]}"
echo "Tasks:      ${tasks[*]}"
echo "Dtype:      $DTYPE"
echo "Target:     $TARGET_MODULES (attention-only, activations mode)"
echo "Grad ckpt:  enabled"
echo "============================================"
echo ""

for technique in "${techniques[@]}"; do
    technique_desc=$(get_technique_desc "$technique")

    for task in "${tasks[@]}"; do
        epochs=$(get_epochs "$technique" "$task")
        time_limit=$(get_time_limit "$technique" "$task")
        job_name="${MODEL_SHORT}_${technique}_${task}"
        run_name="${technique}_${MODEL_SHORT}_${task}"

        get_job_resources "$technique" "$task"

        python_cmd=$(build_python_cmd "$technique" "$task" "$epochs" "$run_name")

        if [[ "$LOCAL_MODE" == true ]]; then
            echo "========================================"
            echo "Running locally: $job_name"
            echo "Config: $technique_desc"
            echo "Command: $python_cmd"
            echo "========================================"
            eval "$python_cmd"
            ((job_count++))
            continue
        fi

        account_line=""
        if [[ -n "$ACCOUNT" ]]; then
            account_line="#SBATCH --account=$ACCOUNT"
        fi

        sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=./logs/${job_name}_%j.out
#SBATCH --error=./logs/${job_name}_%j.err
#SBATCH --time=$time_limit
#SBATCH --gpus=$gpu_type
#SBATCH --mem=$gpu_mem
#SBATCH --cpus-per-task=4
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

export HF_HOME=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p \$HF_HOME

echo '========================================'
echo "Job: $job_name"
echo "Model: $MODEL"
echo "Technique: $technique"
echo "Config: $technique_desc"
echo "Task: $task"
echo "Epochs: $epochs"
echo "Dtype: $DTYPE"
echo "Time limit: $time_limit"
echo "Cache: \$HF_HOME"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi
export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src
$python_cmd
echo '========================================'
echo "Finished: \$(date)"
echo '========================================'
EOF
)
        echo "  [$sbatch_id] $job_name  ($technique_desc, ${task}, ${epochs}ep, ${time_limit})"
        ((job_count++))
    done
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Results CSV:          ./results/mo53_glue.csv"
echo "Logs directory:       ./logs/"
echo "============================================"
