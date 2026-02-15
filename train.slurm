#!/bin/bash
#SBATCH --job-name=train_classical_chinese
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# ============================================================================
# SLURM Training Script for Classical Chinese BiLSTM Model
# Using torchrun with 2 GPUs
# ============================================================================

# Note: removed 'set -e' because it prevents the error handling
# code (TRAIN_EXIT_CODE check) from executing when torchrun fails

echo "========================================================================"
echo "Job Information"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "========================================================================"

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.slurm"

# Create necessary directories
mkdir -p "${SAVE_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${SLURM_LOG_DIR}"

# Navigate to workspace
cd "${WORKSPACE}"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================
echo ""
echo "Setting up environment..."
echo "========================================================================"

# Load Anaconda
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
else
    echo "Error: Conda not found!"
    exit 1
fi

# Activate local conda environment
conda activate "${CONDA_ENV}"

echo "Conda environment: ${CONDA_ENV}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ============================================================================
# GPU INFORMATION
# ============================================================================
echo "GPU Information:"
echo "========================================================================"
nvidia-smi
echo ""

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
echo "Training Configuration:"
echo "========================================================================"
echo "Task: ${TASK}"
echo "Model: BiLSTM + $([ "$USE_CRF" = "true" ] && echo "CRF" || echo "Linear")"
echo "Number of GPUs: 2 (torchrun)"
echo "CPUs: ${SLURM_CPUS}"
echo "Memory: ${SLURM_MEM}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Max Length: ${MAX_LENGTH}"
echo "Warmup Steps: ${WARMUP_STEPS}"
echo "Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "AMP: ${USE_AMP}"
echo "Save Dir: ${SAVE_DIR}"
echo "Log Dir: ${LOG_DIR}"
if [ ! -z "$RESUME_CHECKPOINT" ]; then
    echo "Resume From: ${RESUME_CHECKPOINT}"
fi
echo "========================================================================"
echo ""

# ============================================================================
# BUILD TRAINING COMMAND
# ============================================================================

TRAIN_CMD="train.py \
    --task ${TASK} \
    --data_dir ${DATA_DIR} \
    --vocab_path ${VOCAB_PATH} \
    --train_split ${TRAIN_SPLIT} \
    --val_split ${VAL_SPLIT} \
    --test_split ${TEST_SPLIT} \
    --shuffle_buffer ${SHUFFLE_BUFFER} \
    --embedding_dim ${EMBEDDING_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_layers ${NUM_LAYERS} \
    --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --gradient_clip ${GRADIENT_CLIP} \
    --warmup_steps ${WARMUP_STEPS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --max_length ${MAX_LENGTH} \
    --save_every_n_steps ${SAVE_EVERY_N_STEPS} \
    --val_max_samples ${VAL_MAX_SAMPLES} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}"

if [ "$USE_CRF" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_crf"
fi

if [ "$USE_AMP" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_amp"
fi

if [ ! -z "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume ${RESUME_CHECKPOINT}"
    echo "Resuming training from: ${RESUME_CHECKPOINT}"
fi

# ============================================================================
# RUN TRAINING WITH TORCHRUN (2 GPUs)
# ============================================================================
echo "Starting distributed training with torchrun on 2 GPUs..."
echo "========================================================================"

export MASTER_ADDR=localhost
export MASTER_PORT=12355

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    ${TRAIN_CMD}

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    conda deactivate
    exit $TRAIN_EXIT_CODE
fi
echo "========================================================================"

# ============================================================================
# RUN EVALUATION WITH TORCHRUN (2 GPUs)
# ============================================================================
BEST_MODEL="${SAVE_DIR}/best_model.pt"

if [ -f "$BEST_MODEL" ]; then
    echo ""
    echo "========================================================================"
    echo "Running evaluation on best model with torchrun..."
    echo "========================================================================"
    
    export MASTER_PORT=12356  # Different port
    
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=2 \
        evaluate.py \
        --checkpoint ${BEST_MODEL} \
        --data_dir ${DATA_DIR} \
        --vocab_path ${VOCAB_PATH} \
        --test_split ${TEST_SPLIT} \
        --num_samples ${NUM_SAMPLES} \
        --output_dir "${WORKSPACE}/evaluation_results/${TASK}" \
        --seed ${SEED}
    
    EVAL_EXIT_CODE=$?
    
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        echo "Evaluation completed successfully!"
    else
        echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
    fi
else
    echo "Warning: Best model not found at ${BEST_MODEL}"
fi

# ============================================================================
# JOB COMPLETION
# ============================================================================
echo ""
echo "========================================================================"
echo "Job completed!"
echo "End Time: $(date)"
echo "Results saved to: ${SAVE_DIR}"
echo "Logs saved to: ${LOG_DIR}"
echo "========================================================================"

conda deactivate
