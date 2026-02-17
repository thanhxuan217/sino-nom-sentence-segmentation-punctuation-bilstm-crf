#!/bin/bash
#SBATCH --job-name=lstm-seg
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
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
source "config.slurm"

# Create necessary directories
mkdir -p "${SAVE_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${SLURM_LOG_DIR}"

# Activate conda environment
echo "Activating conda environment..."
source /media02/ddien02/datnd/miniconda3/etc/profile.d/conda.sh
conda activate /media02/ddien02/thanhxuan217/envs/bilstm_crf

echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

# Print GPU information
echo ""
echo "GPU Information:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    echo "nvidia-smi not found"
fi
echo ""

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
echo "Training Configuration:"
echo "========================================================================"
echo "Task: ${TASK}"
echo "Model: BiLSTM + $([ "$USE_CRF" = "true" ] && echo "CRF" || echo "Linear")"
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
    --seed ${SEED} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE}"

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

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
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
