#!/bin/bash
# config.slurm - Configuration for SLURM jobs

# ============================================================================
# WORKSPACE & ENVIRONMENT
# ============================================================================
export WORKSPACE="/media02/ddien02/thanhxuan217/main_src"
export CONDA_ENV="${WORKSPACE}/envs/classical_chinese"  # Local conda environment

# ============================================================================
# SLURM RESOURCE LIMITS (FIXED - DO NOT CHANGE)
# ============================================================================
export SLURM_CPUS=16
export SLURM_MEM="64G"
export SLURM_GPUS=2          # Fixed: Always use 2 GPUs
export SLURM_TIME="48:00:00"  # 48 hours max
export SLURM_PARTITION="gpu"

# ============================================================================
# TASK CONFIGURATION
# ============================================================================
export TASK="segmentation"  # "segmentation" hoặc "punctuation"
export USE_CRF="true"       # true để dùng CRF, false để dùng Linear

# ============================================================================
# DATA PATHS (relative to WORKSPACE)
# ============================================================================
export TRAIN_DATA="${WORKSPACE}/data/train.jsonl"
export VAL_DATA="${WORKSPACE}/data/val.jsonl"
export TEST_DATA="${WORKSPACE}/data/test.jsonl"

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
export EMBEDDING_DIM=128
export HIDDEN_DIM=256
export NUM_LAYERS=2
export DROPOUT=0.3

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
export BATCH_SIZE=32
export NUM_EPOCHS=50
export LEARNING_RATE=0.001
export WEIGHT_DECAY=0.00001
export GRADIENT_CLIP=5.0

# ============================================================================
# PATHS
# ============================================================================
export SAVE_DIR="${WORKSPACE}/checkpoints/${TASK}"
export LOG_DIR="${WORKSPACE}/logs/${TASK}"
export SLURM_LOG_DIR="${WORKSPACE}/slurm_logs"

# ============================================================================
# RESUME TRAINING
# ============================================================================
# Để trống nếu train từ đầu
# Ví dụ: export RESUME_CHECKPOINT="${SAVE_DIR}/latest_checkpoint.pt"
export RESUME_CHECKPOINT=""

# ============================================================================
# OTHER SETTINGS
# ============================================================================
export NUM_WORKERS=4
export SEED=42
export NUM_SAMPLES=50  # Số samples hiển thị khi evaluate

# ============================================================================
# JOB SETTINGS
# ============================================================================
export JOB_NAME="${TASK}_bilstm_$([ "$USE_CRF" = "true" ] && echo "crf" || echo "linear")"
export EMAIL="xuanhuynh233@gmail.com"  # Email nhận thông báo (optional)
export MAIL_TYPE="END,FAIL"
