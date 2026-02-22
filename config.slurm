#!/bin/bash
# config.slurm - Configuration for SLURM jobs

# ============================================================================
# WORKSPACE & ENVIRONMENT
# ============================================================================
export MASTER_ADDR="localhost"
export MASTER_PORT="29500" # Default torch port, can be changed
export WORKSPACE="/media02/ddien02/thanhxuan217/main_src"

# ============================================================================
# TASK CONFIGURATION
# ============================================================================
export TASK="segmentation"  # "segmentation" hoặc "punctuation"
export USE_CRF="true"       # true để dùng CRF, false để dùng Linear

# ============================================================================
# DATA PATHS (Parquet streaming)
# ============================================================================
export DATA_DIR="${WORKSPACE}/data"
export VOCAB_PATH="${WORKSPACE}/vocab.json"
export TRAIN_SPLIT="train"
export VAL_SPLIT="val"
export TEST_SPLIT="test" # val if run in all validation set, else test
export SHUFFLE_BUFFER=50000

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
export EMBEDDING_DIM=256
export HIDDEN_DIM=512
export NUM_LAYERS=3
export DROPOUT=0.5

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
export BATCH_SIZE=128
export NUM_EPOCHS=20
export LEARNING_RATE=0.002
export WEIGHT_DECAY=0.00001
export GRADIENT_CLIP=5.0
export MAX_LENGTH=256
export WARMUP_STEPS=2000
export GRADIENT_ACCUMULATION_STEPS=2
export USE_AMP="true"          # Mixed precision training
export SAVE_EVERY_N_STEPS=5000  # Intra-epoch checkpoint
export VAL_MAX_SAMPLES=6000     # Giới hạn mẫu val mỗi epoch (0 = không giới hạn)
export EARLY_STOPPING_PATIENCE=5    # Số epoch kiên nhẫn trước khi dừng sớm

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
export NUM_WORKERS=0
export SEED=42
export NUM_SAMPLES=50  # Số samples hiển thị khi evaluate

export CHECKPOINT="${SAVE_DIR}/best_model.pt"
