#!/bin/bash
#SBATCH --job-name=resume_train_classical_chinese
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
# SLURM Resume Training Script
# Using torchrun with 2 GPUs
# ============================================================================

set -e

echo "========================================================================"
echo "Resume Training Job Information"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================================================"

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.slurm"

cd "${WORKSPACE}"

# ============================================================================
# FIND LATEST CHECKPOINT
# ============================================================================

LATEST_CHECKPOINT="${SAVE_DIR}/latest_checkpoint.pt"

if [ ! -f "$LATEST_CHECKPOINT" ]; then
    echo "Error: Latest checkpoint not found at ${LATEST_CHECKPOINT}"
    echo ""
    echo "Available checkpoints:"
    ls -lh "${SAVE_DIR}"/*.pt 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

echo "Resuming from: ${LATEST_CHECKPOINT}"
echo ""

python << EOF
import torch
ckpt = torch.load('${LATEST_CHECKPOINT}', map_location='cpu')
print(f"Checkpoint Information:")
print(f"  Last completed epoch: {ckpt['epoch']}")
print(f"  Best validation F1: {ckpt['best_val_f1']:.4f}")
print(f"  Best epoch: {ckpt['best_epoch']}")
print(f"  Will resume from epoch: {ckpt['epoch'] + 1}")
print("")
EOF

export RESUME_CHECKPOINT="${LATEST_CHECKPOINT}"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================

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

conda activate "${CONDA_ENV}"

echo "Environment:"
echo "========================================================================"
echo "Conda environment: ${CONDA_ENV}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

echo "GPU Information:"
echo "========================================================================"
nvidia-smi
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================

TRAIN_CMD="train.py \
    --task ${TASK} \
    --train_data ${TRAIN_DATA} \
    --val_data ${VAL_DATA} \
    --test_data ${TEST_DATA} \
    --embedding_dim ${EMBEDDING_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_layers ${NUM_LAYERS} \
    --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --gradient_clip ${GRADIENT_CLIP} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --resume ${RESUME_CHECKPOINT}"

if [ "$USE_CRF" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_crf"
fi

# ============================================================================
# RUN TRAINING WITH TORCHRUN (2 GPUs)
# ============================================================================

echo "Resuming training with torchrun on 2 GPUs..."
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
    echo "Resume training completed successfully!"
    
    # Run evaluation with torchrun
    BEST_MODEL="${SAVE_DIR}/best_model.pt"
    if [ -f "$BEST_MODEL" ]; then
        echo ""
        echo "Running evaluation with torchrun on 2 GPUs..."
        
        export MASTER_PORT=12356
        
        torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=2 \
            evaluate.py \
            --checkpoint ${BEST_MODEL} \
            --test_data ${TEST_DATA} \
            --num_samples ${NUM_SAMPLES} \
            --output_dir "${WORKSPACE}/evaluation_results/${TASK}" \
            --seed ${SEED}
    fi
else
    echo "Resume training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "End Time: $(date)"
echo "========================================================================"

conda deactivate

exit $TRAIN_EXIT_CODE
