#!/bin/bash
#SBATCH --job-name=lstm_eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# ============================================================================
# SLURM Evaluation Script
# Using torchrun with 2 GPUs (distributed evaluation)
# ============================================================================

set -e

echo "========================================================================"
echo "Evaluation Job Information"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================================================"

# Load configuration
source "config.slurm"

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

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT}"
    echo ""
    echo "Available checkpoints in ${SAVE_DIR}:"
    ls -lht "${SAVE_DIR}"/*.pt 2>/dev/null || echo "No checkpoints found"
    echo ""
    exit 1
fi

echo "Checkpoint Information:"
echo "========================================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "File size: $(du -h ${CHECKPOINT} | cut -f1)"
echo "Modified: $(stat -c %y ${CHECKPOINT} 2>/dev/null || stat -f %Sm ${CHECKPOINT})"
echo ""

# Checkpoint details will be printed by evaluate.py

echo "Evaluation Settings:"
echo "========================================================================"
echo "Test data: ${DATA_DIR}/${TEST_SPLIT}/"
echo "Vocab: ${VOCAB_PATH}"
echo "Number of samples: ${NUM_SAMPLES}"
echo "Output directory: ${WORKSPACE}/evaluation_results/${TASK}"
echo "Random seed: ${SEED}"
echo "Number of GPUs: 2"
echo "========================================================================"
echo ""

# ============================================================================
# RUN EVALUATION WITH TORCHRUN (2 GPUs)
# ============================================================================

echo "Starting distributed evaluation with torchrun on 2 GPUs..."
echo "========================================================================"

# Run with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    evaluate.py \
    --checkpoint ${CHECKPOINT} \
    --data_dir ${DATA_DIR} \
    --vocab_path ${VOCAB_PATH} \
    --test_split ${TEST_SPLIT} \
    --task ${TEST_SPLIT} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${WORKSPACE}/evaluation_${TEST_SPLIT}_results/${TASK}" \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED}

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation completed successfully!"
    echo ""
    echo "Results:"
    echo "  - Metrics: ${WORKSPACE}/evaluation_results/${TASK}/test_metrics.json"
    echo "  - Samples: ${WORKSPACE}/evaluation_results/${TASK}/test_samples.txt"
    echo "  - Sample Results: ${WORKSPACE}/evaluation_results/${TASK}/sample_results.json"
    echo ""
    # Performance summary is now printed by evaluate.py
else
    echo "✗ Evaluation failed with exit code: $EXIT_CODE"
fi

echo ""
echo "End Time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
