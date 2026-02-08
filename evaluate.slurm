#!/bin/bash
#SBATCH --job-name=eval_classical_chinese
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.slurm"

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
echo "Using 2 GPUs for distributed evaluation"
echo "========================================================================"
echo ""

# ============================================================================
# CHECKPOINT SELECTION
# ============================================================================

# Checkpoint có thể được truyền qua biến môi trường hoặc argument
CHECKPOINT="${1:-${SAVE_DIR}/best_model.pt}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT}"
    echo ""
    echo "Available checkpoints in ${SAVE_DIR}:"
    ls -lht "${SAVE_DIR}"/*.pt 2>/dev/null || echo "No checkpoints found"
    echo ""
    conda deactivate
    exit 1
fi

echo "Checkpoint Information:"
echo "========================================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "File size: $(du -h ${CHECKPOINT} | cut -f1)"
echo "Modified: $(stat -c %y ${CHECKPOINT} 2>/dev/null || stat -f %Sm ${CHECKPOINT})"
echo ""

# Display checkpoint details
python << EOF
import torch
try:
    ckpt = torch.load('${CHECKPOINT}', map_location='cpu')
    print(f"Model Information:")
    print(f"  Trained Epoch: {ckpt['epoch']}")
    print(f"  Best Validation F1: {ckpt['best_val_f1']:.4f}")
    print(f"  Best Epoch: {ckpt['best_epoch']}")
    print(f"  Task: {ckpt['training_config']['task_type']}")
    print(f"  Model Type: BiLSTM + {'CRF' if ckpt['model_config']['use_crf'] else 'Linear'}")
except Exception as e:
    print(f"Warning: Could not load checkpoint info: {e}")
print("")
EOF

echo "Evaluation Settings:"
echo "========================================================================"
echo "Test data: ${TEST_DATA}"
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

# Set environment variables for distributed
export MASTER_ADDR=localhost
export MASTER_PORT=12356  # Different port from training

# Run with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    evaluate.py \
    --checkpoint ${CHECKPOINT} \
    --test_data ${TEST_DATA} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${WORKSPACE}/evaluation_results/${TASK}" \
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
    echo ""
    
    # Display summary metrics if available
    if [ -f "${WORKSPACE}/evaluation_results/${TASK}/test_metrics.json" ]; then
        echo "Performance Summary:"
        python << EOF
import json
with open('${WORKSPACE}/evaluation_results/${TASK}/test_metrics.json', 'r') as f:
    metrics = json.load(f)
overall = metrics['metrics']['overall']
print(f"  Overall F1: {overall['f1']:.4f}")
print(f"  Precision:  {overall['precision']:.4f}")
print(f"  Recall:     {overall['recall']:.4f}")
print(f"  Samples:    {overall['total_samples']}")
print(f"  GPUs used:  {metrics['num_gpus']}")
EOF
    fi
else
    echo "✗ Evaluation failed with exit code: $EXIT_CODE"
fi

echo ""
echo "End Time: $(date)"
echo "========================================================================"

conda deactivate

exit $EXIT_CODE
