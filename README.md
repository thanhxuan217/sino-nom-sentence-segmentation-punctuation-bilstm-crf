# README.md

```markdown
# Classical Chinese BiLSTM - Sentence Segmentation & Punctuation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Mô hình BiLSTM+CRF/Linear cho xử lý văn bản Hán cổ (Classical Chinese) với hai tác vụ:
- **Sentence Segmentation**: Phân đoạn câu theo schema BEMS (Begin, Middle, End, Single)
- **Sentence Punctuation**: Dự đoán dấu câu (，、。：；？！)

---

## 📋 Mục lục

- [Cấu trúc Project](#-cấu-trúc-project)
- [Yêu cầu Hệ thống](#-yêu-cầu-hệ-thống)
- [Setup Environment](#-setup-environment)
- [Cấu hình Training](#️-cấu-hình-training)
- [Hướng dẫn Sử dụng](#-hướng-dẫn-sử-dụng)
- [Quản lý Jobs](#-quản-lý-jobs)
- [Kết quả Training](#-kết-quả-training)
- [Troubleshooting](#-troubleshooting)
- [Examples](#-examples)

---

## 📁 Cấu trúc Project

/media02/ddien02/thanhxuan217/main_src/
│
├── README.md                          # 📖 Hướng dẫn này
├── requirements.txt                   # 📦 Python dependencies
│
├── envs/                              # 🐍 Conda environments (local)
│   └── classical_chinese/             # Environment cho project này
│
├── config.slurm                       # ⚙️  Cấu hình training
├── train.slurm                        # 🚀 SLURM script training
├── evaluate.slurm                     # 📊 SLURM script evaluation
├── resume.slurm                       # 🔄 SLURM script resume training
│
├── train.py                           # 🎓 Training script
├── evaluate.py                        # 📈 Evaluation script
│
├── src/                               # 💻 Source code
│   ├── __init__.py
│   ├── config.py                      # Configuration classes
│   ├── dataset.py                     # Dataset & DataLoader
│   ├── model.py                       # BiLSTM+CRF/Linear models
│   ├── trainer.py                     # Training logic
│   └── metrics.py                     # Evaluation metrics
│
├── data/                              # 📚 Dữ liệu (JSONL format)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
│
├── checkpoints/                       # 💾 Model checkpoints
│   └── {task}/
│       ├── best_model.pt
│       ├── latest_checkpoint.pt
│       └── checkpoint_epoch_*.pt
│
├── logs/                              # 📝 Training logs
│   └── {task}/
│       ├── val_metrics_epoch_*.json
│       └── test_metrics_final.json
│
├── slurm_logs/                        # 📋 SLURM output logs
│   ├── train_classical_chinese_*.out
│   └── train_classical_chinese_*.err
│
└── evaluation_results/                # 📊 Evaluation results
    └── {task}/
        ├── test_metrics.json
        └── test_samples.txt
```

---

## 💻 Yêu cầu Hệ thống

### SLURM Cluster Constraints

| Resource | Limit |
|----------|-------|
| **Maximum jobs/group** | 2 jobs đồng thời |
| **GPU per job** | Maximum 2 GPUs |
| **CPU per job** | Maximum 16 cores |
| **Memory per job** | Maximum 64GB RAM |
| **Time per job** | Maximum 48 hours |
| **Partition** | `gpu` |

### Software Requirements

- **Python**: 3.10+
- **CUDA**: 11.8+
- **Anaconda/Miniconda**: Latest version
- **SLURM**: Workload manager

---

## 🚀 Setup Environment

### Bước 1: Khởi tạo Workspace

```bash
# Di chuyển vào workspace
cd /media02/ddien02/thanhxuan217/main_src

# Tạo thư mục cho conda environment (local)
mkdir -p envs
```

### Bước 2: Load Anaconda Module

```bash
# Tùy vào hệ thống, chọn một trong các cách sau:

# Option 1: System-wide Anaconda
source /opt/anaconda3/etc/profile.d/conda.sh

# Option 2: User Anaconda
source $HOME/anaconda3/etc/profile.d/conda.sh

# Option 3: Miniconda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Verify conda loaded
which conda
# Expected: /path/to/conda
```

### Bước 3: Tạo Local Conda Environment

**⚠️ QUAN TRỌNG**: Environment phải nằm trong workspace để dễ quản lý.

```bash
# Tạo environment trong thư mục envs/
conda create --prefix ./envs/classical_chinese python=3.10 -y

# Activate environment
conda activate ./envs/classical_chinese

# Verify activation
which python
# Expected: /media02/ddien02/thanhxuan217/main_src/envs/classical_chinese/bin/python

python --version
# Expected: Python 3.10.x
```

### Bước 4: Cài đặt Dependencies

```bash
# Đảm bảo đang ở trong workspace và environment đã được activate
cd /media02/ddien02/thanhxuan217/main_src
conda activate ./envs/classical_chinese

# Hoặc cài tất cả từ requirements.txt
pip install -r requirements.txt
```

### Bước 5: Verify Installation

```bash
# Test PyTorch và CUDA
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
EOF
```

**Expected Output:**
```
PyTorch version: 2.5.1
CUDA available: True
CUDA version: 11.8
Number of GPUs: 2 (or more)
GPU 0: NVIDIA A100-SXM4-40GB (example)
```

```bash
# Test các thư viện khác
python << EOF
import torchcrf
import sklearn
import numpy as np
import pandas as pd
from src.config import LabelConfig
print("✓ All dependencies installed successfully!")
print("✓ Project modules loaded successfully!")
EOF
```

### Bước 6: Tạo Directories

```bash
# Tạo các thư mục cần thiết
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p slurm_logs
mkdir -p evaluation_results

# Verify
ls -la
```

### Bước 7: Chuẩn bị Dữ liệu

**Format JSONL** - Mỗi dòng là một JSON object:

```jsonl
{"text": "君不見君有疾若他故不見使者", "labels": ["M", "M", "E", "B", "M", "M", "M", "M", "E", "B", "E"]}
{"text": "使犬夫受受聘享也大夫上卿也", "labels": ["B", "M", "E", "B", "E", "B", "M", "E", "B", "M", "E"]}
```

**Đặt files vào thư mục data:**

```bash
# Copy hoặc move data files
cp /path/to/your/train.jsonl data/
cp /path/to/your/val.jsonl data/
cp /path/to/your/test.jsonl data/

# Verify
ls -lh data/
# Expected:
# train.jsonl
# val.jsonl
# test.jsonl
```

**Kiểm tra format:**

```bash
# Xem 2 dòng đầu tiên
head -2 data/train.jsonl

# Đếm số dòng
wc -l data/*.jsonl
```

---

## ⚙️ Cấu hình Training

### Chỉnh sửa `config.slurm`

```bash
nano config.slurm
```

### Các tham số quan trọng:

#### 1. Task Configuration

```bash
# Task type: "segmentation" hoặc "punctuation"
export TASK="segmentation"

# Model head: true (CRF) hoặc false (Linear)
export USE_CRF="true"
```

#### 2. Data Paths

```bash
# Đường dẫn tương đối từ WORKSPACE
export TRAIN_DATA="${WORKSPACE}/data/train.jsonl"
export VAL_DATA="${WORKSPACE}/data/val.jsonl"
export TEST_DATA="${WORKSPACE}/data/test.jsonl"
```

#### 3. Model Hyperparameters

```bash
export EMBEDDING_DIM=128      # Character embedding dimension
export HIDDEN_DIM=256         # LSTM hidden dimension
export NUM_LAYERS=2           # Number of LSTM layers
export DROPOUT=0.3            # Dropout rate
```

#### 4. Training Hyperparameters

```bash
export BATCH_SIZE=32          # Batch size per GPU
export NUM_EPOCHS=50          # Total training epochs
export LEARNING_RATE=0.001    # Initial learning rate
export WEIGHT_DECAY=0.00001   # L2 regularization
export GRADIENT_CLIP=5.0      # Gradient clipping threshold
```

#### 5. Resume Training

```bash
# Để trống nếu train từ đầu
export RESUME_CHECKPOINT=""

# Hoặc chỉ định checkpoint cụ thể để resume
# export RESUME_CHECKPOINT="${WORKSPACE}/checkpoints/segmentation/latest_checkpoint.pt"
```

#### 6. Other Settings

```bash
export NUM_WORKERS=4          # DataLoader workers
export SEED=42                # Random seed
export NUM_SAMPLES=50         # Số samples hiển thị khi evaluate
```

---

## 📖 Hướng dẫn Sử dụng

### 🧪 Bước 1: Test với srun (BẮT BUỘC)

**⚠️ QUAN TRỌNG**: 
- Luôn test với `srun` trước khi submit batch job
- Không giữ resource quá lâu (< 30 phút)
- Đảm bảo code chạy được trước khi submit job dài

#### Test Training (1 epoch)

```bash
srun --partition=gpu \
     --gres=gpu:1 \
     --cpus-per-task=8 \
     --mem=32G \
     --time=00:30:00 \
     --pty bash << 'EOF'

# Load conda
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /media02/ddien02/thanhxuan217/main_src/envs/classical_chinese

# Navigate to workspace
cd /media02/ddien02/thanhxuan217/main_src

# Run test training
python train.py \
    --task segmentation \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --test_data data/test.jsonl \
    --batch_size 16 \
    --num_epochs 1 \
    --save_dir test_checkpoints \
    --log_dir test_logs

conda deactivate
EOF
```

#### Test Evaluation

```bash
srun --partition=gpu \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --mem=16G \
     --time=00:15:00 \
     --pty bash << 'EOF'

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /media02/ddien02/thanhxuan217/main_src/envs/classical_chinese
cd /media02/ddien02/thanhxuan217/main_src

python evaluate.py \
    --checkpoint checkpoints/segmentation/best_model.pt \
    --test_data data/test.jsonl \
    --num_samples 10

conda deactivate
EOF
```

**Nếu test thành công**, tiến hành submit batch job.

---

### 🚀 Bước 2: Submit Jobs với sbatch

#### ✅ Kiểm tra Job Limit

```bash
# Kiểm tra số job đang chạy
squeue -u $USER

# Đảm bảo < 2 jobs
# Nếu đã có 2 jobs, đợi một job hoàn thành trước khi submit job mới
```

#### 🎓 Training từ đầu

```bash
# 1. Kiểm tra config
cat config.slurm | grep -E "TASK|USE_CRF|BATCH_SIZE|NUM_EPOCHS"

# 2. Submit job
sbatch train.slurm

# 3. Lấy Job ID từ output
# Submitted batch job 12345

# 4. Monitor job
tail -f slurm_logs/train_classical_chinese_12345.out

# 5. Kiểm tra job status
squeue -j 12345
```

#### 🔄 Resume Training

```bash
# 1. Kiểm tra checkpoint tồn tại
ls -lh checkpoints/segmentation/latest_checkpoint.pt

# 2. Submit resume job
sbatch resume.slurm

# 3. Monitor
tail -f slurm_logs/resume_train_classical_chinese_*.out
```

#### 📊 Evaluation Only

```bash
# 1. Kiểm tra best model
ls -lh checkpoints/segmentation/best_model.pt

# 2. Submit evaluation
sbatch evaluate.slurm

# 3. Monitor
tail -f slurm_logs/eval_classical_chinese_*.out
```

---

## 🎮 Quản lý Jobs

### Xem Job Status

```bash
# Xem tất cả jobs của bạn
squeue -u $USER

# Xem chi tiết hơn
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R"

# Giải thích output:
# JOBID     PARTITION NAME                           ST       TIME  TIME_LIMI  NODES NODELIST(REASON)
# 12345     gpu       train_classical_chinese        R       10:23  2-00:00:00      1 gpu01
```

### Xem Job History

```bash
# Xem jobs trong 7 ngày qua
sacct -u $USER \
      --starttime $(date -d '7 days ago' +%Y-%m-%d) \
      --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize

# Xem chi tiết một job
sacct -j 12345 --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS,MaxVMSize
```

### Monitor Training Progress

```bash
# Theo dõi log realtime
tail -f slurm_logs/train_classical_chinese_12345.out

# Xem 100 dòng cuối
tail -n 100 slurm_logs/train_classical_chinese_12345.out

# Grep specific info
grep "Epoch" slurm_logs/train_classical_chinese_12345.out
grep "F1:" slurm_logs/train_classical_chinese_12345.out
grep "Best" slurm_logs/train_classical_chinese_12345.out
```

### Kiểm tra GPU Usage (nếu đang chạy interactive)

```bash
# Trong srun session
watch -n 1 nvidia-smi

# Xem GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Kiểm tra Checkpoints

```bash
# List checkpoints
ls -lht checkpoints/segmentation/

# Xem thông tin checkpoint
python << EOF
import torch
ckpt = torch.load('checkpoints/segmentation/best_model.pt', map_location='cpu')
print(f"Epoch: {ckpt['epoch']}")
print(f"Best F1: {ckpt['best_val_f1']:.4f}")
print(f"Best Epoch: {ckpt['best_epoch']}")
EOF
```

### Cancel Jobs

```bash
# Cancel một job cụ thể
scancel 12345

# Cancel tất cả jobs của bạn
scancel -u $USER

# Cancel jobs theo tên
scancel --name=train_classical_chinese
```

### Job Priority & Queue Info

```bash
# Xem vị trí job trong queue
squeue -u $USER --start

# Xem thông tin partition
sinfo -p gpu

# Xem node availability
sinfo -N -p gpu
```

---

## 📊 Kết quả Training

### Checkpoints Directory

```
checkpoints/segmentation/
├── best_model.pt              # Model tốt nhất (highest validation F1)
├── latest_checkpoint.pt       # Checkpoint mới nhất (để resume)
├── checkpoint_epoch_5.pt      # Checkpoint định kỳ (mỗi 5 epochs)
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_15.pt
├── ...
├── label_config.json          # Label configuration
└── training_config.json       # Training configuration
```

**Checkpoint Structure:**

```python
{
    'epoch': 45,                           # Epoch hiện tại
    'model_state_dict': {...},             # Model weights
    'optimizer_state_dict': {...},         # Optimizer state
    'scheduler_state_dict': {...},         # Scheduler state
    'best_val_f1': 0.9234,                # Best validation F1
    'best_epoch': 42,                      # Epoch của best model
    'patience_counter': 3,                 # Early stopping counter
    'training_config': {...},              # Training configuration
    'model_config': {...}                  # Model configuration
}
```

### Training Logs

```
logs/segmentation/
├── val_metrics_epoch_1.json    # Validation metrics epoch 1
├── val_metrics_epoch_2.json    # Validation metrics epoch 2
├── ...
├── val_metrics_epoch_50.json
└── test_metrics_final.json     # Final test metrics
```

**Metrics JSON Format:**

```json
{
  "epoch": 45,
  "split": "val",
  "task_type": "segmentation",
  "training_config": {
    "task_type": "segmentation",
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.001
  },
  "model_config": {
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "use_crf": true
  },
  "metrics": {
    "per_class": {
      "B": {
        "precision": 0.9234,
        "recall": 0.9156,
        "f1": 0.9195,
        "support": 15234
      },
      "M": {
        "precision": 0.9123,
        "recall": 0.9087,
        "f1": 0.9105,
        "support": 25123
      },
      "E": {
        "precision": 0.9267,
        "recall": 0.9198,
        "f1": 0.9232,
        "support": 15234
      },
      "S": {
        "precision": 0.8956,
        "recall": 0.8892,
        "f1": 0.8924,
        "support": 4532
      }
    },
    "overall": {
      "precision": 0.9145,
      "recall": 0.9083,
      "f1": 0.9114,
      "total_samples": 60123
    },
    "confusion_matrix": [[...], [...], [...], [...]],
    "label_names": ["B", "M", "E", "S"]
  }
}
```

### Evaluation Results

```
evaluation_results/segmentation/
├── test_metrics.json          # Test metrics (JSON format)
└── test_samples.txt           # Sample predictions (text format)
```

**Sample Output Format** (`test_samples.txt`):

```
======================================================================
TEST SAMPLES - SEGMENTATION
======================================================================

Sample 1/50 (Index: 1234)
----------------------------------------------------------------------
Char   Pred   True   Status
----------------------------------------
君      B      B      ✓
不      M      M      ✓
見      E      E      ✓
君      B      B      ✓
有      M      M      ✓
疾      E      E      ✓

Predicted:
|君不見||君有疾|

Ground Truth:
|君不見||君有疾|
======================================================================
```

### SLURM Logs

```
slurm_logs/
├── train_classical_chinese_12345.out    # Training stdout
├── train_classical_chinese_12345.err    # Training stderr
├── eval_classical_chinese_12346.out     # Evaluation stdout
└── eval_classical_chinese_12346.err     # Evaluation stderr
```

---

## 🔧 Troubleshooting

### ❌ Lỗi: "conda: command not found"

**Nguyên nhân**: Conda chưa được load vào environment.

**Giải pháp**:

```bash
# Tìm đường dẫn conda
which conda

# Nếu không tìm thấy, load conda
source /opt/anaconda3/etc/profile.d/conda.sh
# hoặc
source $HOME/anaconda3/etc/profile.d/conda.sh
# hoặc
source $HOME/miniconda3/etc/profile.d/conda.sh

# Verify
which conda
conda --version
```

### ❌ Lỗi: "CUDA out of memory"

**Nguyên nhân**: Batch size quá lớn cho GPU.

**Giải pháp**:

```bash
# Option 1: Giảm batch size
nano config.slurm
# Thay đổi:
export BATCH_SIZE=16  # hoặc 8

# Option 2: Sử dụng gradient accumulation
# Trong train.py, thêm accumulation steps
```

### ❌ Lỗi: "Job killed due to timeout"

**Nguyên nhân**: Job vượt quá thời gian cho phép (48h).

**Giải pháp**:

```bash
# Option 1: Giảm số epochs
export NUM_EPOCHS=30

# Option 2: Resume training từ checkpoint
sbatch resume.slurm

# Option 3: Tăng batch size để training nhanh hơn
export BATCH_SIZE=64  # nếu GPU memory cho phép
```

### ❌ Lỗi: "No module named 'src'"

**Nguyên nhân**: Python không tìm thấy module src.

**Giải pháp**:

```bash
# Đảm bảo đang ở đúng workspace
cd /media02/ddien02/thanhxuan217/main_src

# Kiểm tra src/ tồn tại
ls -la src/

# Kiểm tra PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/media02/ddien02/thanhxuan217/main_src"
```

### ❌ Lỗi: "Job pending quá lâu"

**Nguyên nhân**: Cluster đang bận, không đủ resource.

**Giải pháp**:

```bash
# Xem lý do pending
squeue -u $USER --start

# Xem node availability
sinfo -p gpu

# Nếu cần gấp, có thể giảm resource request
# Sửa trong .slurm file:
#SBATCH --gres=gpu:1    # thay vì gpu:2
#SBATCH --cpus-per-task=8  # thay vì 16
```

### ❌ Lỗi: "RuntimeError: DataLoader worker exited unexpectedly"

**Nguyên nhân**: Vấn đề với num_workers hoặc data corrupted.

**Giải pháp**:

```bash
# Option 1: Giảm num_workers
export NUM_WORKERS=0  # hoặc 2

# Option 2: Kiểm tra data
python << EOF
import json
with open('data/train.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except:
            print(f"Error at line {i}: {line}")
EOF
```

### ❌ Lỗi: "FileNotFoundError: checkpoint not found"

**Nguyên nhân**: Checkpoint file không tồn tại.

**Giải pháp**:

```bash
# Kiểm tra checkpoints
ls -lh checkpoints/segmentation/

# Nếu muốn resume nhưng không có checkpoint, train từ đầu
export RESUME_CHECKPOINT=""
sbatch train.slurm
```

### 🔍 Debug Tips

```bash
# 1. Kiểm tra SLURM logs
tail -n 50 slurm_logs/train_classical_chinese_*.err

# 2. Test với small dataset
head -100 data/train.jsonl > data/train_small.jsonl
# Rồi test với data_small

# 3. Enable verbose logging
python train.py --task segmentation ... --log_interval 10

# 4. Kiểm tra GPU
nvidia-smi
nvidia-smi dmon  # Monitor realtime

# 5. Check disk space
df -h /media02/ddien02/thanhxuan217/
du -sh /media02/ddien02/thanhxuan217/main_src/*
```

---

## 🎯 Examples

### Example 1: Training Segmentation với CRF (Recommended)

```bash
# Step 1: Edit config.slurm
nano config.slurm
```

```bash
# config.slurm
export TASK="segmentation"
export USE_CRF="true"
export BATCH_SIZE=32
export NUM_EPOCHS=50
export LEARNING_RATE=0.001
export HIDDEN_DIM=256
export NUM_LAYERS=2
```

```bash
# Step 2: Test với srun
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 \
     --pty bash -c "
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /media02/ddien02/thanhxuan217/main_src/envs/classical_chinese
cd /media02/ddien02/thanhxuan217/main_src
python train.py --task segmentation --train_data data/train.jsonl --val_data data/val.jsonl --test_data data/test.jsonl --batch_size 16 --num_epochs 1 --use_crf
"

# Step 3: Nếu test OK, submit batch job
sbatch train.slurm

# Step 4: Monitor
squeue -u $USER
tail -f slurm_logs/train_classical_chinese_*.out
```

### Example 2: Training Punctuation với Linear Head

```bash
# Step 1: Edit config
nano config.slurm
```

```bash
export TASK="punctuation"
export USE_CRF="false"
export BATCH_SIZE=32
export NUM_EPOCHS=50
```

```bash
# Step 2: Submit
sbatch train.slurm

# Step 3: Monitor
tail -f slurm_logs/train_classical_chinese_*.out | grep -E "Epoch|F1|Best"
```

### Example 3: Resume Training sau khi bị gián đoạn

```bash
# Step 1: Kiểm tra checkpoint
ls -lh checkpoints/segmentation/latest_checkpoint.pt

# Expected output:
# -rw-r--r-- 1 user group 45M Jan 15 10:30 latest_checkpoint.pt

# Step 2: Xem epoch đã train
python << EOF
import torch
ckpt = torch.load('checkpoints/segmentation/latest_checkpoint.pt', map_location='cpu')
print(f"Last completed epoch: {ckpt['epoch']}")
print(f"Best F1 so far: {ckpt['best_val_f1']:.4f}")
print(f"Will resume from epoch: {ckpt['epoch'] + 1}")
EOF

# Step 3: Submit resume job
sbatch resume.slurm

# Step 4: Verify resume
tail -f slurm_logs/resume_train_classical_chinese_*.out | head -20
# Should see: "Resumed from epoch X"
```

### Example 4: Evaluate multiple checkpoints

```bash
# Evaluate best model
python evaluate.py \
    --checkpoint checkpoints/segmentation/best_model.pt \
    --test_data data/test.jsonl \
    --num_samples 100 \
    --output_dir evaluation_results/segmentation/best

# Evaluate specific epoch
python evaluate.py \
    --checkpoint checkpoints/segmentation/checkpoint_epoch_30.pt \
    --test_data data/test.jsonl \
    --num_samples 100 \
    --output_dir evaluation_results/segmentation/epoch30

# Compare results
diff evaluation_results/segmentation/best/test_metrics.json \
     evaluation_results/segmentation/epoch30/test_metrics.json
```

### Example 5: Hyperparameter Search

```bash
# Train with different configurations
for HIDDEN_DIM in 128 256 512; do
    for NUM_LAYERS in 2 3; do
        export HIDDEN_DIM=$HIDDEN_DIM
        export NUM_LAYERS=$NUM_LAYERS
        export SAVE_DIR="checkpoints/segmentation_h${HIDDEN_DIM}_l${NUM_LAYERS}"
        
        # Wait if already have 2 jobs
        while [ $(squeue -u $USER -h | wc -l) -ge 2 ]; do
            echo "Waiting for job slot..."
            sleep 60
        done
        
        sbatch train.slurm
        sleep 5
    done
done
```

---

## 📚 Additional Resources

### Dataset Format Reference

**Segmentation JSONL:**
```jsonl
{"text": "君子有三樂", "labels": ["B", "M", "E", "B", "E"]}
{"text": "父母俱存兄弟無故一樂也", "labels": ["B", "M", "M", "E", "B", "M", "M", "M", "E", "S", "S", "S"]}
```

**Punctuation JSONL:**
```jsonl
{"text": "學而時習之不亦說乎", "labels": ["O", "O", "O", "O", "O", "，", "O", "O", "O", "？"]}
{"text": "有朋自遠方來不亦樂乎", "labels": ["O", "O", "O", "O", "O", "O", "，", "O", "O", "O", "？"]}
```

### Label Schema

**Segmentation (BEMS):**
- `B` (Begin): Token ở đầu câu
- `M` (Middle): Token ở giữa câu
- `E` (End): Token ở cuối câu
- `S` (Single): Câu chỉ có một ký tự

**Punctuation:**
- `O`: Không có dấu câu
- `，`: Dấu phẩy
- `。`: Dấu chấm
- `：`: Dấu hai chấm
- `、`: Dấu đốt
- `；`: Dấu chấm phẩy
- `？`: Dấu hỏi
- `！`: Dấu cảm

### Useful Commands Cheat Sheet

```bash
# === Environment ===
conda activate /media02/ddien02/thanhxuan217/main_src/envs/classical_chinese
conda deactivate

# === Job Management ===
squeue -u $USER                          # My jobs
squeue -u $USER -o "%.18i %.30j %.8T"   # Compact view
sacct -u $USER --starttime today         # Today's history
scancel <JOB_ID>                         # Cancel job
scancel -u $USER                         # Cancel all my jobs

# === Monitoring ===
tail -f slurm_logs/train_*.out          # Follow training log
watch -n 5 'squeue -u $USER'            # Auto-refresh job status
nvidia-smi                               # GPU info

# === Data ===
wc -l data/*.jsonl                      # Count lines
head -5 data/train.jsonl                # View first 5
tail -5 data/train.jsonl                # View last 5

# === Results ===
ls -lht checkpoints/segmentation/       # List checkpoints
cat logs/segmentation/test_metrics_final.json | jq .  # Pretty print JSON
grep "F1" logs/segmentation/val_metrics_*.json        # Extract F1 scores
```

---

## 📞 Support

### Common Issues

1. **Job không chạy**: Kiểm tra `squeue -u $USER --start`
2. **Out of memory**: Giảm batch size
3. **Slow training**: Tăng batch size, check GPU usage
4. **Poor performance**: Tune hyperparameters, check data quality

### Getting Help

```bash
# Check cluster documentation
man sbatch
man squeue
man scancel

# Contact support (nếu có)
# Email: support@your-cluster.edu
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- EvalHan2024 evaluation framework
- PyTorch team
- TorchCRF library

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Author**: Classical Chinese NLP Team
