# src/config.py
"""
Cấu hình cho mô hình BiLSTM+CRF xử lý văn bản Hán cổ
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Cấu hình model architecture"""
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    use_crf: bool = True  # True để dùng CRF, False để dùng Linear


@dataclass
class TrainingConfig:
    """Cấu hình training"""
    # Task
    task_type: str = "segmentation"  # "punctuation" hoặc "segmentation"
    
    # Data paths
    train_data: str = ""
    val_data: str = ""
    test_data: str = ""
    
    # Training hyperparameters
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    warmup_steps: int = 500
    
    # Data
    max_length: int = 256
    num_workers: int = 4
    
    # Checkpoint & Logging
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    resume_from: Optional[str] = None  # Path to checkpoint để resume
    save_interval: int = 5
    eval_interval: int = 1
    log_interval: int = 100
    
    # Test samples
    num_test_samples: int = 50
    
    # Distributed
    distributed: bool = True
    
    # Early stopping
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        """Tạo directories nếu chưa tồn tại"""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        """Convert config thành dictionary"""
        return asdict(self)
    
    def save(self, path: str):
        """Lưu config ra file JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, path: str):
        """Load config từ file JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class LabelConfig:
    """Cấu hình labels cho từng task"""
    
    # Task labels
    PUNCTUATION_LABELS = ['O', '，', '。', '：', '、', '；', '？', '！']
    SEGMENTATION_LABELS = ['B', 'M', 'E', 'S']
    
    def __init__(self, task_type: str = 'segmentation'):
        """
        Args:
            task_type: 'punctuation' hoặc 'segmentation'
        """
        self.task_type = task_type
        
        if task_type == 'punctuation':
            self.labels = self.PUNCTUATION_LABELS
        elif task_type == 'segmentation':
            self.labels = self.SEGMENTATION_LABELS
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.num_labels = len(self.labels)
    
    def save(self, path: str):
        """Lưu label config"""
        config_dict = {
            'task_type': self.task_type,
            'labels': self.labels,
            'label2id': self.label2id,
            'id2label': {str(k): v for k, v in self.id2label.items()},
            'num_labels': self.num_labels
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load label config từ file"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(task_type=config_dict['task_type'])
