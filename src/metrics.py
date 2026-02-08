# src/metrics.py
"""
Metrics cho evaluation theo chuẩn EvalHan2024
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from typing import Dict, List, Tuple
import json


class MetricsCalculator:
    """Tính toán metrics cho token classification"""
    
    def __init__(self, label_config, ignore_labels: List[str] = None):
        """
        Args:
            label_config: LabelConfig object
            ignore_labels: List các label cần ignore
        """
        self.label_config = label_config
        
        # Đối với punctuation, ignore label 'O'
        if ignore_labels is None:
            if label_config.task_type == 'punctuation':
                ignore_labels = ['O']
            else:
                ignore_labels = []
        
        self.ignore_label_ids = [
            label_config.label2id[label] for label in ignore_labels 
            if label in label_config.label2id
        ]
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Tính precision, recall, F1
        
        Args:
            predictions: [total_tokens]
            labels: [total_tokens]
        """
        # Filter padding và ignore labels
        valid_mask = (labels != -100)
        for ignore_id in self.ignore_label_ids:
            valid_mask = valid_mask & (labels != ignore_id)
        
        filtered_preds = predictions[valid_mask]
        filtered_labels = labels[valid_mask]
        
        if len(filtered_labels) == 0:
            return self._empty_metrics()
        
        # Get unique labels
        label_ids = sorted(set(filtered_labels))
        label_names = [self.label_config.id2label[lid] for lid in label_ids]
        
        # Compute metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            filtered_labels,
            filtered_preds,
            labels=label_ids,
            average=None,
            zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, label_name in enumerate(label_names):
            per_class_metrics[label_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Overall metrics (macro average)
        overall_precision = np.mean(precision)
        overall_recall = np.mean(recall)
        overall_f1 = np.mean(f1)
        
        # Confusion matrix
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=label_ids)
        
        return {
            'per_class': per_class_metrics,
            'overall': {
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1': float(overall_f1),
                'total_samples': int(np.sum(support))
            },
            'confusion_matrix': cm.tolist(),
            'label_names': label_names
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics"""
        return {
            'per_class': {},
            'overall': {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'total_samples': 0
            },
            'confusion_matrix': [],
            'label_names': []
        }


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_crf: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thu thập predictions từ toàn bộ dataset
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            lengths = batch['lengths'].to(device)
            
            outputs = model(input_ids, lengths)
            
            if use_crf:
                # CRF decode
                predictions = outputs['predictions']
                for i, pred in enumerate(predictions):
                    all_predictions.extend(pred)
                    length = lengths[i].item()
                    all_labels.extend(label_ids[i, :length].cpu().numpy())
            else:
                # Linear head
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                
                for i in range(len(input_ids)):
                    length = lengths[i].item()
                    all_predictions.extend(predictions[i, :length].cpu().numpy())
                    all_labels.extend(label_ids[i, :length].cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)
