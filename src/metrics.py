# src/metrics.py
"""
Metrics cho evaluation theo chuẩn EvalHan2024
Sử dụng confusion matrix accumulation để tránh lưu toàn bộ predictions/labels lên RAM.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


class MetricsCalculator:
    """Tính toán metrics cho token classification bằng cách cộng dồn confusion matrix."""
    
    def __init__(self, label_config, ignore_labels: List[str] = None):
        """
        Args:
            label_config: LabelConfig object
            ignore_labels: List các label cần ignore
        """
        self.label_config = label_config
        self.num_labels = label_config.num_labels
        
        # Đối với punctuation, ignore label 'O'
        if ignore_labels is None:
            if label_config.task_type == 'punctuation':
                ignore_labels = ['O']
            else:
                ignore_labels = []
        
        self.ignore_label_ids = set(
            label_config.label2id[label] for label in ignore_labels 
            if label in label_config.label2id
        )
        
        # Tất cả label ids cần đánh giá (trừ ignore)
        self.eval_label_ids = sorted(
            lid for lid in range(self.num_labels)
            if lid not in self.ignore_label_ids
        )
        
        # Confusion matrix tích lũy: shape (num_labels, num_labels)
        self._cm = np.zeros((self.num_labels, self.num_labels), dtype=np.int64)
    
    def reset(self):
        """Reset confusion matrix về 0."""
        self._cm[:] = 0
    
    def update_batch(self, predictions: np.ndarray, labels: np.ndarray):
        """
        Cộng dồn confusion matrix từ một batch.
        Chỉ giữ confusion matrix (num_labels x num_labels) trên RAM,
        không lưu predictions/labels.
        
        Args:
            predictions: [batch_tokens] - flat array of predicted label ids
            labels: [batch_tokens] - flat array of true label ids
        """
        # Filter padding (-100)
        valid_mask = (labels != -100)
        preds = predictions[valid_mask]
        labs = labels[valid_mask]
        
        if len(labs) == 0:
            return
        
        # Cộng dồn bằng np.add.at — O(n) và không tạo matrix tạm
        np.add.at(self._cm, (labs, preds), 1)
    
    def compute_metrics(self, predictions: np.ndarray = None, labels: np.ndarray = None) -> Dict:
        """
        Tính precision, recall, F1 từ confusion matrix đã tích lũy.
        
        Nếu truyền predictions và labels, sẽ tính trực tiếp (backward compatible).
        Nếu không truyền, sẽ dùng confusion matrix đã tích lũy qua update_batch().
        """
        if predictions is not None and labels is not None:
            # Backward compatible: tính từ data truyền vào
            self.reset()
            self.update_batch(predictions, labels)
        
        # Lấy sub-matrix cho các label cần đánh giá
        eval_ids = self.eval_label_ids
        if len(eval_ids) == 0:
            return self._empty_metrics()
        
        # Tính support (tổng số true samples mỗi class) từ confusion matrix
        # support[i] = sum of row i (tổng lần label i xuất hiện trong ground truth)
        support = np.array([self._cm[lid, :].sum() for lid in eval_ids], dtype=np.int64)
        
        total_support = support.sum()
        if total_support == 0:
            return self._empty_metrics()
        
        # TP, FP, FN từ confusion matrix
        tp = np.array([self._cm[lid, lid] for lid in eval_ids], dtype=np.int64)
        fp = np.array([self._cm[:, lid].sum() - self._cm[lid, lid] for lid in eval_ids], dtype=np.int64)
        fn = np.array([self._cm[lid, :].sum() - self._cm[lid, lid] for lid in eval_ids], dtype=np.int64)
        
        # Precision, Recall, F1 per class
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0).astype(np.float64)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0).astype(np.float64)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0
        ).astype(np.float64)
        
        # Per-class metrics
        label_names = [self.label_config.id2label[lid] for lid in eval_ids]
        per_class_metrics = {}
        for i, label_name in enumerate(label_names):
            per_class_metrics[label_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Overall metrics (macro average)
        overall_precision = float(np.mean(precision))
        overall_recall = float(np.mean(recall))
        overall_f1 = float(np.mean(f1))
        
        # Trích confusion matrix cho eval labels
        cm_eval = self._cm[np.ix_(eval_ids, eval_ids)]
        
        return {
            'per_class': per_class_metrics,
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'total_samples': int(total_support)
            },
            'confusion_matrix': cm_eval.tolist(),
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


def collect_and_compute_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics_calculator: MetricsCalculator,
    use_crf: bool = False
) -> Dict:
    """
    Streaming evaluation: cộng dồn confusion matrix theo từng batch,
    không lưu toàn bộ predictions/labels lên RAM.
    
    Returns:
        Dict chứa metrics (precision, recall, f1, confusion_matrix, ...)
    """
    model.eval()
    metrics_calculator.reset()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            lengths = batch['lengths'].to(device)
            
            outputs = model(input_ids, lengths)
            
            if use_crf:
                predictions = outputs['predictions']
                for i, pred in enumerate(predictions):
                    length = lengths[i].item()
                    pred_np = np.array(pred[:length])
                    label_np = label_ids[i, :length].cpu().numpy()
                    metrics_calculator.update_batch(pred_np, label_np)
            else:
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=-1)
                
                for i in range(len(input_ids)):
                    length = lengths[i].item()
                    pred_np = preds[i, :length].cpu().numpy()
                    label_np = label_ids[i, :length].cpu().numpy()
                    metrics_calculator.update_batch(pred_np, label_np)
    
    return metrics_calculator.compute_metrics()
