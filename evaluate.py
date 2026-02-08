# evaluate.py
"""
Evaluation script với distributed support
Usage:
    # Single GPU
    python evaluate.py --checkpoint checkpoints/best_model.pt --test_data data/test.jsonl
    
    # Multi-GPU với torchrun (2 GPUs)
    torchrun --nproc_per_node=2 evaluate.py --checkpoint checkpoints/best_model.pt --test_data data/test.jsonl
"""

import argparse
import os
import torch
import torch.distributed as dist
import random
import json
from typing import List

from src.config import LabelConfig, ModelConfig
from src.dataset import ChineseTextDataset, collate_fn
from src.model import create_model
from src.metrics import MetricsCalculator, collect_predictions
from torch.utils.data import DataLoader


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(rank)
        return rank, world_size
    else:
        return 0, 1


def cleanup_distributed():
    """Cleanup distributed"""
    if dist.is_initialized():
        dist.destroy_process_group()


def decode_predictions(
    chars: List[str],
    predictions: List[int],
    true_labels: List[int],
    id2label: dict
) -> str:
    """Format predictions với true labels"""
    lines = []
    lines.append(f"{'Char':<6} {'Pred':<6} {'True':<6} {'Status'}")
    lines.append("-" * 40)
    
    for char, pred_id, true_id in zip(chars, predictions, true_labels):
        pred_label = id2label[pred_id]
        true_label = id2label[true_id]
        
        status = "✓" if pred_label == true_label else "✗"
        lines.append(f"{char:<6} {pred_label:<6} {true_label:<6} {status}")
    
    return "\n".join(lines)


def generate_text_with_labels(
    chars: List[str],
    predictions: List[int],
    id2label: dict,
    task_type: str
) -> str:
    """Generate văn bản với labels"""
    if task_type == 'punctuation':
        result = []
        for char, pred_id in zip(chars, predictions):
            result.append(char)
            label = id2label[pred_id]
            if label != 'O':
                result.append(label)
        return ''.join(result)
    else:  # segmentation
        result = []
        for char, pred_id in zip(chars, predictions):
            label = id2label[pred_id]
            if label in ['B', 'S']:
                result.append('|')
            result.append(char)
            if label in ['E', 'S']:
                result.append('|')
        return ''.join(result)


def evaluate_and_sample(
    checkpoint_path: str,
    test_data_path: str,
    num_samples: int = 50,
    output_dir: str = 'evaluation_results',
    rank: int = 0,
    world_size: int = 1
):
    """Evaluate model và generate samples (distributed)"""
    
    # Only rank 0 prints
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get configs
    task_type = checkpoint['training_config']['task_type']
    model_config_dict = checkpoint['model_config']
    
    # Create configs
    label_config = LabelConfig(task_type=task_type)
    model_config = ModelConfig(**model_config_dict)
    
    if rank == 0:
        print(f"Task: {task_type}")
        print(f"Model: BiLSTM + {'CRF' if model_config.use_crf else 'Linear'}")
        print(f"Distributed evaluation on {world_size} GPU(s)")
    
    # Load dataset
    if rank == 0:
        print(f"\nLoading test data from {test_data_path}")
    
    test_dataset = ChineseTextDataset(
        data_path=test_data_path,
        label_config=label_config,
        max_length=512
    )
    
    if rank == 0:
        print(f"Test examples: {len(test_dataset)}")
    
    # Create dataloader (distributed)
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            sampler=test_sampler,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
    
    # Create model
    model = create_model(
        vocab_size=test_dataset.vocab_size,
        num_labels=label_config.num_labels,
        model_config=model_config
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Wrap with DDP if distributed
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    model.eval()
    
    if rank == 0:
        print(f"Using device: cuda:{rank}")
        print(f"World size: {world_size}")
    
    # Evaluate
    if rank == 0:
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
    
    # Collect predictions from all GPUs
    predictions, labels = collect_predictions(
        model.module if world_size > 1 else model,
        test_loader,
        device,
        use_crf=model_config.use_crf
    )
    
    # Gather results from all ranks (only rank 0 will have complete results)
    if world_size > 1:
        # Gather predictions and labels from all ranks
        all_predictions = [None] * world_size
        all_labels = [None] * world_size
        
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_labels, labels)
        
        if rank == 0:
            # Concatenate results from all ranks
            import numpy as np
            predictions = np.concatenate(all_predictions)
            labels = np.concatenate(all_labels)
    
    # Only rank 0 computes metrics and saves results
    if rank == 0:
        # Compute metrics
        metrics_calc = MetricsCalculator(label_config)
        metrics = metrics_calc.compute_metrics(predictions, labels)
        
        # Print metrics
        print(f"\nTest Results:")
        print(f"{'='*70}")
        
        overall = metrics['overall']
        print(f"Overall - P: {overall['precision']:.4f}, "
              f"R: {overall['recall']:.4f}, F1: {overall['f1']:.4f}")
        print(f"Total samples: {overall['total_samples']}")
        
        print(f"\nPer-class metrics:")
        print(f"{'-'*70}")
        for label_name, label_metrics in metrics['per_class'].items():
            print(f"{label_name:>5} - P: {label_metrics['precision']:.4f}, "
                  f"R: {label_metrics['recall']:.4f}, "
                  f"F1: {label_metrics['f1']:.4f}, "
                  f"Support: {label_metrics['support']}")
        
        # Save metrics
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'test_metrics.json')
        
        output = {
            'checkpoint': checkpoint_path,
            'task_type': task_type,
            'epoch': checkpoint['epoch'],
            'num_gpus': world_size,
            'metrics': metrics
        }
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved metrics to {metrics_path}")
        
        # Generate samples
        print("\n" + "="*70)
        print(f"GENERATING {num_samples} SAMPLES")
        print("="*70)
        
        num_samples = min(num_samples, len(test_dataset))
        sample_indices = random.sample(range(len(test_dataset)), num_samples)
        
        samples_output = []
        samples_output.append("="*70)
        samples_output.append(f"TEST SAMPLES - {task_type.upper()}")
        samples_output.append("="*70)
        
        with torch.no_grad():
            for idx, sample_idx in enumerate(sample_indices, 1):
                sample = test_dataset[sample_idx]
                
                input_ids = sample['input_ids'].unsqueeze(0).to(device)
                label_ids = sample['label_ids']
                length = sample['length']
                lengths = torch.tensor([length]).to(device)
                
                # Predict
                outputs = (model.module if world_size > 1 else model)(input_ids, lengths)
                
                if model_config.use_crf:
                    preds = outputs['predictions'][0]
                else:
                    logits = outputs['logits']
                    preds = torch.argmax(logits, dim=-1)[0][:length].cpu().numpy()
                
                # Get chars and labels
                chars = [test_dataset.id2char[cid.item()] for cid in input_ids[0][:length]]
                true_labels = label_ids[:length].numpy()
                
                # Format output
                samples_output.append(f"\nSample {idx}/{num_samples} (Index: {sample_idx})")
                samples_output.append("-"*70)
                
                # Detailed comparison
                samples_output.append(decode_predictions(
                    chars, preds, true_labels, label_config.id2label
                ))
                
                # Generated text
                samples_output.append("\nPredicted:")
                samples_output.append(generate_text_with_labels(
                    chars, preds, label_config.id2label, task_type
                ))
                
                # True text
                samples_output.append("\nGround Truth:")
                samples_output.append(generate_text_with_labels(
                    chars, true_labels, label_config.id2label, task_type
                ))
                
                samples_output.append("="*70)
        
        # Print and save samples
        result_text = "\n".join(samples_output)
        print(result_text)
        
        samples_path = os.path.join(output_dir, 'test_samples.txt')
        with open(samples_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        
        print(f"\nSaved samples to {samples_path}")
        print("\nEvaluation completed!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model (Distributed)')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data (JSONL)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to display')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size = setup_distributed()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Evaluate
    try:
        evaluate_and_sample(
            checkpoint_path=args.checkpoint,
            test_data_path=args.test_data,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            rank=rank,
            world_size=world_size
        )
    finally:
        # Cleanup
        if world_size > 1:
            cleanup_distributed()


if __name__ == '__main__':
    main()
