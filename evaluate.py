# evaluate.py
"""
Evaluation script với distributed support và Parquet streaming
Usage:
    # Single GPU
    python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data/ --vocab_path vocab.json

    # Multi-GPU với torchrun (2 GPUs)
    torchrun --nproc_per_node=2 evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data/ --vocab_path vocab.json
"""

import argparse
import os
import torch
import torch.distributed as dist
import random
import json
from typing import List

from src.config import LabelConfig, ModelConfig
from src.dataset import load_vocab, ParquetStreamingDataset, collate_fn
from src.model import create_model
from src.metrics import MetricsCalculator, collect_and_compute_metrics
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


def print_checkpoint_info(checkpoint_path: str):
    """Print checkpoint details (moved from slurm script)"""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print("Model Information:")
        print(f"  Trained Epoch: {ckpt['epoch']}")
        print(f"  Best Validation F1: {ckpt['best_val_f1']:.4f}")
        print(f"  Best Epoch: {ckpt['best_epoch']}")
        print(f"  Task: {ckpt['training_config']['task_type']}")
        print(f"  Model Type: BiLSTM + {'CRF' if ckpt['model_config']['use_crf'] else 'Linear'}")
    except Exception as e:
        print(f"Warning: Could not load checkpoint info: {e}")
    print("")


def print_metrics_summary(metrics_path: str):
    """Print performance summary from saved metrics JSON (moved from slurm script)"""
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        overall = metrics['metrics']['overall']
        print("Performance Summary:")
        print(f"  Overall F1: {overall['f1']:.4f}")
        print(f"  Precision:  {overall['precision']:.4f}")
        print(f"  Recall:     {overall['recall']:.4f}")
        print(f"  Samples:    {overall['total_samples']}")
        print(f"  GPUs used:  {metrics['num_gpus']}")
    except Exception as e:
        print(f"Warning: Could not load metrics summary: {e}")
    print("")


def generate_sample_results(
    data_dir: str,
    test_split: str,
    label_config,
    vocab: dict,
    model,
    device,
    use_crf: bool,
    task_type: str,
    output_dir: str,
    world_size: int = 1,
    max_samples: int = 100,
    seed: int = 42,
):
    """Generate structured sample results with gold/pred labels and labeled text.
    
    Outputs a JSON file with up to max_samples entries, each containing:
      - idx, text, gold_labels, pred_labels, gold_text_labeled, pred_text_labeled
    """
    sample_dataset = ParquetStreamingDataset(
        data_dir=data_dir,
        split=test_split,
        label_config=label_config,
        vocab=vocab,
        max_length=512,
        shuffle_buffer=1000,
        seed=seed,
    )

    id2char_vocab = vocab['id2char']
    results = []

    with torch.no_grad():
        for idx, sample in enumerate(sample_dataset):
            if idx >= max_samples:
                break

            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            label_ids = sample['label_ids']
            length = sample['length']
            lengths = torch.tensor([length]).to(device)

            # Predict
            outputs = (model.module if world_size > 1 else model)(input_ids, lengths)

            if use_crf:
                preds = outputs['predictions'][0]
            else:
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=-1)[0][:length].cpu().numpy()

            # Get chars and labels
            chars = [id2char_vocab.get(cid.item(), '?') for cid in input_ids[0][:length]]
            true_labels_arr = label_ids[:length].numpy()

            text = ''.join(chars)
            gold_labels = [label_config.id2label[int(lid)] for lid in true_labels_arr]
            pred_labels = [label_config.id2label[int(pid)] for pid in preds]

            gold_text = generate_text_with_labels(
                chars, true_labels_arr, label_config.id2label, task_type
            )
            pred_text = generate_text_with_labels(
                chars, preds, label_config.id2label, task_type
            )

            result = {
                "idx": idx,
                "text": text,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "gold_text_labeled": gold_text,
                "pred_text_labeled": pred_text,
            }
            results.append(result)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'sample_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} sample results to {results_path}")
    return results


def evaluate_and_sample(
    checkpoint_path: str,
    data_dir: str,
    vocab_path: str,
    test_split: str = 'test',
    task: str = 'test',
    num_samples: int = 50,
    output_dir: str = 'evaluation_results',
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0
):
    """Evaluate model và generate samples (distributed)"""

    # Only rank 0 prints
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
        print_checkpoint_info(checkpoint_path)

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

    # Load vocab
    if rank == 0:
        print(f"\nLoading vocab from {vocab_path}")
    vocab = load_vocab(vocab_path)
    vocab_size = vocab.get('vocab_size', len(vocab['char2id']))

    # Load streaming test dataset
    if rank == 0:
        print(f"Loading test data from {data_dir}/{test_split}/")

    test_dataset = ParquetStreamingDataset(
        data_dir=data_dir,
        split=test_split,
        label_config=label_config,
        vocab=vocab,
        max_length=512,
        shuffle_buffer=0,
        no_shard=True,  # rank=0 đọc toàn bộ test set (không chia shard theo GPU)
    )

    if rank == 0:
        print(f"Vocab size: {vocab_size}")

    # Create dataloader (streaming - không cần DistributedSampler)
    # num_workers=0 by default để tránh IterableDataset bị shard sai
    # (mỗi worker tạo ra một iterator độc lập → dữ liệu bị đọc lặp nhiều lần)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(num_workers > 0),
    )

    # Create model
    model = create_model(
        vocab_size=vocab_size,
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
        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)

    # Streaming evaluation: cộng dồn confusion matrix, không lưu toàn bộ lên RAM
    metrics_calc = MetricsCalculator(label_config)
    metrics = collect_and_compute_metrics(
        model.module if world_size > 1 else model,
        test_loader,
        device,
        metrics_calc,
        use_crf=model_config.use_crf
    )

    # Only rank 0 prints metrics and saves results
    if rank == 0:
        # Print metrics
        print(f"\nTest Results:")
        print(f"{'=' * 70}")

        overall = metrics['overall']
        print(f"Overall - P: {overall['precision']:.4f}, "
              f"R: {overall['recall']:.4f}, F1: {overall['f1']:.4f}")
        print(f"Total samples: {overall['total_samples']}")

        print(f"\nPer-class metrics:")
        print(f"{'-' * 70}")
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

        # Only generate samples and structured results for test set
        if task == 'test':
            # Generate samples bằng cách lấy từ streaming
            print("\n" + "=" * 70)
            print(f"GENERATING {num_samples} SAMPLES")
            print("=" * 70)

            # Thu thập một số samples từ streaming dataset
            sample_dataset = ParquetStreamingDataset(
                data_dir=data_dir,
                split=test_split,
                label_config=label_config,
                vocab=vocab,
                max_length=512,
                shuffle_buffer=1000,  # Shuffle nhẹ để lấy samples đa dạng
                seed=42,
            )

            samples_output = []
            samples_output.append("=" * 70)
            samples_output.append(f"TEST SAMPLES - {task_type.upper()}")
            samples_output.append("=" * 70)

            char2id = vocab['char2id']
            id2char_vocab = vocab['id2char']

            with torch.no_grad():
                sample_count = 0
                for sample in sample_dataset:
                    if sample_count >= num_samples:
                        break

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
                    chars = [id2char_vocab.get(cid.item(), '?') for cid in input_ids[0][:length]]
                    true_labels_arr = label_ids[:length].numpy()

                    sample_count += 1

                    # Format output
                    samples_output.append(f"\nSample {sample_count}/{num_samples}")
                    samples_output.append("-" * 70)

                    # Detailed comparison
                    samples_output.append(decode_predictions(
                        chars, preds, true_labels_arr, label_config.id2label
                    ))

                    # Generated text
                    samples_output.append("\nPredicted:")
                    samples_output.append(generate_text_with_labels(
                        chars, preds, label_config.id2label, task_type
                    ))

                    # True text
                    samples_output.append("\nGround Truth:")
                    samples_output.append(generate_text_with_labels(
                        chars, true_labels_arr, label_config.id2label, task_type
                    ))

                    samples_output.append("=" * 70)

            # Print and save samples
            result_text = "\n".join(samples_output)
            print(result_text)

            samples_path = os.path.join(output_dir, 'test_samples.txt')
            with open(samples_path, 'w', encoding='utf-8') as f:
                f.write(result_text)

            print(f"\nSaved samples to {samples_path}")

            # Generate structured sample results (max 100)
            print("\n" + "=" * 70)
            print("GENERATING STRUCTURED SAMPLE RESULTS (max 100)")
            print("=" * 70)
            generate_sample_results(
                data_dir=data_dir,
                test_split=test_split,
                label_config=label_config,
                vocab=vocab,
                model=model,
                device=device,
                use_crf=model_config.use_crf,
                task_type=task_type,
                output_dir=output_dir,
                world_size=world_size,
                max_samples=100,
                seed=42,
            )
        else:
            print(f"\nSkipping sample generation (task='{task}', only runs for task='test')")

        # Print metrics summary
        print_metrics_summary(metrics_path)

        print("\nEvaluation completed!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model (Distributed)')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Thư mục gốc chứa data')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Đường dẫn đến vocab.json')
    parser.add_argument('--test_split', type=str, default='test',
                        help='Tên split test (default: test)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to display')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--task', type=str, default='test',
                        choices=['test', 'val'],
                        help='Evaluate on test or val set (samples only generated for test)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader workers (default: 0, safe for IterableDataset)')
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
            data_dir=args.data_dir,
            vocab_path=args.vocab_path,
            test_split=args.test_split,
            task=args.task,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            rank=rank,
            world_size=world_size,
            num_workers=args.num_workers
        )
    finally:
        # Cleanup
        if world_size > 1:
            cleanup_distributed()


if __name__ == '__main__':
    main()
