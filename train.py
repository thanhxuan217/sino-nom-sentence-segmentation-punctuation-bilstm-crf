# train.py
"""
Main training script
Usage:
    # Single GPU
    python train.py --task segmentation --train_data data/train.jsonl --val_data data/val.jsonl --test_data data/test.jsonl
    
    # Multi-GPU với torchrun
    torchrun --nproc_per_node=2 train.py --task segmentation --train_data data/train.jsonl --val_data data/val.jsonl --test_data data/test.jsonl
    
    # Resume training
    python train.py --task segmentation --train_data data/train.jsonl --val_data data/val.jsonl --test_data data/test.jsonl --resume checkpoints/latest_checkpoint.pt
"""

import argparse
import os
import sys
import torch

from src.config import TrainingConfig, ModelConfig, LabelConfig
from src.dataset import ChineseTextDataset, create_dataloaders
from src.model import create_model
from src.trainer import Trainer, setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description='Train BiLSTM model for Classical Chinese')
    
    # Task
    parser.add_argument('--task', type=str, required=True,
                        choices=['punctuation', 'segmentation'],
                        help='Task type')
    
    # Data
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (JSONL)')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data (JSONL)')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data (JSONL)')
    
    # Model
    parser.add_argument('--use_crf', action='store_true',
                        help='Use CRF instead of Linear head')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=5.0,
                        help='Gradient clipping')
    
    # Paths
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get distributed info
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Configurations
    label_config = LabelConfig(task_type=args.task)
    
    model_config = ModelConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_crf=args.use_crf
    )
    
    training_config = TrainingConfig(
        task_type=args.task,
        train_data=args.train_data,
        val_data=args.val_data,
        test_data=args.test_data,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_from=args.resume,
        num_workers=args.num_workers
    )
    
    if rank == 0:
        print("="*70)
        print(f"Training Configuration:")
        print(f"  Task: {args.task}")
        print(f"  Model: BiLSTM + {'CRF' if args.use_crf else 'Linear'}")
        print(f"  Distributed: {world_size} GPU(s)")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Epochs: {args.num_epochs}")
        if args.resume:
            print(f"  Resume from: {args.resume}")
        print("="*70)
    
    # Load datasets
    if rank == 0:
        print("\nLoading datasets...")
    
    train_dataset = ChineseTextDataset(
        data_path=args.train_data,
        label_config=label_config,
        max_length=512
    )
    
    # Share vocab với val và test
    vocab = train_dataset.get_vocab()
    
    val_dataset = ChineseTextDataset(
        data_path=args.val_data,
        label_config=label_config,
        max_length=512,
        vocab=vocab
    )
    
    test_dataset = ChineseTextDataset(
        data_path=args.test_data,
        label_config=label_config,
        max_length=512,
        vocab=vocab
    )
    
    if rank == 0:
        print(f"  Train: {len(train_dataset)} examples")
        print(f"  Val: {len(val_dataset)} examples")
        print(f"  Test: {len(test_dataset)} examples")
        print(f"  Vocab size: {train_dataset.vocab_size}")
        print(f"  Num labels: {label_config.num_labels}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size
    )
    
    # Create model
    model = create_model(
        vocab_size=train_dataset.vocab_size,
        num_labels=label_config.num_labels,
        model_config=model_config
    )
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model parameters: {num_params:,}")
    
    # Save configs
    if rank == 0:
        label_config.save(os.path.join(args.save_dir, 'label_config.json'))
        training_config.save(os.path.join(args.save_dir, 'training_config.json'))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_config=label_config,
        training_config=training_config,
        model_config=model_config,
        rank=rank,
        world_size=world_size
    )
    
    # Train
    if rank == 0:
        print("\nStarting training...\n")
    
    trainer.train()
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()


if __name__ == '__main__':
    main()
