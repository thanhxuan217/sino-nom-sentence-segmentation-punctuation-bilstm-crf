# train.py
"""
Main training script với Parquet streaming support
Usage:
    # Single GPU
    python train.py --task segmentation --data_dir data/ --vocab_path vocab.json

    # Multi-GPU với torchrun
    torchrun --nproc_per_node=2 train.py --task segmentation --data_dir data/ --vocab_path vocab.json

    # Resume training
    python train.py --task segmentation --data_dir data/ --vocab_path vocab.json --resume checkpoints/latest_checkpoint.pt
"""

import argparse
import os
import sys
import torch

from src.config import TrainingConfig, ModelConfig, LabelConfig
from src.dataset import load_vocab, create_streaming_dataloaders
from src.model import create_model
from src.trainer import Trainer, setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description='Train BiLSTM model for Classical Chinese')

    # Task
    parser.add_argument('--task', type=str, required=True,
                        choices=['punctuation', 'segmentation'],
                        help='Task type')

    # Data (Parquet streaming)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Thư mục gốc chứa data (có sub-folders train/, val/, test/)')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Đường dẫn đến vocab.json')
    parser.add_argument('--train_split', type=str, default='train',
                        help='Tên split training (default: train)')
    parser.add_argument('--val_split', type=str, default='val',
                        help='Tên split validation (default: val)')
    parser.add_argument('--test_split', type=str, default='test',
                        help='Tên split test (default: test)')
    parser.add_argument('--shuffle_buffer', type=int, default=10000,
                        help='Kích thước shuffle buffer (default: 10000)')
    parser.add_argument('--val_max_samples', type=int, default=6000,
                        help='Số mẫu validation tối đa mỗi epoch (0 = không giới hạn, default: 6000)')

    # Model
    parser.add_argument('--use_crf', action='store_true',
                        help='Use CRF instead of Linear head')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Max training steps per epoch (-1 = unlimited)')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=5.0,
                        help='Gradient clipping')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Number of warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--save_every_n_steps', type=int, default=5000,
                        help='Save intra-epoch checkpoint every N steps')

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
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience')

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
        train_data=args.data_dir,
        val_data=args.data_dir,
        test_data=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        max_length=args.max_length,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_from=args.resume,
        num_workers=args.num_workers,
        save_every_n_steps=args.save_every_n_steps,
        early_stopping_patience=args.early_stopping_patience
    )

    if rank == 0:
        print("=" * 70)
        print(f"Training Configuration:")
        print(f"  Task: {args.task}")
        print(f"  Model: BiLSTM + {'CRF' if args.use_crf else 'Linear'}")
        print(f"  Distributed: {world_size} GPU(s)")
        print(f"  Data dir: {args.data_dir}")
        print(f"  Vocab: {args.vocab_path}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Max length: {args.max_length}")
        print(f"  Warmup steps: {args.warmup_steps}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  AMP: {args.use_amp}")
        print(f"  Shuffle buffer: {args.shuffle_buffer}")
        print(f"  Val max samples: {args.val_max_samples} {'(unlimited)' if args.val_max_samples == 0 else ''}")
        print(f"  Streaming: True (Parquet)")
        if args.resume:
            print(f"  Resume from: {args.resume}")
        print("=" * 70)

    # Load vocab
    if rank == 0:
        print("\nLoading vocabulary...")

    vocab = load_vocab(args.vocab_path)
    vocab_size = vocab.get('vocab_size', len(vocab['char2id']))

    if rank == 0:
        print(f"  Vocab size: {vocab_size}")
        print(f"  Num labels: {label_config.num_labels}")

    # Create streaming dataloaders
    if rank == 0:
        print("\nCreating streaming dataloaders...")

    train_loader, val_loader, full_val_loader, test_loader, train_dataset, _ = create_streaming_dataloaders(
        data_dir=args.data_dir,
        label_config=label_config,
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        val_max_samples=args.val_max_samples,
    )

    if rank == 0:
        print("  Streaming dataloaders created successfully")
        if args.val_max_samples > 0:
            print(f"  Val loader: limited to {args.val_max_samples} samples per epoch")
            print(f"  Full val loader: all samples (after training)")

    # Create model
    model = create_model(
        vocab_size=vocab_size,
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

        # Lưu vocab path vào checkpoint dir để evaluate dùng lại
        import shutil
        vocab_save_path = os.path.join(args.save_dir, 'vocab.json')
        if not os.path.exists(vocab_save_path):
            shutil.copy2(args.vocab_path, vocab_save_path)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        full_val_loader=full_val_loader,
        test_loader=test_loader,
        label_config=label_config,
        training_config=training_config,
        model_config=model_config,
        rank=rank,
        world_size=world_size
    )

    # Store train_dataset reference for set_epoch
    trainer.train_dataset = train_dataset

    # Train
    if rank == 0:
        print("\nStarting training...\n")

    trainer.train()

    # Cleanup
    if world_size > 1:
        cleanup_distributed()


if __name__ == '__main__':
    main()
