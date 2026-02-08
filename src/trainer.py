# src/trainer.py
"""
Trainer với distributed training và resume support
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
from pathlib import Path

from .metrics import MetricsCalculator, collect_predictions


class Trainer:
    """Trainer với resume support"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        label_config,
        training_config,
        model_config,
        rank: int = 0,
        world_size: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_config = label_config
        self.training_config = training_config
        self.model_config = model_config
        self.rank = rank
        self.world_size = world_size
        
        # Device
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Wrap với DDP
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank
            )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=(rank == 0)
        )
        
        # Metrics
        self.metrics_calculator = MetricsCalculator(label_config)
        
        # Training state
        self.start_epoch = 1
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Resume nếu có checkpoint
        if training_config.resume_from:
            self.resume_from_checkpoint(training_config.resume_from)
    
    def save_checkpoint(self, epoch: int, val_f1: float, is_best: bool = False):
        """Lưu checkpoint"""
        if self.rank != 0:
            return
        
        model_to_save = self.model.module if self.world_size > 1 else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'training_config': self.training_config.to_dict(),
            'model_config': vars(self.model_config)
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.training_config.save_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest as resumable
        latest_path = os.path.join(
            self.training_config.save_dir,
            'latest_checkpoint.pt'
        )
        torch.save(checkpoint, latest_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.training_config.save_dir,
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model (F1: {val_f1:.4f})")
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training từ checkpoint"""
        if self.rank == 0:
            print(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer và scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        if self.rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train một epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Set epoch cho sampler
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=(self.rank != 0)
        )
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Forward
            outputs = self.model(input_ids, lengths, label_ids)
            loss = outputs['loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if self.rank == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / num_batches:.4f}'
                })
            
            # Logging
            if self.rank == 0 and step % self.training_config.log_interval == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader, split_name: str = "Val") -> Dict:
        """Evaluate"""
        if self.rank != 0:
            return None
        
        print(f"\n{'='*60}")
        print(f"Evaluating on {split_name} set...")
        print(f"{'='*60}")
        
        # Collect predictions
        predictions, labels = collect_predictions(
            self.model.module if self.world_size > 1 else self.model,
            dataloader,
            self.device,
            use_crf=self.model_config.use_crf
        )
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_metrics(predictions, labels)
        
        # Print
        self._print_metrics(metrics, split_name)
        
        return metrics
    
    def _print_metrics(self, metrics: Dict, split_name: str):
        """Print metrics"""
        print(f"\n{split_name} Results:")
        print(f"{'='*60}")
        
        overall = metrics['overall']
        print(f"Overall - P: {overall['precision']:.4f}, "
              f"R: {overall['recall']:.4f}, F1: {overall['f1']:.4f}")
        print(f"Total samples: {overall['total_samples']}")
        
        print(f"\nPer-class metrics:")
        print(f"{'-'*60}")
        for label_name, label_metrics in metrics['per_class'].items():
            print(f"{label_name:>5} - P: {label_metrics['precision']:.4f}, "
                  f"R: {label_metrics['recall']:.4f}, "
                  f"F1: {label_metrics['f1']:.4f}, "
                  f"Support: {label_metrics['support']}")
        print(f"{'='*60}\n")
    
    def save_metrics(self, metrics: Dict, split: str, epoch: int):
        """Lưu metrics ra JSON"""
        if self.rank != 0:
            return
        
        output = {
            'epoch': epoch,
            'split': split,
            'task_type': self.label_config.task_type,
            'training_config': self.training_config.to_dict(),
            'model_config': vars(self.model_config),
            'metrics': metrics
        }
        
        save_path = os.path.join(
            self.training_config.log_dir,
            f'{split}_metrics_epoch_{epoch}.json'
        )
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"Saved metrics to {save_path}")
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.training_config.num_epochs + 1):
            # Train
            avg_loss = self.train_epoch(epoch)
            
            if self.rank == 0:
                print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.4f}")
            
            # Evaluate
            if epoch % self.training_config.eval_interval == 0:
                val_metrics = self.evaluate(self.val_loader, "Validation")
                
                if self.rank == 0 and val_metrics:
                    val_f1 = val_metrics['overall']['f1']
                    
                    # Update scheduler
                    self.scheduler.step(val_f1)
                    
                    # Check best
                    is_best = val_f1 > self.best_val_f1
                    if is_best:
                        self.best_val_f1 = val_f1
                        self.best_epoch = epoch
                        self.patience_counter = 0
                        print(f"✓ New best model! F1: {val_f1:.4f}")
                    else:
                        self.patience_counter += 1
                    
                    # Save checkpoint
                    if epoch % self.training_config.save_interval == 0 or is_best:
                        self.save_checkpoint(epoch, val_f1, is_best)
                    
                    # Save metrics
                    self.save_metrics(val_metrics, 'val', epoch)
                    
                    # Early stopping
                    if self.patience_counter >= self.training_config.early_stopping_patience:
                        print(f"\nEarly stopping after {epoch} epochs")
                        print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
                        break
        
        # Final test
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Best epoch: {self.best_epoch}, Best F1: {self.best_val_f1:.4f}")
            print(f"{'='*60}")


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed"""
    dist.destroy_process_group()
