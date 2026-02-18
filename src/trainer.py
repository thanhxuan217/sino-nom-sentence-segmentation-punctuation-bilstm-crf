# src/trainer.py
"""
Trainer với distributed training, AMP, gradient accumulation, và resume support
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Optional

from .metrics import MetricsCalculator, collect_and_compute_metrics


class Trainer:
    """Trainer với AMP, gradient accumulation, và resume support"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        full_val_loader,
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
        self.full_val_loader = full_val_loader
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
        
        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(
            training_config, 'gradient_accumulation_steps', 1
        )
        
        # Mixed Precision
        self.use_amp = getattr(training_config, 'use_amp', False)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # OneCycleLR scheduler (replaces warmup + ReduceLROnPlateau)
        # Estimate total steps for OneCycleLR
        self.warmup_steps = getattr(training_config, 'warmup_steps', 2000)
        # We'll create the scheduler after we know the number of batches
        self.scheduler = None
        self.global_step = 0
        
        # Intra-epoch checkpoint
        self.save_every_n_steps = getattr(training_config, 'save_every_n_steps', 0)
        
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
    
    def _create_scheduler(self, steps_per_epoch: int):
        """Create OneCycleLR scheduler based on actual steps per epoch."""
        total_steps = steps_per_epoch * self.training_config.num_epochs
        pct_start = min(self.warmup_steps / max(total_steps, 1), 0.3)
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.training_config.learning_rate,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy='cos',
            div_factor=25.0,       # initial_lr = max_lr / 25
            final_div_factor=1e4,  # final_lr = initial_lr / 1e4
        )
        
        if self.rank == 0:
            print(f"  OneCycleLR: total_steps={total_steps}, "
                  f"warmup={int(pct_start * total_steps)} steps, "
                  f"max_lr={self.training_config.learning_rate}")
    
    def save_checkpoint(self, epoch: int, val_f1: float, is_best: bool = False,
                        step: Optional[int] = None):
        """Lưu checkpoint"""
        if self.rank != 0:
            return
        
        model_to_save = self.model.module if self.world_size > 1 else self.model
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'training_config': self.training_config.to_dict(),
            'model_config': vars(self.model_config)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save with step info if intra-epoch
        if step is not None:
            checkpoint_path = os.path.join(
                self.training_config.save_dir,
                f'checkpoint_epoch_{epoch}_step_{step}.pt'
            )
        else:
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
            print(f"✓ Saved best model (F1: {val_f1:.4f})")
    
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
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        if self.rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}, step {self.global_step}")
            print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
    
    def train_epoch(self, epoch: int, total_batches: Optional[int] = None) -> float:
        """Train một epoch với AMP và gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Set epoch cho sampler hoặc streaming dataset
        if self.world_size > 1 and hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        # Set epoch cho streaming dataset (ParquetStreamingDataset)
        if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'set_epoch'):
            self.train_dataset.set_epoch(epoch)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=(self.rank != 0),
            total=total_batches
        )
        
        self.optimizer.zero_grad()
        
        step = -1
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Forward with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(input_ids, lengths, label_ids)
                loss = outputs['loss']
                # Scale loss by accumulation steps
                loss = loss / self.gradient_accumulation_steps
            
            # Backward with scaler
            self.scaler.scale(loss).backward()
            
            # Accumulate and step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (unscale first for proper clipping)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.gradient_clip
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Intra-epoch checkpoint
                if (self.save_every_n_steps > 0 and 
                    self.global_step % self.save_every_n_steps == 0):
                    if self.rank == 0:
                        print(f"\n💾 Saving intra-epoch checkpoint at step {self.global_step}...")
                    self.save_checkpoint(epoch, self.best_val_f1, step=self.global_step)
            
            # Track loss (unscaled)
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss / num_batches:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Logging
            if self.rank == 0 and step % self.training_config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - epoch_start_time
                if total_batches and total_batches > 0:
                    pct = (step + 1) / total_batches * 100
                    eta_seconds = elapsed / (step + 1) * (total_batches - step - 1)
                    eta_h = int(eta_seconds // 3600)
                    eta_m = int((eta_seconds % 3600) // 60)
                    eta_s = int(eta_seconds % 60)
                    print(f"Epoch {epoch}, Step {step}/{total_batches} [{pct:.1f}%], "
                          f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, "
                          f"LR: {current_lr:.2e}, ETA: {eta_h}h{eta_m:02d}m{eta_s:02d}s")
                else:
                    elapsed_h = int(elapsed // 3600)
                    elapsed_m = int((elapsed % 3600) // 60)
                    elapsed_s = int(elapsed % 60)
                    print(f"Epoch {epoch}, Step {step}, "
                          f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, "
                          f"LR: {current_lr:.2e}, Elapsed: {elapsed_h}h{elapsed_m:02d}m{elapsed_s:02d}s")
        
        # Handle remaining gradients if steps not divisible by accumulation
        if (step + 1) % self.gradient_accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader, split_name: str = "Val") -> Optional[Dict]:
        """Evaluate"""
        if self.rank != 0:
            return None
        
        print(f"\n{'='*60}")
        print(f"Evaluating on {split_name} set...")
        print(f"{'='*60}")
        
        # Streaming evaluation: cộng dồn confusion matrix, không lưu toàn bộ lên RAM
        metrics = collect_and_compute_metrics(
            self.model.module if self.world_size > 1 else self.model,
            dataloader,
            self.device,
            self.metrics_calculator,
            use_crf=self.model_config.use_crf
        )
        
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
            'global_step': self.global_step,
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
        # Determine steps per epoch and total batches for tqdm
        try:
            num_batches = len(self.train_loader)
            steps_per_epoch = num_batches // self.gradient_accumulation_steps
        except TypeError:
            # IterableDataset - estimate based on dataset size
            # Use a reasonable estimate; OneCycleLR will still work
            steps_per_epoch = 50000  # Conservative estimate
            num_batches = steps_per_epoch * self.gradient_accumulation_steps
            if self.rank == 0:
                print(f"⚠ IterableDataset: estimating {steps_per_epoch} steps/epoch")

        # Create scheduler on first call (need to know steps_per_epoch)
        if self.scheduler is None:
            self._create_scheduler(steps_per_epoch)
            
            # If resuming, fast-forward scheduler
            if self.global_step > 0 and self.scheduler is not None:
                for _ in range(self.global_step):
                    self.scheduler.step()
                if self.rank == 0:
                    print(f"  Fast-forwarded scheduler to step {self.global_step}")
        
        for epoch in range(self.start_epoch, self.training_config.num_epochs + 1):
            # Train
            avg_loss = self.train_epoch(epoch, total_batches=num_batches)
            
            if self.rank == 0:
                print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.4f}")
            
            # Evaluate
            if epoch % self.training_config.eval_interval == 0:
                val_metrics = self.evaluate(self.val_loader, "Validation")
                
                if self.rank == 0 and val_metrics:
                    val_f1 = val_metrics['overall']['f1']
                    
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
                        print(f"\n⏹ Early stopping after {epoch} epochs")
                        print(f"Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
                        break
        
        # Final full validation evaluation
        if self.rank == 0 and self.full_val_loader is not None:
            print(f"\n{'='*60}")
            print(f"Running FULL validation evaluation...")
            print(f"{'='*60}")
            full_val_metrics = self.evaluate(self.full_val_loader, "Full Validation")
            if full_val_metrics:
                self.save_metrics(full_val_metrics, 'full_val', self.best_epoch)

        # Final summary
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Best epoch: {self.best_epoch}, Best F1: {self.best_val_f1:.4f}")
            print(f"Total steps: {self.global_step}")
            print(f"{'='*60}")


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training"""
    
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
