"""
Training loop and utilities for Graph JEPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from jepa_graph.models.graph_jepa import GraphJEPA
from jepa_graph.training.losses import JEPALoss, CombinedLoss


@dataclass
class TrainingConfig:
    """Configuration for JEPA training."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    max_epochs: int = 100
    batch_size: int = 32
    gradient_clip: float = 1.0
    lr_schedule: str = "cosine"
    min_lr: float = 1e-6
    loss_type: str = "mse"
    loss_normalize: bool = True
    vicreg_weight: float = 0.0
    ema_decay: float = 0.996
    ema_decay_end: float = 1.0
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 10
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    use_wandb: bool = False
    wandb_project: str = "jepa-graph"
    wandb_run_name: Optional[str] = None


class JEPATrainer:
    """Trainer for Graph JEPA pretraining."""
    
    def __init__(
        self,
        model: GraphJEPA,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.device)
        
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or self._create_scheduler()
        
        self.loss_fn = CombinedLoss(
            jepa_loss=JEPALoss(
                loss_type=config.loss_type,
                normalize=config.loss_normalize,
            ),
            vicreg_weight=config.vicreg_weight,
        )
        
        self.scaler = torch.amp.GradScaler() if config.mixed_precision else None
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics_history: List[Dict[str, float]] = []
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
        
        if config.use_wandb:
            self._setup_wandb()
    
    def _create_optimizer(self) -> Optimizer:
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(param_groups, lr=self.config.learning_rate)
    
    def _create_scheduler(self) -> LRScheduler:
        if self.config.lr_schedule == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.min_lr,
            )
        else:
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda _: 1.0
            )
    
    def _setup_wandb(self):
        try:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__,
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            self.config.use_wandb = False
    
    def train(self) -> Dict[str, Any]:
        print(f"Starting training for {self.config.max_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches per epoch: {len(self.train_loader)}")
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            train_metrics = self._train_epoch()
            
            if self.val_loader and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self._validate()
                
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint("best.pt")
            else:
                val_metrics = {}
            
            self.scheduler.step()
            
            metrics = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.metrics_history.append(metrics)
            
            self._print_metrics(epoch, train_metrics, val_metrics)
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")
        
        self.save_checkpoint("final.pt")
        
        return {
            "history": self.metrics_history,
            "best_val_loss": self.best_val_loss,
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            context_data = batch["context_data"].to(self.device)
            target_data = batch["target_data"].to(self.device)
            
            with torch.amp.autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
                predicted, target, model_loss = self.model(
                    context_data=context_data,
                    target_data=target_data,
                )
                
                loss, components = self.loss_fn(predicted, target)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()
            
            self.model.update_target_encoder()
            
            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0) + v
            num_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {"loss": avg_loss, **avg_components}
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            context_data = batch["context_data"].to(self.device)
            target_data = batch["target_data"].to(self.device)
            
            predicted, target, _ = self.model(
                context_data=context_data,
                target_data=target_data,
            )
            
            loss, _ = self.loss_fn(predicted, target)
            total_loss += loss.item()
            num_batches += 1
        
        return {"loss": total_loss / num_batches}
    
    def _print_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        msg = f"Epoch {epoch+1}/{self.config.max_epochs}"
        msg += f" | Train Loss: {train_metrics['loss']:.4f}"
        
        if val_metrics:
            msg += f" | Val Loss: {val_metrics['loss']:.4f}"
        
        msg += f" | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        
        print(msg)
        
        if self.config.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics.get("loss"),
                "lr": self.optimizer.param_groups[0]["lr"],
            })
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"Resumed from epoch {checkpoint['epoch']}")


def main():
    """Entry point for training script."""
    import argparse
    from torch_geometric.datasets import TUDataset
    from jepa_graph.models.graph_jepa import GraphJEPAConfig
    from jepa_graph.data.dataset import create_jepa_dataloader
    from jepa_graph.data.masking import GraphMasker, MaskingStrategy
    
    parser = argparse.ArgumentParser(description="Train Graph JEPA")
    parser.add_argument("--dataset", type=str, default="PROTEINS")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-type", type=str, default="gat")
    parser.add_argument("--masking", type=str, default="khop")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    dataset = TUDataset(root="/tmp/TU", name=args.dataset)
    in_channels = dataset[0].x.size(1) if dataset[0].x is not None else 1
    
    strategy_map = {
        "khop": MaskingStrategy.KHOP_BALL,
        "random": MaskingStrategy.RANDOM_NODES,
        "subgraph": MaskingStrategy.RANDOM_SUBGRAPH,
    }
    masker = GraphMasker(strategy=strategy_map.get(args.masking, MaskingStrategy.KHOP_BALL))
    
    train_loader = create_jepa_dataloader(
        dataset=dataset,
        masker=masker,
        batch_size=args.batch_size,
    )
    
    config = GraphJEPAConfig(
        encoder_type=args.encoder_type,
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=args.hidden_dim,
    )
    model = config.build_model()
    
    training_config = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
    
    trainer = JEPATrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
    )
    
    results = trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
