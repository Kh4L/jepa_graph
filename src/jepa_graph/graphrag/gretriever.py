"""
G-Retriever integration for Graph JEPA.

G-Retriever (https://arxiv.org/abs/2402.07630) combines:
- GNN encoder for graph understanding
- LLM for text generation
- Subgraph retrieval for context

This module provides:
- JEPA-pretrained GNN as the graph encoder for PyG's GRetriever
- Training pipeline aligned with the baseline
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def adjust_learning_rate(
    param_group: dict,
    base_lr: float,
    epoch: float,
    num_epochs: int,
    min_lr: float = 5e-6,
    warmup_epochs: int = 1,
) -> float:
    """
    Decay learning rate with half-cycle cosine after warmup.
    
    Matches G-Retriever baseline.
    """
    if epoch < warmup_epochs:
        lr = base_lr
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) /
                          (num_epochs - warmup_epochs))
        )
    param_group['lr'] = lr
    return lr


def get_loss(
    model,
    batch,
    model_type: str = "gnn+llm",
) -> Tensor:
    """
    Compute loss for G-Retriever model.
    
    Matches baseline: https://github.com/puririshi98/gretriever-stark-prime
    
    Args:
        model: GRetriever or LLM model
        batch: Batch of data
        model_type: 'llm' or 'gnn+llm'
        
    Returns:
        Loss tensor
    """
    if model_type == 'llm':
        return model(batch.question, batch.label, batch.desc)
    else:  # GNN+LLM (GRetriever)
        return model(
            batch.question,
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.label,
            batch.edge_attr,
            batch.desc if hasattr(batch, 'desc') else "",
        )


def inference_step(
    model,
    batch,
    model_type: str = "gnn+llm",
    max_out_tokens: int = 128,
) -> List[str]:
    """
    Perform inference on a batch.
    
    Args:
        model: GRetriever or LLM model
        batch: Batch of data
        model_type: 'llm' or 'gnn+llm'
        max_out_tokens: Maximum output tokens
        
    Returns:
        List of generated answers
    """
    if model_type == 'llm':
        return model.inference(
            batch.question,
            batch.desc,
            max_out_tokens=max_out_tokens,
        )
    else:  # GNN+LLM
        return model.inference(
            batch.question,
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.edge_attr,
            batch.desc if hasattr(batch, 'desc') else "",
            max_out_tokens=max_out_tokens,
        )


def create_gretriever_with_jepa_encoder(
    jepa_encoder: nn.Module,
    llm_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    sys_prompt: str = (
        "You are an expert assistant that answers questions "
        "based on the provided knowledge graph context. "
        "Give concise answers without explanation."
    ),
) -> nn.Module:
    """
    Create PyG's GRetriever model using a JEPA-pretrained GNN encoder.
    
    This is the recommended way to use JEPA with G-Retriever - it uses
    the actual PyG GRetriever class.
    
    Args:
        jepa_encoder: JEPA-pretrained graph encoder
        llm_model_name: HuggingFace model name for LLM
        sys_prompt: System prompt for LLM
        
    Returns:
        GRetriever model with JEPA encoder as the GNN component
    """
    try:
        from torch_geometric.nn.models import GRetriever, LLM
    except ImportError:
        raise ImportError(
            "PyG GRetriever not available. Install with:\n"
            "pip install torch_geometric[llm]\n"
            "pip install transformers accelerate sentencepiece"
        )
    
    # Create LLM component
    llm = LLM(model_name=llm_model_name, sys_prompt=sys_prompt)
    
    # Create GRetriever with JEPA encoder as the GNN
    model = GRetriever(llm=llm, gnn=jepa_encoder)
    
    return model


def load_jepa_encoder(
    checkpoint_path: str,
    encoder_config: Dict[str, Any],
    device: str = "cuda",
) -> nn.Module:
    """
    Load a JEPA-pretrained encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to saved encoder weights
        encoder_config: Configuration dict for encoder architecture
        device: Device to load to
        
    Returns:
        Loaded encoder module
    """
    from jepa_graph.models.encoders import GraphEncoder
    
    encoder = GraphEncoder(
        in_channels=encoder_config['in_channels'],
        hidden_channels=encoder_config['hidden_channels'],
        out_channels=encoder_config['out_channels'],
        num_layers=encoder_config.get('num_layers', 3),
        gnn_type=encoder_config.get('gnn_type', 'gat'),
        heads=encoder_config.get('heads', 4),
        dropout=encoder_config.get('dropout', 0.1),
    )
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(state_dict)
    print(f"Loaded JEPA encoder from {checkpoint_path}")
    
    return encoder.to(device)


def create_gretriever_from_jepa(
    jepa_encoder_path: str,
    encoder_config: Dict[str, Any],
    llm_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    freeze_gnn: bool = False,
    device: str = "cuda",
) -> nn.Module:
    """
    Create G-Retriever using a saved JEPA-pretrained encoder.
    
    This loads the encoder and creates a full GRetriever model.
    
    Args:
        jepa_encoder_path: Path to saved encoder weights
        encoder_config: Config dict for encoder architecture
        llm_model_name: LLM model name
        freeze_gnn: Whether to freeze GNN weights during finetuning
        device: Device
        
    Returns:
        GRetriever model ready for finetuning
    """
    # Load JEPA encoder
    encoder = load_jepa_encoder(jepa_encoder_path, encoder_config, device)
    
    # Optionally freeze
    if freeze_gnn:
        for param in encoder.parameters():
            param.requires_grad = False
        print("GNN encoder weights frozen")
    
    # Create GRetriever
    model = create_gretriever_with_jepa_encoder(
        jepa_encoder=encoder,
        llm_model_name=llm_model_name,
    )
    
    return model.to(device)


class GRetrieverTrainer:
    """
    Trainer for G-Retriever with JEPA-pretrained encoder.
    
    Follows the exact training protocol from the baseline:
    https://github.com/puririshi98/gretriever-stark-prime
    
    - Cosine LR with warmup
    - Gradient clipping at 0.1
    - AdamW with β=(0.9, 0.95)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        lr: float = 1e-5,
        epochs: int = 3,
        grad_clip: float = 0.1,
        accumulation_steps: int = 2,
        device: str = "cuda",
    ):
        """
        Args:
            model: GRetriever model (from create_gretriever_from_jepa)
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            lr: Learning rate
            epochs: Number of training epochs
            grad_clip: Gradient clipping value
            accumulation_steps: Gradient accumulation steps
            device: Device
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.accumulation_steps = accumulation_steps
        self.device = device
        
        # Setup optimizer (matches baseline exactly)
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            [{'params': params, 'lr': lr, 'weight_decay': 0.05}],
            betas=(0.9, 0.95),
        )
        
        self.num_oom_errors = 0
    
    def train(self) -> nn.Module:
        """
        Run training loop.
        
        Returns:
            Trained model
        """
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            epoch_str = f'Epoch: {epoch + 1}|{self.epochs}'
            
            loader = tqdm(self.train_loader, desc=epoch_str)
            
            for step, batch in enumerate(loader):
                self.optimizer.zero_grad()
                
                # Clear description to save memory (matches baseline)
                if hasattr(batch, 'desc'):
                    batch.desc = ""
                
                try:
                    loss = get_loss(self.model, batch)
                    loss.backward()
                    
                    clip_grad_norm_(
                        self.optimizer.param_groups[0]['params'],
                        self.grad_clip,
                    )
                    
                    # LR adjustment every accumulation_steps
                    if (step + 1) % self.accumulation_steps == 0:
                        adjust_learning_rate(
                            self.optimizer.param_groups[0],
                            self.lr,
                            step / len(self.train_loader) + epoch,
                            self.epochs,
                        )
                    
                    self.optimizer.step()
                    epoch_loss += float(loss.detach())
                    
                    loader.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self.num_oom_errors += 1
                    print(f"OOM error at step {step}, total: {self.num_oom_errors}")
            
            # Print epoch stats
            train_loss = epoch_loss / len(self.train_loader)
            print(f"{epoch_str}, Train Loss: {train_loss:.4f}")
            
            # Sequence length stats (if available)
            if hasattr(self.model, 'seq_length_stats') and self.model.seq_length_stats:
                stats = self.model.seq_length_stats
                print(f"  Seq len - avg: {sum(stats)/len(stats):.0f}, "
                      f"min: {min(stats)}, max: {max(stats)}")
            
            # Validation
            if self.val_loader is not None:
                val_loss = self._validate()
                print(f"{epoch_str}, Val Loss: {val_loss:.4f}")
        
        if self.num_oom_errors > 0:
            print(f"Total OOM errors: {self.num_oom_errors} "
                  f"({100*self.num_oom_errors/len(self.train_loader)/self.epochs:.1f}%)")
        
        torch.cuda.empty_cache()
        self.model.eval()
        return self.model
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_loader:
            if hasattr(batch, 'desc'):
                batch.desc = ""
            loss = get_loss(self.model, batch)
            total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(self.val_loader)
    
    @torch.no_grad()
    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Run evaluation on test set.
        
        Returns:
            List of evaluation outputs with pred, question, label
        """
        if self.test_loader is None:
            raise ValueError("No test loader provided")
        
        self.model.eval()
        eval_output = []
        
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            if hasattr(batch, 'desc'):
                batch.desc = ""
            
            pred = inference_step(self.model, batch)
            
            eval_output.append({
                'pred': pred,
                'question': batch.question,
                'label': batch.label,
            })
        
        return eval_output
