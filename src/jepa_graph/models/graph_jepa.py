"""
Graph JEPA: Joint Embedding Predictive Architecture for Graphs.

Main model class that combines:
- Context encoder (online GNN/Transformer)
- Target encoder (EMA copy)
- Predictor (context -> target embedding prediction)
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from jepa_graph.models.encoders import GraphEncoder, GraphTransformerEncoder
from jepa_graph.models.predictor import JEPAPredictor


class GraphJEPA(nn.Module):
    """
    Graph JEPA model for self-supervised graph representation learning.
    
    Architecture:
    1. Context encoder: Processes visible (unmasked) part of graph
    2. Target encoder: EMA copy, processes masked target region (stop-grad)
    3. Predictor: Maps context embeddings to predicted target embeddings
    """
    
    def __init__(
        self,
        encoder: Union[GraphEncoder, GraphTransformerEncoder],
        predictor: JEPAPredictor,
        ema_decay: float = 0.996,
        ema_decay_end: float = 1.0,
        ema_anneal_steps: int = 10000,
    ):
        super().__init__()
        
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        self.predictor = predictor
        
        self.ema_decay = ema_decay
        self.ema_decay_end = ema_decay_end
        self.ema_anneal_steps = ema_anneal_steps
        self.register_buffer("ema_step", torch.tensor(0, dtype=torch.long))
    
    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder with EMA of context encoder weights."""
        step = self.ema_step.item()
        if step < self.ema_anneal_steps:
            decay = self.ema_decay + (self.ema_decay_end - self.ema_decay) * (
                step / self.ema_anneal_steps
            )
        else:
            decay = self.ema_decay_end
        
        for online_params, target_params in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_params.data.mul_(decay).add_(online_params.data, alpha=1 - decay)
        
        self.ema_step += 1
    
    def forward(
        self,
        context_data: Data,
        target_data: Data,
        context_target_mapping: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        context_node_emb, context_graph_emb = self.context_encoder(
            x=context_data.x,
            edge_index=context_data.edge_index,
            edge_attr=getattr(context_data, "edge_attr", None),
            batch=getattr(context_data, "batch", None),
            return_node_embeddings=True,
        )
        
        with torch.no_grad():
            target_node_emb, target_graph_emb = self.target_encoder(
                x=target_data.x,
                edge_index=target_data.edge_index,
                edge_attr=getattr(target_data, "edge_attr", None),
                batch=getattr(target_data, "batch", None),
                return_node_embeddings=True,
            )
            target_node_emb = target_node_emb.detach()
        
        num_target_nodes = target_node_emb.size(0)
        
        if hasattr(context_data, "batch"):
            predicted_embeddings = self.predictor(
                context_embeddings=context_node_emb.unsqueeze(0),
                num_target_nodes=num_target_nodes,
            )
            predicted_embeddings = predicted_embeddings.squeeze(0)
        else:
            predicted_embeddings = self.predictor(
                context_embeddings=context_node_emb,
                num_target_nodes=num_target_nodes,
            )
            if predicted_embeddings.dim() == 3:
                predicted_embeddings = predicted_embeddings.squeeze(0)
        
        loss = self.compute_loss(predicted_embeddings, target_node_emb)
        
        return predicted_embeddings, target_node_emb, loss
    
    def compute_loss(
        self,
        predicted: Tensor,
        target: Tensor,
        loss_type: str = "mse",
    ) -> Tensor:
        if loss_type == "mse":
            predicted_norm = F.normalize(predicted, dim=-1)
            target_norm = F.normalize(target, dim=-1)
            loss = F.mse_loss(predicted_norm, target_norm)
        elif loss_type == "cosine":
            loss = 1 - F.cosine_similarity(predicted, target, dim=-1).mean()
        elif loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(predicted, target)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
    
    @torch.no_grad()
    def encode(
        self,
        data: Data,
        use_target_encoder: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        encoder = self.target_encoder if use_target_encoder else self.context_encoder
        
        return encoder(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=getattr(data, "edge_attr", None),
            batch=getattr(data, "batch", None),
            return_node_embeddings=True,
        )
    
    def get_encoder_for_downstream(
        self,
        use_target: bool = True,
    ) -> Union[GraphEncoder, GraphTransformerEncoder]:
        if use_target:
            return copy.deepcopy(self.target_encoder)
        return copy.deepcopy(self.context_encoder)


class GraphJEPAConfig:
    """Configuration for Graph JEPA model."""
    
    def __init__(
        self,
        encoder_type: str = "gat",
        in_channels: int = 128,
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        predictor_dim: int = 256,
        num_predictor_layers: int = 2,
        use_structural_tokens: bool = True,
        num_structural_tokens: int = 4,
        ema_decay: float = 0.996,
        ema_decay_end: float = 1.0,
        ema_anneal_steps: int = 10000,
        loss_type: str = "mse",
    ):
        self.encoder_type = encoder_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.predictor_dim = predictor_dim
        self.num_predictor_layers = num_predictor_layers
        self.use_structural_tokens = use_structural_tokens
        self.num_structural_tokens = num_structural_tokens
        
        self.ema_decay = ema_decay
        self.ema_decay_end = ema_decay_end
        self.ema_anneal_steps = ema_anneal_steps
        
        self.loss_type = loss_type
    
    def build_model(self) -> GraphJEPA:
        if self.encoder_type == "transformer":
            encoder = GraphTransformerEncoder(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_encoder_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
        else:
            encoder = GraphEncoder(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.num_encoder_layers,
                gnn_type=self.encoder_type,
                heads=self.num_heads,
                dropout=self.dropout,
            )
        
        predictor = JEPAPredictor(
            embed_dim=self.out_channels,
            predictor_dim=self.predictor_dim,
            num_heads=self.num_heads,
            num_layers=self.num_predictor_layers,
            dropout=self.dropout,
            use_structural_tokens=self.use_structural_tokens,
            num_structural_tokens=self.num_structural_tokens,
        )
        
        return GraphJEPA(
            encoder=encoder,
            predictor=predictor,
            ema_decay=self.ema_decay,
            ema_decay_end=self.ema_decay_end,
            ema_anneal_steps=self.ema_anneal_steps,
        )
