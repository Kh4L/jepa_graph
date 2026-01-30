"""
JEPA Predictor module.

The predictor maps context embeddings to predicted target embeddings,
optionally incorporating structural/positional tokens.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat


class JEPAPredictor(nn.Module):
    """
    Predictor network for Graph JEPA.
    
    Takes context embeddings and predicts target embeddings using:
    - Multi-head attention to aggregate context information
    - Structural tokens to encode context-target relationships
    - MLP layers for final prediction
    """
    
    def __init__(
        self,
        embed_dim: int,
        predictor_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_structural_tokens: bool = True,
        num_structural_tokens: int = 4,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.use_structural_tokens = use_structural_tokens
        self.num_structural_tokens = num_structural_tokens
        
        self.input_proj = nn.Linear(embed_dim, predictor_dim)
        
        if use_structural_tokens:
            self.structural_tokens = nn.Parameter(
                torch.randn(num_structural_tokens, predictor_dim) * 0.02
            )
        
        self.target_query = nn.Parameter(torch.randn(1, predictor_dim) * 0.02)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=predictor_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(predictor_dim),
            nn.Linear(predictor_dim, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, embed_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        context_embeddings: Tensor,
        num_target_nodes: int,
        context_positions: Optional[Tensor] = None,
        target_positions: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if context_embeddings.dim() == 2:
            context_embeddings = context_embeddings.unsqueeze(0)
        
        batch_size, num_context, _ = context_embeddings.shape
        device = context_embeddings.device
        
        context_h = self.input_proj(context_embeddings)
        
        if self.use_structural_tokens:
            struct_tokens = repeat(
                self.structural_tokens, 
                "n d -> b n d", 
                b=batch_size
            )
            context_h = torch.cat([struct_tokens, context_h], dim=1)
            
            if context_mask is not None:
                struct_mask = torch.ones(
                    batch_size, self.num_structural_tokens, 
                    dtype=torch.bool, device=device
                )
                context_mask = torch.cat([struct_mask, context_mask], dim=1)
        
        target_queries = repeat(
            self.target_query, 
            "1 d -> b n d", 
            b=batch_size, 
            n=num_target_nodes
        )
        
        if target_positions is not None:
            target_queries = target_queries + target_positions
        
        attn_mask = ~context_mask if context_mask is not None else None
        
        target_h, _ = self.cross_attention(
            query=target_queries,
            key=context_h,
            value=context_h,
            key_padding_mask=attn_mask,
        )
        
        target_h = self.transformer(target_h)
        predicted_embeddings = self.output_proj(target_h)
        
        return predicted_embeddings


class NodeLevelPredictor(nn.Module):
    """Node-level predictor that predicts individual target node embeddings."""
    
    def __init__(
        self,
        embed_dim: int,
        predictor_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.distance_embedding = nn.Embedding(32, predictor_dim)
        
        layers = []
        in_dim = embed_dim + predictor_dim
        
        for i in range(num_layers):
            out_dim = predictor_dim if i < num_layers - 1 else embed_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        context_embedding: Tensor,
        target_distances: Tensor,
    ) -> Tensor:
        target_distances = target_distances.clamp(0, 31)
        dist_emb = self.distance_embedding(target_distances)
        
        if context_embedding.dim() == 2:
            context_embedding = context_embedding.unsqueeze(1).expand(-1, dist_emb.size(1), -1)
        
        h = torch.cat([context_embedding, dist_emb], dim=-1)
        predicted = self.mlp(h)
        
        return predicted
