"""
Fusion modules for combining graph and text representations.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FusionType(Enum):
    """Available fusion strategies."""
    CONCAT = auto()
    CROSS_ATTENTION = auto()
    GATING = auto()
    BILINEAR = auto()
    HADAMARD = auto()


class GraphTextFusion(nn.Module):
    """Fusion module for combining graph and text representations."""
    
    def __init__(
        self,
        graph_dim: int,
        text_dim: int,
        output_dim: int,
        fusion_type: FusionType = FusionType.CROSS_ATTENTION,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        self.graph_proj = nn.Linear(graph_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        if fusion_type == FusionType.CONCAT:
            self.output_proj = nn.Linear(output_dim * 2, output_dim)
            
        elif fusion_type == FusionType.CROSS_ATTENTION:
            self.graph_to_text = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.text_to_graph = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.output_proj = nn.Linear(output_dim * 2, output_dim)
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            
        elif fusion_type == FusionType.GATING:
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid(),
            )
            self.output_proj = nn.Linear(output_dim, output_dim)
            
        elif fusion_type == FusionType.BILINEAR:
            self.bilinear = nn.Bilinear(output_dim, output_dim, output_dim)
            
        elif fusion_type == FusionType.HADAMARD:
            self.output_proj = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        graph_repr: Tensor,
        text_repr: Tensor,
        graph_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        graph_h = self.graph_proj(graph_repr)
        text_h = self.text_proj(text_repr)
        
        is_pooled_graph = graph_h.dim() == 2
        is_pooled_text = text_h.dim() == 2
        
        if is_pooled_graph:
            graph_h = graph_h.unsqueeze(1)
        if is_pooled_text:
            text_h = text_h.unsqueeze(1)
        
        if self.fusion_type == FusionType.CONCAT:
            fused = self._concat_fusion(graph_h, text_h)
        elif self.fusion_type == FusionType.CROSS_ATTENTION:
            fused = self._cross_attention_fusion(graph_h, text_h, graph_mask, text_mask)
        elif self.fusion_type == FusionType.GATING:
            fused = self._gating_fusion(graph_h, text_h)
        elif self.fusion_type == FusionType.BILINEAR:
            fused = self._bilinear_fusion(graph_h, text_h)
        elif self.fusion_type == FusionType.HADAMARD:
            fused = self._hadamard_fusion(graph_h, text_h)
        
        if is_pooled_graph and is_pooled_text:
            fused = fused.mean(dim=1)
        
        return fused
    
    def _concat_fusion(self, graph_h: Tensor, text_h: Tensor) -> Tensor:
        graph_pooled = graph_h.mean(dim=1, keepdim=True)
        text_pooled = text_h.mean(dim=1, keepdim=True)
        concat = torch.cat([graph_pooled, text_pooled], dim=-1)
        fused = self.output_proj(concat)
        return fused
    
    def _cross_attention_fusion(
        self,
        graph_h: Tensor,
        text_h: Tensor,
        graph_mask: Optional[Tensor],
        text_mask: Optional[Tensor],
    ) -> Tensor:
        graph_attended, _ = self.graph_to_text(
            query=graph_h,
            key=text_h,
            value=text_h,
            key_padding_mask=text_mask,
        )
        graph_attended = self.norm1(graph_h + graph_attended)
        
        text_attended, _ = self.text_to_graph(
            query=text_h,
            key=graph_h,
            value=graph_h,
            key_padding_mask=graph_mask,
        )
        text_attended = self.norm2(text_h + text_attended)
        
        graph_pooled = graph_attended.mean(dim=1, keepdim=True)
        text_pooled = text_attended.mean(dim=1, keepdim=True)
        
        concat = torch.cat([graph_pooled, text_pooled], dim=-1)
        fused = self.output_proj(concat)
        
        return fused
    
    def _gating_fusion(self, graph_h: Tensor, text_h: Tensor) -> Tensor:
        graph_pooled = graph_h.mean(dim=1, keepdim=True)
        text_pooled = text_h.mean(dim=1, keepdim=True)
        
        concat = torch.cat([graph_pooled, text_pooled], dim=-1)
        gate = self.gate(concat)
        
        fused = gate * graph_pooled + (1 - gate) * text_pooled
        fused = self.output_proj(fused)
        
        return fused
    
    def _bilinear_fusion(self, graph_h: Tensor, text_h: Tensor) -> Tensor:
        graph_pooled = graph_h.mean(dim=1)
        text_pooled = text_h.mean(dim=1)
        fused = self.bilinear(graph_pooled, text_pooled)
        return fused.unsqueeze(1)
    
    def _hadamard_fusion(self, graph_h: Tensor, text_h: Tensor) -> Tensor:
        graph_pooled = graph_h.mean(dim=1, keepdim=True)
        text_pooled = text_h.mean(dim=1, keepdim=True)
        fused = graph_pooled * text_pooled
        fused = self.output_proj(fused)
        return fused


class GraphToSequence(nn.Module):
    """Convert graph representation to sequence for LLM input."""
    
    def __init__(
        self,
        graph_dim: int,
        llm_dim: int,
        num_tokens: int = 32,
        use_attention_pool: bool = True,
    ):
        super().__init__()
        
        self.graph_dim = graph_dim
        self.llm_dim = llm_dim
        self.num_tokens = num_tokens
        
        self.query_tokens = nn.Parameter(
            torch.randn(num_tokens, graph_dim) * 0.02
        )
        
        if use_attention_pool:
            self.attention = nn.MultiheadAttention(
                embed_dim=graph_dim,
                num_heads=8,
                batch_first=True,
            )
        else:
            self.attention = None
        
        self.output_proj = nn.Sequential(
            nn.Linear(graph_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
    
    def forward(
        self,
        node_embeddings: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
        
        B, N, D = node_embeddings.shape
        
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        
        if self.attention is not None:
            output, _ = self.attention(
                query=queries,
                key=node_embeddings,
                value=node_embeddings,
            )
        else:
            pooled = node_embeddings.mean(dim=1, keepdim=True)
            output = queries + pooled
        
        output = self.output_proj(output)
        
        return output
