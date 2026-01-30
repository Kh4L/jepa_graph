"""
Graph Encoders for JEPA pretraining.

Provides both GNN-based and Graph Transformer-based encoders that can be used
as the context encoder (online) and target encoder (EMA) in the JEPA framework.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINEConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch


class GraphEncoder(nn.Module):
    """
    GNN-based graph encoder using message passing layers.
    
    Supports multiple GNN architectures: GCN, GAT, GIN.
    Outputs node embeddings and optionally a graph-level embedding.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        gnn_type: str = "gat",
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        pooling: str = "mean",
    ):
        """
        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer ('gcn', 'gat', 'gin')
            heads: Number of attention heads (for GAT)
            dropout: Dropout probability
            edge_dim: Edge feature dimension (optional)
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            pooling: Graph-level pooling type ('mean', 'add', 'none')
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.use_residual = use_residual
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        
        for i in range(num_layers):
            in_dim = hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            
            if gnn_type == "gcn":
                conv = GCNConv(in_dim, out_dim)
            elif gnn_type == "gat":
                conv = GATv2Conv(
                    in_dim,
                    out_dim // heads if i < num_layers - 1 else out_dim // heads,
                    heads=heads,
                    concat=True if i < num_layers - 1 else False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
                if i < num_layers - 1:
                    out_dim = (out_dim // heads) * heads
            elif gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
                conv = GINEConv(mlp, edge_dim=edge_dim) if edge_dim else GINEConv(mlp)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.convs.append(conv)
            
            if use_layer_norm and i < num_layers - 1:
                self.norms.append(nn.LayerNorm(out_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        if use_residual and hidden_channels != out_channels:
            self.output_proj = nn.Linear(hidden_channels, out_channels)
        else:
            self.output_proj = None
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        return_node_embeddings: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through the graph encoder.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            batch: Batch assignment vector [num_nodes] (optional)
            return_node_embeddings: Whether to return node-level embeddings
            
        Returns:
            node_embeddings: Node embeddings [num_nodes, out_channels]
            graph_embedding: Graph embedding [batch_size, out_channels]
        """
        h = self.input_proj(x)
        
        for i, conv in enumerate(self.convs):
            h_prev = h
            
            if self.gnn_type in ["gat", "gin"] and edge_attr is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            
            if i < self.num_layers - 1:
                if self.norms is not None:
                    h = self.norms[i](h)
                h = F.gelu(h)
                h = self.dropout(h)
                
                if self.use_residual:
                    h = h + h_prev
        
        node_embeddings = h if return_node_embeddings else None
        
        if self.pooling == "mean":
            graph_embedding = global_mean_pool(h, batch)
        elif self.pooling == "add":
            graph_embedding = global_add_pool(h, batch)
        else:
            graph_embedding = None
        
        return node_embeddings, graph_embedding
    
    @torch.no_grad()
    def get_ema_copy(self) -> "GraphEncoder":
        """Create a copy of this encoder for EMA target encoder."""
        return copy.deepcopy(self)


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer encoder using self-attention over graph structure.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        edge_dim: Optional[int] = None,
        max_nodes: int = 512,
        use_structural_encoding: bool = True,
        structural_encoding_dim: int = 16,
        pooling: str = "mean",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.pooling = pooling
        self.use_structural_encoding = use_structural_encoding
        
        input_dim = in_channels
        if use_structural_encoding:
            input_dim += structural_encoding_dim
        self.input_proj = nn.Linear(input_dim, hidden_channels)
        
        self.pos_embedding = nn.Embedding(max_nodes, hidden_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if use_edge_features and edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, num_heads)
        else:
            self.edge_proj = None
        
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_channels))
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        structural_encoding: Optional[Tensor] = None,
        return_node_embeddings: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.use_structural_encoding and structural_encoding is not None:
            x = torch.cat([x, structural_encoding], dim=-1)
        
        h = self.input_proj(x)
        h_dense, mask = to_dense_batch(h, batch)
        batch_size, max_nodes, _ = h_dense.shape
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        h_dense = torch.cat([cls_tokens, h_dense], dim=1)
        
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)
        
        positions = torch.arange(h_dense.size(1), device=h_dense.device)
        h_dense = h_dense + self.pos_embedding(positions).unsqueeze(0)
        
        attn_mask = ~mask
        h_dense = self.transformer(h_dense, src_key_padding_mask=attn_mask)
        h_dense = self.output_proj(h_dense)
        
        graph_embedding = h_dense[:, 0]
        
        if return_node_embeddings:
            node_dense = h_dense[:, 1:]
            node_embeddings = node_dense[mask[:, 1:]]
        else:
            node_embeddings = None
        
        return node_embeddings, graph_embedding
    
    @torch.no_grad()
    def get_ema_copy(self) -> "GraphTransformerEncoder":
        return copy.deepcopy(self)


class HeteroGraphEncoder(nn.Module):
    """Encoder for heterogeneous graphs (knowledge graphs)."""
    
    def __init__(
        self,
        node_types: list[str],
        edge_types: list[Tuple[str, str, str]],
        in_channels_dict: dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.out_channels = out_channels
        
        self.input_projs = nn.ModuleDict({
            ntype: nn.Linear(in_channels_dict[ntype], hidden_channels)
            for ntype in node_types
        })
        
        from torch_geometric.nn import HeteroConv, SAGEConv
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(in_dim, out_dim)
            
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            
            if i < num_layers - 1:
                self.norms.append(nn.ModuleDict({
                    ntype: nn.LayerNorm(out_dim) for ntype in node_types
                }))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[Tuple[str, str, str], Tensor],
        return_node_embeddings: bool = True,
    ) -> dict[str, Tensor]:
        h_dict = {
            ntype: self.input_projs[ntype](x)
            for ntype, x in x_dict.items()
        }
        
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)
            
            if i < len(self.convs) - 1:
                h_dict = {
                    ntype: F.gelu(self.norms[i][ntype](h))
                    for ntype, h in h_dict.items()
                }
                h_dict = {
                    ntype: self.dropout(h)
                    for ntype, h in h_dict.items()
                }
        
        return h_dict
