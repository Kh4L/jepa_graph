"""
Structural and positional encodings for graphs.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, degree
import numpy as np


def compute_random_walk_pe(
    edge_index: Tensor,
    num_nodes: int,
    walk_length: int = 16,
    add_self_loops: bool = True,
) -> Tensor:
    """Compute Random Walk Positional Encoding."""
    from torch_geometric.utils import add_self_loops as add_loops, to_dense_adj
    
    device = edge_index.device
    
    if add_self_loops:
        edge_index, _ = add_loops(edge_index, num_nodes=num_nodes)
    
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float)
    deg_inv = 1.0 / deg.clamp(min=1)
    
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    adj = adj * deg_inv.view(-1, 1)
    
    pe = torch.zeros(num_nodes, walk_length, device=device)
    
    P_power = torch.eye(num_nodes, device=device)
    for k in range(walk_length):
        P_power = P_power @ adj
        pe[:, k] = P_power.diag()
    
    return pe


def compute_laplacian_pe(
    edge_index: Tensor,
    num_nodes: int,
    pe_dim: int = 16,
    normalization: str = "sym",
) -> Tensor:
    """Compute Laplacian Positional Encoding."""
    from scipy.sparse.linalg import eigsh
    
    edge_index_lap, edge_weight = get_laplacian(
        edge_index, 
        normalization=normalization if normalization != "none" else None,
        num_nodes=num_nodes,
    )
    
    L = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes=num_nodes)
    L = L.tocsc()
    
    try:
        k = min(pe_dim + 1, num_nodes - 1)
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM", return_eigenvectors=True)
        
        idx = eigenvalues.argsort()
        eigenvectors = eigenvectors[:, idx]
        pe = eigenvectors[:, 1:pe_dim+1]
        pe = np.real(pe)
        
        if pe.shape[1] < pe_dim:
            padding = np.zeros((num_nodes, pe_dim - pe.shape[1]))
            pe = np.concatenate([pe, padding], axis=1)
        
        return torch.from_numpy(pe).float()
        
    except Exception:
        return torch.zeros(num_nodes, pe_dim)


def compute_degree_encoding(
    edge_index: Tensor,
    num_nodes: int,
    max_degree: int = 64,
) -> Tensor:
    """Compute degree-based encoding."""
    row = edge_index[0]
    deg = degree(row, num_nodes, dtype=torch.long)
    deg = deg.clamp(max=max_degree - 1)
    deg_one_hot = F.one_hot(deg, num_classes=max_degree).float()
    return deg_one_hot


def add_structural_encoding(
    data: Data,
    encoding_type: str = "random_walk",
    encoding_dim: int = 16,
    **kwargs,
) -> Data:
    """Add structural encoding to a PyG Data object."""
    if encoding_type == "random_walk":
        pe = compute_random_walk_pe(
            data.edge_index,
            data.num_nodes,
            walk_length=encoding_dim,
            **kwargs,
        )
    elif encoding_type == "laplacian":
        pe = compute_laplacian_pe(
            data.edge_index,
            data.num_nodes,
            pe_dim=encoding_dim,
            **kwargs,
        )
    elif encoding_type == "degree":
        pe = compute_degree_encoding(
            data.edge_index,
            data.num_nodes,
            max_degree=encoding_dim,
        )
    elif encoding_type == "combined":
        rwpe = compute_random_walk_pe(
            data.edge_index, data.num_nodes, walk_length=encoding_dim // 2
        )
        lappe = compute_laplacian_pe(
            data.edge_index, data.num_nodes, pe_dim=encoding_dim // 2
        )
        pe = torch.cat([rwpe, lappe], dim=-1)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    data.pe = pe.to(data.x.device if data.x is not None else pe.device)
    
    return data


class StructuralEncodingTransform:
    """PyG Transform that adds structural encodings to graphs."""
    
    def __init__(
        self,
        encoding_type: str = "random_walk",
        encoding_dim: int = 16,
        **kwargs,
    ):
        self.encoding_type = encoding_type
        self.encoding_dim = encoding_dim
        self.kwargs = kwargs
    
    def __call__(self, data: Data) -> Data:
        return add_structural_encoding(
            data,
            encoding_type=self.encoding_type,
            encoding_dim=self.encoding_dim,
            **self.kwargs,
        )
