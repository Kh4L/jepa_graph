"""
Graph masking strategies for JEPA pretraining.

Provides various strategies to split a graph into context and target regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple
import random

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
import networkx as nx


class MaskingStrategy(Enum):
    """Available masking strategies."""
    KHOP_BALL = auto()
    EDGE_PATH = auto()
    RANDOM_NODES = auto()
    RANDOM_SUBGRAPH = auto()
    METIS_PARTITION = auto()


@dataclass
class MaskedGraphPair:
    """Container for context and target graph regions."""
    context_data: Data
    target_data: Data
    context_node_ids: Tensor
    target_node_ids: Tensor
    original_data: Data
    context_to_target_distances: Optional[Tensor] = None
    target_positions_in_original: Optional[Tensor] = None


class GraphMasker:
    """Graph masking for JEPA pretraining."""
    
    def __init__(
        self,
        strategy: MaskingStrategy = MaskingStrategy.KHOP_BALL,
        mask_ratio: float = 0.15,
        k_hops: int = 2,
        min_target_nodes: int = 1,
        max_target_nodes: Optional[int] = None,
        include_boundary_in_context: bool = True,
    ):
        self.strategy = strategy
        self.mask_ratio = mask_ratio
        self.k_hops = k_hops
        self.min_target_nodes = min_target_nodes
        self.max_target_nodes = max_target_nodes
        self.include_boundary_in_context = include_boundary_in_context
    
    def mask(self, data: Data) -> MaskedGraphPair:
        if self.strategy == MaskingStrategy.KHOP_BALL:
            return self._mask_khop_ball(data)
        elif self.strategy == MaskingStrategy.EDGE_PATH:
            return self._mask_edge_path(data)
        elif self.strategy == MaskingStrategy.RANDOM_NODES:
            return self._mask_random_nodes(data)
        elif self.strategy == MaskingStrategy.RANDOM_SUBGRAPH:
            return self._mask_random_subgraph(data)
        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")
    
    def _mask_khop_ball(self, data: Data) -> MaskedGraphPair:
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        
        num_anchors = max(1, int(num_nodes * self.mask_ratio / (2 * self.k_hops + 1)))
        anchor_nodes = torch.randperm(num_nodes)[:num_anchors]
        
        target_node_ids, target_edge_index, _, edge_mask = k_hop_subgraph(
            node_idx=anchor_nodes,
            num_hops=self.k_hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        
        if len(target_node_ids) < self.min_target_nodes:
            target_node_ids, target_edge_index, _, edge_mask = k_hop_subgraph(
                node_idx=anchor_nodes,
                num_hops=self.k_hops + 1,
                edge_index=edge_index,
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
        
        if self.max_target_nodes and len(target_node_ids) > self.max_target_nodes:
            keep_idx = torch.randperm(len(target_node_ids))[:self.max_target_nodes]
            target_node_ids = target_node_ids[keep_idx]
            target_edge_index, _ = subgraph(
                target_node_ids, edge_index, relabel_nodes=True, num_nodes=num_nodes
            )
        
        all_nodes = torch.arange(num_nodes, device=edge_index.device)
        target_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        target_mask[target_node_ids] = True
        context_node_ids = all_nodes[~target_mask]
        
        context_edge_index, _ = subgraph(
            context_node_ids, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        
        context_data = Data(
            x=data.x[context_node_ids],
            edge_index=context_edge_index,
        )
        
        target_data = Data(
            x=data.x[target_node_ids],
            edge_index=target_edge_index,
        )
        
        distances = self._compute_target_context_distances(
            data, target_node_ids, context_node_ids
        )
        
        return MaskedGraphPair(
            context_data=context_data,
            target_data=target_data,
            context_node_ids=context_node_ids,
            target_node_ids=target_node_ids,
            original_data=data,
            context_to_target_distances=distances,
        )
    
    def _mask_edge_path(self, data: Data) -> MaskedGraphPair:
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        
        G = self._to_networkx(data)
        num_pairs = max(1, int(num_nodes * self.mask_ratio / 3))
        nodes = list(range(num_nodes))
        
        target_nodes = set()
        
        for _ in range(num_pairs):
            if len(nodes) < 2:
                break
            src, tgt = random.sample(nodes, 2)
            
            try:
                path = nx.shortest_path(G, src, tgt)
                target_nodes.update(path)
            except nx.NetworkXNoPath:
                continue
        
        if len(target_nodes) < self.min_target_nodes:
            return self._mask_random_nodes(data)
        
        target_node_ids = torch.tensor(list(target_nodes), dtype=torch.long)
        target_edge_index, _ = subgraph(
            target_node_ids, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        
        context_mask = torch.ones(num_nodes, dtype=torch.bool)
        context_mask[target_node_ids] = False
        context_node_ids = torch.arange(num_nodes)[context_mask]
        
        context_edge_index, _ = subgraph(
            context_node_ids, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        
        context_data = Data(x=data.x[context_node_ids], edge_index=context_edge_index)
        target_data = Data(x=data.x[target_node_ids], edge_index=target_edge_index)
        
        return MaskedGraphPair(
            context_data=context_data,
            target_data=target_data,
            context_node_ids=context_node_ids,
            target_node_ids=target_node_ids,
            original_data=data,
        )
    
    def _mask_random_nodes(self, data: Data) -> MaskedGraphPair:
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        
        num_target = max(self.min_target_nodes, int(num_nodes * self.mask_ratio))
        if self.max_target_nodes:
            num_target = min(num_target, self.max_target_nodes)
        
        perm = torch.randperm(num_nodes)
        target_node_ids = perm[:num_target]
        context_node_ids = perm[num_target:]
        
        target_edge_index, _ = subgraph(
            target_node_ids, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        context_edge_index, _ = subgraph(
            context_node_ids, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        
        context_data = Data(x=data.x[context_node_ids], edge_index=context_edge_index)
        target_data = Data(x=data.x[target_node_ids], edge_index=target_edge_index)
        
        return MaskedGraphPair(
            context_data=context_data,
            target_data=target_data,
            context_node_ids=context_node_ids,
            target_node_ids=target_node_ids,
            original_data=data,
        )
    
    def _mask_random_subgraph(self, data: Data) -> MaskedGraphPair:
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        
        start_node = random.randint(0, num_nodes - 1)
        adj = self._build_adjacency_list(edge_index, num_nodes)
        
        num_target = max(self.min_target_nodes, int(num_nodes * self.mask_ratio))
        if self.max_target_nodes:
            num_target = min(num_target, self.max_target_nodes)
        
        visited = {start_node}
        frontier = [start_node]
        
        while len(visited) < num_target and frontier:
            current = random.choice(frontier)
            neighbors = adj.get(current, [])
            
            if neighbors:
                next_node = random.choice(neighbors)
                if next_node not in visited:
                    visited.add(next_node)
                    frontier.append(next_node)
            else:
                frontier.remove(current)
        
        target_node_ids = torch.tensor(list(visited), dtype=torch.long)
        
        context_mask = torch.ones(num_nodes, dtype=torch.bool)
        context_mask[target_node_ids] = False
        context_node_ids = torch.arange(num_nodes)[context_mask]
        
        target_edge_index, _ = subgraph(
            target_node_ids, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        context_edge_index, _ = subgraph(
            context_node_ids, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        
        context_data = Data(x=data.x[context_node_ids], edge_index=context_edge_index)
        target_data = Data(x=data.x[target_node_ids], edge_index=target_edge_index)
        
        return MaskedGraphPair(
            context_data=context_data,
            target_data=target_data,
            context_node_ids=context_node_ids,
            target_node_ids=target_node_ids,
            original_data=data,
        )
    
    def _compute_target_context_distances(
        self,
        data: Data,
        target_node_ids: Tensor,
        context_node_ids: Tensor,
    ) -> Tensor:
        G = self._to_networkx(data)
        
        distances = []
        context_set = set(context_node_ids.tolist())
        
        for target_node in target_node_ids.tolist():
            min_dist = float("inf")
            for ctx_node in context_set:
                try:
                    dist = nx.shortest_path_length(G, target_node, ctx_node)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    continue
            distances.append(min_dist if min_dist != float("inf") else -1)
        
        return torch.tensor(distances, dtype=torch.long)
    
    @staticmethod
    def _to_networkx(data: Data) -> nx.Graph:
        edge_index = data.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        return G
    
    @staticmethod
    def _build_adjacency_list(edge_index: Tensor, num_nodes: int) -> dict:
        adj = {i: [] for i in range(num_nodes)}
        src, dst = edge_index.cpu().tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)
        return adj


class MultiScaleMasker:
    """Multi-scale masking for hierarchical JEPA pretraining."""
    
    def __init__(self, scales: List[dict] = None):
        if scales is None:
            scales = [
                {"strategy": MaskingStrategy.KHOP_BALL, "k_hops": 1, "mask_ratio": 0.1},
                {"strategy": MaskingStrategy.KHOP_BALL, "k_hops": 2, "mask_ratio": 0.15},
                {"strategy": MaskingStrategy.RANDOM_SUBGRAPH, "mask_ratio": 0.25},
            ]
        
        self.maskers = [
            GraphMasker(
                strategy=s["strategy"],
                k_hops=s.get("k_hops", 2),
                mask_ratio=s.get("mask_ratio", 0.15),
            )
            for s in scales
        ]
    
    def mask(self, data: Data) -> List[MaskedGraphPair]:
        return [masker.mask(data) for masker in self.maskers]
