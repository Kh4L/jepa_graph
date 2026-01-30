"""
Dataset classes for Graph JEPA pretraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader

from jepa_graph.data.masking import GraphMasker, MaskedGraphPair, MaskingStrategy


class GraphJEPADataset(Dataset):
    """Dataset wrapper that applies JEPA masking to graphs."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        masker: Optional[GraphMasker] = None,
        transform: Optional[Callable] = None,
        num_masks_per_graph: int = 1,
    ):
        super().__init__(transform=transform)
        
        self.base_dataset = base_dataset
        self.masker = masker or GraphMasker(
            strategy=MaskingStrategy.KHOP_BALL,
            k_hops=2,
            mask_ratio=0.15,
        )
        self.num_masks_per_graph = num_masks_per_graph
    
    def len(self) -> int:
        return len(self.base_dataset) * self.num_masks_per_graph
    
    def get(self, idx: int) -> MaskedGraphPair:
        base_idx = idx // self.num_masks_per_graph
        data = self.base_dataset[base_idx]
        masked_pair = self.masker.mask(data)
        return masked_pair


class JEPACollater:
    """Custom collater for batching MaskedGraphPair objects."""
    
    def __init__(self, follow_batch: Optional[List[str]] = None):
        self.follow_batch = follow_batch or []
    
    def __call__(self, batch: List[MaskedGraphPair]) -> dict:
        from torch_geometric.data import Batch
        
        context_list = [pair.context_data for pair in batch]
        target_list = [pair.target_data for pair in batch]
        
        context_batch = Batch.from_data_list(context_list)
        target_batch = Batch.from_data_list(target_list)
        
        context_node_ids = [pair.context_node_ids for pair in batch]
        target_node_ids = [pair.target_node_ids for pair in batch]
        
        distances = None
        if batch[0].context_to_target_distances is not None:
            distances = [pair.context_to_target_distances for pair in batch]
        
        return {
            "context_data": context_batch,
            "target_data": target_batch,
            "context_node_ids": context_node_ids,
            "target_node_ids": target_node_ids,
            "distances": distances,
            "batch_size": len(batch),
        }


def create_jepa_dataloader(
    dataset: Dataset,
    masker: Optional[GraphMasker] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    num_masks_per_graph: int = 1,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for JEPA pretraining."""
    jepa_dataset = GraphJEPADataset(
        base_dataset=dataset,
        masker=masker,
        num_masks_per_graph=num_masks_per_graph,
    )
    
    collater = JEPACollater()
    
    return DataLoader(
        jepa_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collater,
        **kwargs,
    )


class SubgraphSampler:
    """Samples subgraphs from a large graph for JEPA pretraining."""
    
    def __init__(
        self,
        num_hops: int = 3,
        max_nodes: int = 256,
        min_nodes: int = 32,
        sample_strategy: str = "random_walk",
    ):
        self.num_hops = num_hops
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.sample_strategy = sample_strategy
    
    def sample(self, data: Data, anchor_node: Optional[int] = None) -> Data:
        from torch_geometric.utils import k_hop_subgraph, subgraph
        
        if anchor_node is None:
            anchor_node = torch.randint(0, data.num_nodes, (1,)).item()
        
        if self.sample_strategy == "khop":
            node_ids, edge_index, _, _ = k_hop_subgraph(
                node_idx=anchor_node,
                num_hops=self.num_hops,
                edge_index=data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )
        elif self.sample_strategy == "random_walk":
            node_ids = self._random_walk_sample(data, anchor_node)
            edge_index, _ = subgraph(
                node_ids, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
            )
        else:
            raise ValueError(f"Unknown sample strategy: {self.sample_strategy}")
        
        if len(node_ids) > self.max_nodes:
            keep_idx = torch.randperm(len(node_ids))[:self.max_nodes]
            node_ids = node_ids[keep_idx]
            edge_index, _ = subgraph(
                node_ids, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
            )
        
        subgraph_data = Data(
            x=data.x[node_ids] if data.x is not None else None,
            edge_index=edge_index,
        )
        
        return subgraph_data
    
    def _random_walk_sample(self, data: Data, start_node: int) -> Tensor:
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        adj = {i: [] for i in range(num_nodes)}
        src, dst = edge_index.tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)
        
        visited = {start_node}
        frontier = [start_node]
        
        import random
        while len(visited) < self.max_nodes and frontier:
            current = random.choice(frontier)
            neighbors = adj.get(current, [])
            
            if neighbors:
                next_node = random.choice(neighbors)
                if next_node not in visited:
                    visited.add(next_node)
                    frontier.append(next_node)
            
            if len(frontier) > 50:
                frontier = random.sample(frontier, 25)
        
        return torch.tensor(list(visited), dtype=torch.long)
