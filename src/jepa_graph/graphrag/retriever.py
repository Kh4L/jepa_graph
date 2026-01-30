"""
Graph retrieval components for GraphRAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    subgraph: Data
    node_ids: Tensor
    scores: Optional[Tensor] = None
    paths: Optional[List[List[int]]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseRetriever(ABC):
    """Base class for graph retrievers."""
    
    @abstractmethod
    def retrieve(
        self,
        query: Union[str, Tensor],
        graph: Data,
        top_k: int = 10,
        **kwargs,
    ) -> RetrievalResult:
        pass


class GraphRetriever(BaseRetriever):
    """Graph retriever using JEPA-pretrained encoder."""
    
    def __init__(
        self,
        graph_encoder: nn.Module,
        text_encoder: Optional[nn.Module] = None,
        retrieval_strategy: str = "semantic",
        k_hops: int = 2,
        max_nodes: int = 64,
    ):
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.retrieval_strategy = retrieval_strategy
        self.k_hops = k_hops
        self.max_nodes = max_nodes
        
        self._node_embedding_cache: Optional[Tensor] = None
        self._cached_graph_id: Optional[int] = None
    
    def retrieve(
        self,
        query: Union[str, Tensor],
        graph: Data,
        top_k: int = 10,
        **kwargs,
    ) -> RetrievalResult:
        if isinstance(query, str):
            if self.text_encoder is None:
                raise ValueError("Text encoder required for string queries")
            query_embedding = self._encode_text(query)
        else:
            query_embedding = query
        
        node_embeddings = self._get_node_embeddings(graph)
        scores = self._compute_relevance(query_embedding, node_embeddings)
        
        if self.retrieval_strategy == "semantic":
            return self._retrieve_semantic(graph, scores, top_k)
        elif self.retrieval_strategy == "khop":
            return self._retrieve_khop(graph, scores, top_k)
        elif self.retrieval_strategy == "path":
            return self._retrieve_path(graph, scores, top_k)
        else:
            raise ValueError(f"Unknown strategy: {self.retrieval_strategy}")
    
    @torch.no_grad()
    def _get_node_embeddings(self, graph: Data) -> Tensor:
        graph_id = id(graph)
        
        if self._cached_graph_id == graph_id and self._node_embedding_cache is not None:
            return self._node_embedding_cache
        
        self.graph_encoder.eval()
        node_emb, _ = self.graph_encoder(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=getattr(graph, "edge_attr", None),
            batch=getattr(graph, "batch", None),
            return_node_embeddings=True,
        )
        
        self._node_embedding_cache = node_emb
        self._cached_graph_id = graph_id
        
        return node_emb
    
    def _encode_text(self, text: str) -> Tensor:
        return self.text_encoder(text)
    
    def _compute_relevance(
        self,
        query_embedding: Tensor,
        node_embeddings: Tensor,
    ) -> Tensor:
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=-1)
        node_norm = F.normalize(node_embeddings, dim=-1)
        scores = (query_norm @ node_norm.T).squeeze(0)
        return scores
    
    def _retrieve_semantic(
        self,
        graph: Data,
        scores: Tensor,
        top_k: int,
    ) -> RetrievalResult:
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        
        all_nodes = set(top_indices.tolist())
        
        edge_index = graph.edge_index
        for node in top_indices.tolist():
            neighbors = edge_index[1][edge_index[0] == node].tolist()
            all_nodes.update(neighbors)
        
        if len(all_nodes) > self.max_nodes:
            node_list = sorted(all_nodes, key=lambda n: scores[n].item(), reverse=True)
            all_nodes = set(node_list[:self.max_nodes])
        
        node_ids = torch.tensor(list(all_nodes), dtype=torch.long)
        
        sub_edge_index, _ = subgraph(
            node_ids, edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
        )
        
        retrieved_graph = Data(
            x=graph.x[node_ids],
            edge_index=sub_edge_index,
        )
        
        return RetrievalResult(
            subgraph=retrieved_graph,
            node_ids=node_ids,
            scores=scores[node_ids],
        )
    
    def _retrieve_khop(
        self,
        graph: Data,
        scores: Tensor,
        top_k: int,
    ) -> RetrievalResult:
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        
        node_ids, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=top_indices,
            num_hops=self.k_hops,
            edge_index=graph.edge_index,
            relabel_nodes=True,
            num_nodes=graph.num_nodes,
        )
        
        if len(node_ids) > self.max_nodes:
            keep_idx = torch.argsort(scores[node_ids], descending=True)[:self.max_nodes]
            node_ids = node_ids[keep_idx]
            sub_edge_index, _ = subgraph(
                node_ids, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
            )
        
        retrieved_graph = Data(
            x=graph.x[node_ids],
            edge_index=sub_edge_index,
        )
        
        return RetrievalResult(
            subgraph=retrieved_graph,
            node_ids=node_ids,
            scores=scores[node_ids],
        )
    
    def _retrieve_path(
        self,
        graph: Data,
        scores: Tensor,
        top_k: int,
    ) -> RetrievalResult:
        import networkx as nx
        
        top_scores, top_indices = torch.topk(scores, min(top_k * 2, len(scores)))
        
        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))
        edges = list(zip(
            graph.edge_index[0].tolist(),
            graph.edge_index[1].tolist()
        ))
        G.add_edges_from(edges)
        
        all_path_nodes = set(top_indices.tolist())
        paths = []
        
        top_list = top_indices.tolist()
        for i in range(min(len(top_list), top_k)):
            for j in range(i + 1, min(len(top_list), top_k)):
                try:
                    path = nx.shortest_path(G, top_list[i], top_list[j])
                    paths.append(path)
                    all_path_nodes.update(path)
                except nx.NetworkXNoPath:
                    continue
        
        if len(all_path_nodes) > self.max_nodes:
            node_list = sorted(all_path_nodes, key=lambda n: scores[n].item(), reverse=True)
            all_path_nodes = set(node_list[:self.max_nodes])
        
        node_ids = torch.tensor(list(all_path_nodes), dtype=torch.long)
        
        sub_edge_index, _ = subgraph(
            node_ids, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
        )
        
        retrieved_graph = Data(
            x=graph.x[node_ids],
            edge_index=sub_edge_index,
        )
        
        return RetrievalResult(
            subgraph=retrieved_graph,
            node_ids=node_ids,
            scores=scores[node_ids],
            paths=paths,
        )


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining multiple retrieval strategies."""
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        fusion_method: str = "reciprocal_rank",
        top_k: int = 10,
    ):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.fusion_method = fusion_method
        self.top_k = top_k
    
    def retrieve(
        self,
        query: Union[str, Tensor],
        graph: Data,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrievalResult:
        top_k = top_k or self.top_k
        
        all_results = []
        for retriever in self.retrievers:
            result = retriever.retrieve(query, graph, top_k=top_k * 2, **kwargs)
            all_results.append(result)
        
        if self.fusion_method == "reciprocal_rank":
            return self._reciprocal_rank_fusion(all_results, graph, top_k)
        elif self.fusion_method == "weighted_sum":
            return self._weighted_sum_fusion(all_results, graph, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _reciprocal_rank_fusion(
        self,
        results: List[RetrievalResult],
        graph: Data,
        top_k: int,
        k: int = 60,
    ) -> RetrievalResult:
        node_scores: Dict[int, float] = {}
        
        for result, weight in zip(results, self.weights):
            node_ids = result.node_ids.tolist()
            
            if result.scores is not None:
                sorted_indices = torch.argsort(result.scores, descending=True)
                node_ids = [node_ids[i] for i in sorted_indices.tolist()]
            
            for rank, node_id in enumerate(node_ids):
                rrf_score = weight / (k + rank + 1)
                node_scores[node_id] = node_scores.get(node_id, 0) + rrf_score
        
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, s in sorted_nodes[:top_k]]
        top_scores_list = [s for n, s in sorted_nodes[:top_k]]
        
        node_ids = torch.tensor(top_nodes, dtype=torch.long)
        scores = torch.tensor(top_scores_list, dtype=torch.float)
        
        sub_edge_index, _ = subgraph(
            node_ids, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
        )
        
        retrieved_graph = Data(
            x=graph.x[node_ids],
            edge_index=sub_edge_index,
        )
        
        return RetrievalResult(
            subgraph=retrieved_graph,
            node_ids=node_ids,
            scores=scores,
        )
    
    def _weighted_sum_fusion(
        self,
        results: List[RetrievalResult],
        graph: Data,
        top_k: int,
    ) -> RetrievalResult:
        node_scores: Dict[int, float] = {}
        
        for result, weight in zip(results, self.weights):
            if result.scores is None:
                continue
            
            scores = result.scores
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            for node_id, score in zip(result.node_ids.tolist(), scores.tolist()):
                node_scores[node_id] = node_scores.get(node_id, 0) + weight * score
        
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, s in sorted_nodes[:top_k]]
        top_scores_list = [s for n, s in sorted_nodes[:top_k]]
        
        node_ids = torch.tensor(top_nodes, dtype=torch.long)
        scores = torch.tensor(top_scores_list, dtype=torch.float)
        
        sub_edge_index, _ = subgraph(
            node_ids, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
        )
        
        retrieved_graph = Data(
            x=graph.x[node_ids],
            edge_index=sub_edge_index,
        )
        
        return RetrievalResult(
            subgraph=retrieved_graph,
            node_ids=node_ids,
            scores=scores,
        )
