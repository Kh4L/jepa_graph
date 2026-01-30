"""
End-to-end GraphRAG pipeline with JEPA-pretrained encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from jepa_graph.graphrag.retriever import GraphRetriever, RetrievalResult
from jepa_graph.graphrag.fusion import GraphTextFusion, GraphToSequence, FusionType


@dataclass
class GraphRAGOutput:
    """Output from GraphRAG pipeline."""
    answer: str
    retrieved_subgraph: Data
    node_ids: Tensor
    relevance_scores: Optional[Tensor]
    graph_representation: Tensor
    metadata: Dict[str, Any]


class GraphRAGPipeline(nn.Module):
    """End-to-end GraphRAG pipeline."""
    
    def __init__(
        self,
        graph_encoder: nn.Module,
        text_encoder: Optional[nn.Module] = None,
        llm: Optional[nn.Module] = None,
        retriever: Optional[GraphRetriever] = None,
        fusion: Optional[GraphTextFusion] = None,
        graph_to_seq: Optional[GraphToSequence] = None,
        graph_dim: int = 256,
        text_dim: int = 768,
        llm_dim: int = 4096,
    ):
        super().__init__()
        
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        self.llm = llm
        
        self.retriever = retriever or GraphRetriever(
            graph_encoder=graph_encoder,
            text_encoder=text_encoder,
        )
        
        self.fusion = fusion or GraphTextFusion(
            graph_dim=graph_dim,
            text_dim=text_dim,
            output_dim=graph_dim,
            fusion_type=FusionType.CROSS_ATTENTION,
        )
        
        self.graph_to_seq = graph_to_seq or GraphToSequence(
            graph_dim=graph_dim,
            llm_dim=llm_dim,
        )
    
    def forward(
        self,
        query: str,
        graph: Data,
        top_k: int = 10,
        max_new_tokens: int = 256,
        return_retrieval: bool = True,
        **generation_kwargs,
    ) -> GraphRAGOutput:
        if self.text_encoder is not None:
            query_embedding = self._encode_query(query)
        else:
            query_embedding = None
        
        retrieval_result = self.retriever.retrieve(
            query=query if query_embedding is None else query_embedding,
            graph=graph,
            top_k=top_k,
        )
        
        graph_repr = self._encode_graph(retrieval_result.subgraph)
        
        if query_embedding is not None:
            fused_repr = self.fusion(graph_repr, query_embedding)
        else:
            fused_repr = graph_repr
        
        graph_tokens = self.graph_to_seq(fused_repr)
        
        if self.llm is not None:
            answer = self._generate_answer(
                query=query,
                graph_tokens=graph_tokens,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
        else:
            answer = "[LLM not configured - graph representation computed]"
        
        return GraphRAGOutput(
            answer=answer,
            retrieved_subgraph=retrieval_result.subgraph,
            node_ids=retrieval_result.node_ids,
            relevance_scores=retrieval_result.scores,
            graph_representation=fused_repr,
            metadata={
                "num_retrieved_nodes": len(retrieval_result.node_ids),
                "retrieval_paths": retrieval_result.paths,
            },
        )
    
    @torch.no_grad()
    def _encode_query(self, query: str) -> Tensor:
        if hasattr(self.text_encoder, "encode"):
            return self.text_encoder.encode(query)
        else:
            raise NotImplementedError("Text encoder must have 'encode' method")
    
    @torch.no_grad()
    def _encode_graph(self, subgraph: Data) -> Tensor:
        self.graph_encoder.eval()
        
        node_emb, graph_emb = self.graph_encoder(
            x=subgraph.x,
            edge_index=subgraph.edge_index,
            edge_attr=getattr(subgraph, "edge_attr", None),
            batch=getattr(subgraph, "batch", None),
            return_node_embeddings=True,
        )
        
        if node_emb is not None:
            return node_emb
        return graph_emb
    
    def _generate_answer(
        self,
        query: str,
        graph_tokens: Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> str:
        return f"[Generated answer for: {query}]"
    
    def encode_and_store_graph(self, graph: Data) -> None:
        """Pre-encode and cache graph embeddings for faster retrieval."""
        with torch.no_grad():
            self.graph_encoder.eval()
            node_emb, _ = self.graph_encoder(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=getattr(graph, "edge_attr", None),
                return_node_embeddings=True,
            )
        
        self.retriever._node_embedding_cache = node_emb
        self.retriever._cached_graph_id = id(graph)
