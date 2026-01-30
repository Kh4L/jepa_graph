"""
STARK-Prime dataset integration for Graph JEPA.

STARK-Prime is a large-scale knowledge graph QA benchmark from:
https://github.com/snap-stanford/stark

This module provides:
- Loading STARK-Prime KG and QA pairs
- Creating PyG Data objects with embeddings
- Subgraph retrieval for training
"""

from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import torch
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm


def load_stark_prime_kg(
    dataset_name: str = "prime",
    embed_batch_size: int = 248,
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: str = "./data/stark",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Data:
    """
    Load STARK-Prime knowledge graph as PyG Data.
    
    Args:
        dataset_name: STARK dataset name ('prime', 'mag', 'amazon')
        embed_batch_size: Batch size for embedding computation
        encoder_model: Sentence transformer model for embeddings
        cache_dir: Directory to cache processed data
        device: Device for embedding computation
        
    Returns:
        PyG Data object with node/edge embeddings and triples
    """
    from stark_qa import load_skb
    from torch_geometric.nn import SentenceTransformer
    from torch_geometric.utils import coalesce
    
    cache_path = Path(cache_dir) / f"saved_{dataset_name}_kg.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        print(f"Loading cached KG from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print(f"Building STARK-{dataset_name} knowledge graph...")
    
    # Load STARK knowledge base
    skb = load_skb(dataset_name, download_processed=True, root=cache_dir)
    
    # Initialize sentence encoder
    embedder = SentenceTransformer(model_name=encoder_model).to(device).eval()
    
    # Extract node strings
    num_nodes = skb.num_nodes()
    node_strings = []
    
    print("Extracting node text...")
    for i in tqdm(range(num_nodes), desc="Processing nodes"):
        try:
            node_name = skb[i].dictionary['details']['name']
        except (KeyError, TypeError):
            node_name = skb[i].dictionary.get('name', f"node_{i}")
        
        if isinstance(node_name, list):
            node_name = '. '.join(node_name)
        
        try:
            node_summary = skb[i].dictionary['details']['summary']
            node_str = f"{node_name}. {node_summary}"
        except (KeyError, TypeError):
            node_str = str(node_name)
        
        node_strings.append(node_str.lower())
    
    # Extract edge information
    edge_index = skb.edge_index
    edge_string_index = skb.rel_type_lst()
    edge_strings = []
    
    for edge_type in skb.edge_types:
        edge_strings.append(edge_string_index[edge_type].lower())
    
    # Create triples
    print("Creating triples...")
    triples = []
    for i in tqdm(range(len(edge_strings)), desc="Building triples"):
        src_idx = edge_index[0, i]
        dst_idx = edge_index[1, i]
        triple = (node_strings[src_idx], edge_strings[i], node_strings[dst_idx])
        triples.append(triple)
    
    # Remove duplicates while preserving order
    triples = list(dict.fromkeys(triples))
    
    # Compute node embeddings
    print("Computing node embeddings...")
    with torch.no_grad():
        node_embeddings = embedder.encode(
            node_strings,
            batch_size=embed_batch_size,
            show_progress_bar=True,
        )
    
    # Compute unique edge type embeddings
    unique_edge_strings = list(set(edge_strings))
    print(f"Computing embeddings for {len(unique_edge_strings)} edge types...")
    with torch.no_grad():
        edge_type_embeddings = embedder.encode(
            unique_edge_strings,
            batch_size=embed_batch_size,
        )
    
    # Map edge embeddings
    edge_str_to_idx = {s: i for i, s in enumerate(unique_edge_strings)}
    edge_attr_indices = [edge_str_to_idx[s] for s in edge_strings]
    edge_attr = edge_type_embeddings[edge_attr_indices]
    
    # Create PyG Data
    data = Data(
        x=torch.tensor(node_embeddings, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        num_nodes=num_nodes,
    )
    
    # Store metadata
    data.node_strings = node_strings
    data.edge_strings = edge_strings
    data.triples = triples
    data.node_id = torch.arange(num_nodes)
    data.edge_id = torch.arange(edge_index.shape[1])
    
    # Cleanup
    del embedder
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save cache
    torch.save(data, cache_path)
    print(f"Saved KG to {cache_path}")
    
    return data


def load_stark_prime_qa(
    dataset_name: str = "prime",
    cache_dir: str = "./data/stark",
) -> List[Tuple[str, str]]:
    """
    Load STARK-Prime QA pairs.
    
    Args:
        dataset_name: STARK dataset name
        cache_dir: Cache directory
        
    Returns:
        List of (question, answer) tuples
    """
    from stark_qa import load_qa, load_skb
    
    qa_dataset = load_qa(dataset_name)
    skb = load_skb(dataset_name, download_processed=True, root=cache_dir)
    
    qa_pairs = []
    for q, _, a, _ in qa_dataset:
        # Convert answer node indices to names
        answer_names = []
        for idx in a:
            try:
                name = skb[idx].dictionary['name']
                if isinstance(name, list):
                    name = name[0]
                answer_names.append(name.lower())
            except (KeyError, TypeError):
                answer_names.append(f"node_{idx}")
        
        answer_str = '|'.join(answer_names)
        qa_pairs.append((q, answer_str))
    
    return qa_pairs


def create_subgraph_dataset(
    graph_data: Data,
    qa_pairs: List[Tuple[str, str]],
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_hops: int = 3,
    k_nodes: int = 16,
    fanout: int = 10,
    topk_nodes: int = 5,
    topk_edges: int = 5,
    cache_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, List[Data]]:
    """
    Create subgraph dataset for JEPA pretraining and GraphRAG finetuning.
    
    For each QA pair, retrieves a relevant subgraph using:
    - KNN for seed node selection
    - Neighbor sampling for expansion
    - PCST filtering for relevance
    
    Args:
        graph_data: Full knowledge graph
        qa_pairs: List of (question, answer) pairs
        encoder_model: Sentence encoder for queries
        num_hops: Number of hops for neighbor sampling
        k_nodes: K for KNN seed selection
        fanout: Number of neighbors per hop
        topk_nodes: Top-k nodes for PCST
        topk_edges: Top-k edges for PCST
        cache_path: Path to cache processed dataset
        device: Computation device
        
    Returns:
        Dictionary with 'train', 'validation', 'test' splits
    """
    import random
    from torch_geometric.nn import SentenceTransformer
    
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached subgraph dataset from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    # Try to use PyG's RAG utilities if available
    try:
        from torch_geometric.llm.utils.backend_utils import (
            create_remote_backend_from_graph_data,
            make_pcst_filter,
        )
        from torch_geometric.llm.utils.feature_store import KNNRAGFeatureStore
        from torch_geometric.llm.utils.graph_store import NeighborSamplingRAGGraphStore
        from torch_geometric.llm import RAGQueryLoader
        
        USE_RAG_LOADER = True
    except ImportError:
        print("PyG RAG utilities not available, using simple subgraph sampling")
        USE_RAG_LOADER = False
    
    # Initialize encoder
    text_encoder = SentenceTransformer(model_name=encoder_model).to(device).eval()
    
    if USE_RAG_LOADER:
        # Use PyG's RAG query loader
        subgraph_filter = make_pcst_filter(
            graph_data.triples,
            text_encoder,
            topk=topk_nodes,
            topk_e=topk_edges,
            cost_e=0.5,
            num_clusters=10,
        )
        
        fs, gs = create_remote_backend_from_graph_data(
            graph_data=graph_data,
            path="backend",
            graph_db=NeighborSamplingRAGGraphStore,
            feature_db=KNNRAGFeatureStore,
        ).load()
        
        query_loader = RAGQueryLoader(
            graph_data=(fs, gs),
            subgraph_filter=subgraph_filter,
            config={
                "k_nodes": k_nodes,
                "num_neighbors": [fanout] * num_hops,
                "encoder_model": text_encoder,
            },
        )
        
        # Build dataset
        data_list = []
        for q, a in tqdm(qa_pairs, desc="Building subgraph dataset"):
            subgraph = query_loader.query(q)
            subgraph.question = q
            subgraph.label = a
            data_list.append(subgraph)
    else:
        # Simple fallback: random subgraph sampling
        data_list = []
        for q, a in tqdm(qa_pairs, desc="Building subgraph dataset (simple)"):
            subgraph = _simple_subgraph_sample(
                graph_data, q, text_encoder, k_nodes, num_hops, fanout, device
            )
            subgraph.question = q
            subgraph.label = a
            data_list.append(subgraph)
    
    # Shuffle and split (70:20:10)
    random.shuffle(data_list)
    n = len(data_list)
    
    splits = {
        "train": data_list[:int(0.7 * n)],
        "validation": data_list[int(0.7 * n):int(0.9 * n)],
        "test": data_list[int(0.9 * n):],
    }
    
    print(f"Dataset splits: train={len(splits['train'])}, "
          f"val={len(splits['validation'])}, test={len(splits['test'])}")
    
    if cache_path:
        torch.save(splits, cache_path)
        print(f"Saved to {cache_path}")
    
    return splits


def _simple_subgraph_sample(
    graph_data: Data,
    query: str,
    text_encoder,
    k_nodes: int,
    num_hops: int,
    fanout: int,
    device: str,
) -> Data:
    """Simple subgraph sampling using KNN + k-hop neighborhood."""
    from torch_geometric.utils import k_hop_subgraph
    
    # Encode query
    with torch.no_grad():
        query_emb = text_encoder.encode([query])[0]
        query_emb = torch.tensor(query_emb, device=device)
    
    # Find k nearest nodes
    node_embs = graph_data.x.to(device)
    similarities = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0), node_embs, dim=1
    )
    _, top_k_indices = torch.topk(similarities, min(k_nodes, len(similarities)))
    
    # Expand via k-hop
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=top_k_indices.cpu(),
        num_hops=num_hops,
        edge_index=graph_data.edge_index,
        relabel_nodes=True,
        num_nodes=graph_data.num_nodes,
    )
    
    # Limit size
    max_nodes = k_nodes * (fanout ** num_hops)
    if len(subset) > max_nodes:
        # Keep most similar nodes
        subset_sims = similarities[subset]
        _, keep_idx = torch.topk(subset_sims, max_nodes)
        subset = subset[keep_idx.cpu()]
        from torch_geometric.utils import subgraph
        edge_index, _ = subgraph(
            subset, graph_data.edge_index, relabel_nodes=True,
            num_nodes=graph_data.num_nodes
        )
    
    # Build subgraph
    subgraph = Data(
        x=graph_data.x[subset],
        edge_index=edge_index,
    )
    
    if graph_data.edge_attr is not None and edge_mask is not None:
        subgraph.edge_attr = graph_data.edge_attr[edge_mask]
    
    return subgraph


class STARKPrimeDataset:
    """
    STARK-Prime dataset wrapper for Graph JEPA.
    
    Provides:
    - Full KG for pretraining
    - QA subgraphs for finetuning
    - Compatibility with G-Retriever pipeline
    """
    
    def __init__(
        self,
        dataset_name: str = "prime",
        cache_dir: str = "./data/stark",
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.encoder_model = encoder_model
        self.device = device
        
        self._kg_data = None
        self._qa_pairs = None
        self._splits = None
    
    @property
    def kg(self) -> Data:
        """Get the full knowledge graph."""
        if self._kg_data is None:
            self._kg_data = load_stark_prime_kg(
                dataset_name=self.dataset_name,
                encoder_model=self.encoder_model,
                cache_dir=self.cache_dir,
                device=self.device,
            )
        return self._kg_data
    
    @property
    def qa_pairs(self) -> List[Tuple[str, str]]:
        """Get QA pairs."""
        if self._qa_pairs is None:
            self._qa_pairs = load_stark_prime_qa(
                dataset_name=self.dataset_name,
                cache_dir=self.cache_dir,
            )
        return self._qa_pairs
    
    def get_splits(
        self,
        num_hops: int = 3,
        k_nodes: int = 16,
        fanout: int = 10,
    ) -> Dict[str, List[Data]]:
        """Get train/val/test splits with subgraphs."""
        if self._splits is None:
            cache_path = os.path.join(
                self.cache_dir,
                f"{self.dataset_name}_subgraph_splits.pt"
            )
            self._splits = create_subgraph_dataset(
                graph_data=self.kg,
                qa_pairs=self.qa_pairs,
                encoder_model=self.encoder_model,
                num_hops=num_hops,
                k_nodes=k_nodes,
                fanout=fanout,
                cache_path=cache_path,
                device=self.device,
            )
        return self._splits
    
    def get_jepa_pretraining_data(self) -> Data:
        """Get KG data formatted for JEPA pretraining."""
        kg = self.kg
        # Ensure required attributes
        if not hasattr(kg, 'node_id'):
            kg.node_id = torch.arange(kg.num_nodes)
        if not hasattr(kg, 'edge_id'):
            kg.edge_id = torch.arange(kg.edge_index.size(1))
        return kg
