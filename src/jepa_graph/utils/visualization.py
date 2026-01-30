"""
Visualization utilities for Graph JEPA.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Any

import torch
from torch import Tensor
from torch_geometric.data import Data

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def visualize_graph(
    data: Data,
    node_labels: Optional[List[str]] = None,
    node_colors: Optional[Tensor] = None,
    highlight_nodes: Optional[Tensor] = None,
    title: str = "Graph",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    layout: str = "spring",
) -> Optional[Any]:
    """Visualize a PyG graph using NetworkX and Matplotlib."""
    if not HAS_MATPLOTLIB:
        print("matplotlib/networkx not available for visualization")
        return None
    
    import numpy as np
    
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    
    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    if node_colors is not None:
        colors = node_colors.cpu().numpy()
    else:
        colors = np.ones(data.num_nodes) * 0.5
    
    nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        node_size=300,
        cmap=plt.cm.viridis,
        ax=ax,
    )
    
    if highlight_nodes is not None:
        highlight_list = highlight_nodes.cpu().tolist()
        highlight_pos = {n: pos[n] for n in highlight_list if n in pos}
        nx.draw_networkx_nodes(
            G, highlight_pos,
            nodelist=highlight_list,
            node_color="red",
            node_size=400,
            ax=ax,
        )
    
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    
    if node_labels is not None:
        label_dict = {i: label for i, label in enumerate(node_labels)}
        nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8, ax=ax)
    
    ax.set_title(title)
    ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def visualize_masking(
    original_data: Data,
    context_nodes: Tensor,
    target_nodes: Tensor,
    title: str = "JEPA Masking",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """Visualize the masking pattern for JEPA pretraining."""
    if not HAS_MATPLOTLIB:
        print("matplotlib/networkx not available for visualization")
        return None
    
    G = nx.Graph()
    G.add_nodes_from(range(original_data.num_nodes))
    
    edge_index = original_data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, seed=42)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    context_set = set(context_nodes.cpu().tolist())
    target_set = set(target_nodes.cpu().tolist())
    
    ax = axes[0]
    nx.draw(G, pos, ax=ax, node_size=200, node_color="lightblue", 
            edge_color="gray", alpha=0.7)
    ax.set_title("Original Graph")
    
    ax = axes[1]
    context_G = G.subgraph(context_set)
    node_colors = ["green" for _ in context_G.nodes()]
    nx.draw(context_G, pos, ax=ax, node_size=200, node_color=node_colors,
            edge_color="gray", alpha=0.8)
    for node in target_set:
        if node in pos:
            ax.scatter(pos[node][0], pos[node][1], s=200, c="lightgray", 
                      alpha=0.3, zorder=0)
    ax.set_title("Context (Visible)")
    
    ax = axes[2]
    target_G = G.subgraph(target_set)
    node_colors = ["red" for _ in target_G.nodes()]
    nx.draw(target_G, pos, ax=ax, node_size=200, node_color=node_colors,
            edge_color="gray", alpha=0.8)
    for node in context_set:
        if node in pos:
            ax.scatter(pos[node][0], pos[node][1], s=200, c="lightgray",
                      alpha=0.3, zorder=0)
    ax.set_title("Target (Masked)")
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def visualize_embeddings(
    embeddings: Tensor,
    labels: Optional[Tensor] = None,
    method: str = "tsne",
    title: str = "Embedding Space",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """Visualize embeddings in 2D using dimensionality reduction."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for visualization")
        return None
    
    embeddings_np = embeddings.cpu().numpy()
    
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np) - 1))
        coords = reducer.fit_transform(embeddings_np)
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings_np)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        labels_np = labels.cpu().numpy()
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels_np, 
                           cmap="tab10", s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=50, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig
