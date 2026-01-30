"""
JEPA-Graph: Joint Embedding Predictive Architecture for Graph Neural Networks

This package implements JEPA-style self-supervised pretraining for GNN encoders,
designed for integration with GraphRAG retrieval and question-answering systems.

Key components:
- Graph JEPA pretraining (context encoder, EMA target encoder, predictor)
- Flexible graph masking strategies (node-centric, edge-centric, subgraph)
- GraphRAG integration (retrieval, fusion, LLM interface)
"""

__version__ = "0.1.0"

from jepa_graph.models.graph_jepa import GraphJEPA
from jepa_graph.models.encoders import GraphEncoder, GraphTransformerEncoder
from jepa_graph.models.predictor import JEPAPredictor
from jepa_graph.data.masking import GraphMasker, MaskingStrategy

__all__ = [
    "GraphJEPA",
    "GraphEncoder",
    "GraphTransformerEncoder",
    "JEPAPredictor",
    "GraphMasker",
    "MaskingStrategy",
]
