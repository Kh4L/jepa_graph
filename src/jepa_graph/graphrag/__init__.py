"""GraphRAG integration with JEPA-pretrained encoders."""

from jepa_graph.graphrag.retriever import GraphRetriever, HybridRetriever
from jepa_graph.graphrag.fusion import GraphTextFusion, FusionType
from jepa_graph.graphrag.pipeline import GraphRAGPipeline
from jepa_graph.graphrag.gretriever import (
    # Main functions (use PyG's GRetriever)
    create_gretriever_with_jepa_encoder,
    create_gretriever_from_jepa,
    load_jepa_encoder,
    GRetrieverTrainer,
    # Helper functions (match baseline)
    get_loss,
    inference_step,
    adjust_learning_rate,
)

__all__ = [
    # Retrieval
    "GraphRetriever",
    "HybridRetriever",
    # Fusion
    "GraphTextFusion",
    "FusionType",
    # Pipeline
    "GraphRAGPipeline",
    # G-Retriever integration
    "create_gretriever_with_jepa_encoder",
    "create_gretriever_from_jepa",
    "load_jepa_encoder",
    "GRetrieverTrainer",
    # Utilities
    "get_loss",
    "inference_step",
    "adjust_learning_rate",
]
