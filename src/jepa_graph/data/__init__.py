"""Data utilities for Graph JEPA."""

from jepa_graph.data.masking import GraphMasker, MaskingStrategy
from jepa_graph.data.dataset import GraphJEPADataset, create_jepa_dataloader
from jepa_graph.data.structural_encoding import (
    compute_random_walk_pe,
    compute_laplacian_pe,
    add_structural_encoding,
)

# STARK-Prime dataset (optional, requires stark-qa)
try:
    from jepa_graph.data.stark_prime import STARKPrimeDataset
except ImportError:
    STARKPrimeDataset = None

__all__ = [
    "GraphMasker",
    "MaskingStrategy",
    "GraphJEPADataset",
    "create_jepa_dataloader",
    "compute_random_walk_pe",
    "compute_laplacian_pe",
    "add_structural_encoding",
    "STARKPrimeDataset",
]
