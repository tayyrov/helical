"""
Model Adapters for Cell Reprogramming Experiments

Each adapter provides a unified interface for perturbation experiments
across different models (Geneformer, scGPT, TranscriptFormer, etc.)
"""

from .base_adapter import PerturbationAdapter
from .geneformer_adapter import GeneformerAdapter
from .scgpt_adapter import scGPTAdapter

__all__ = ["PerturbationAdapter", "GeneformerAdapter", "scGPTAdapter"]
