"""
Base Perturbation Adapter

Defines the interface that all model-specific adapters must implement
for unified perturbation experiments.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from datasets import Dataset
import anndata as ad


class PerturbationAdapter(ABC):
    """
    Base class for model-specific perturbation adapters.
    
    Each model adapter provides a unified interface for:
    - Loading and preparing data
    - Extracting baseline embeddings
    - Applying perturbations (overexpression, knockdown, etc.)
    - Extracting perturbed embeddings
    - Computing shifts/distances
    """
    
    def __init__(self, model, model_config):
        """
        Initialize adapter with a model instance.
        
        Parameters
        ----------
        model : HelicalRNAModel
            The model instance (e.g., Geneformer, scGPT)
        model_config : ModelConfig
            The model configuration object
        """
        self.model = model
        self.config = model_config
    
    @abstractmethod
    def process_data(self, adata: ad.AnnData, **kwargs) -> Dataset:
        """
        Process AnnData into model-specific format.
        
        Parameters
        ----------
        adata : AnnData
            Input single-cell data
        **kwargs
            Additional model-specific parameters
            
        Returns
        -------
        Dataset
            Processed dataset ready for model inference
        """
        pass
    
    @abstractmethod
    def extract_embeddings(
        self, 
        dataset: Dataset, 
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Extract cell embeddings from processed dataset.
        
        Parameters
        ----------
        dataset : Dataset
            Processed dataset
        batch_size : int, optional
            Batch size for inference
        **kwargs
            Additional model-specific parameters
            
        Returns
        -------
        np.ndarray
            Cell embeddings (n_cells, embedding_dim)
        """
        pass
    
    @abstractmethod
    def apply_perturbation(
        self,
        dataset: Dataset,
        genes_to_perturb: List[str],
        perturbation_type: str = "overexpress",
        **kwargs
    ) -> Dataset:
        """
        Apply perturbation to dataset (modify genes).
        
        Parameters
        ----------
        dataset : Dataset
            Original processed dataset
        genes_to_perturb : List[str]
            List of gene identifiers (Ensembl IDs or symbols)
        perturbation_type : str
            Type of perturbation: "overexpress", "knockdown", etc.
        **kwargs
            Additional model-specific parameters
            
        Returns
        -------
        Dataset
            Perturbed dataset
        """
        pass
    
    @abstractmethod
    def get_gene_mapping(self) -> Dict[str, str]:
        """
        Get mapping between gene symbols and model's internal gene IDs.
        
        Returns
        -------
        Dict[str, str]
            Mapping from gene symbols/Ensembl IDs to model tokens/IDs
        """
        pass
    
    def compute_shift(
        self,
        baseline_embeddings: np.ndarray,
        perturbed_embeddings: np.ndarray,
        goal_embeddings: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute shift toward goal state after perturbation.
        
        Parameters
        ----------
        baseline_embeddings : np.ndarray
            Embeddings before perturbation (n_cells, dim)
        perturbed_embeddings : np.ndarray
            Embeddings after perturbation (n_cells, dim)
        goal_embeddings : np.ndarray
            Goal state embeddings (can be single vector or per-cell)
        metric : str
            Distance metric: "cosine", "euclidean"
            
        Returns
        -------
        np.ndarray
            Shift values per cell (positive = moved toward goal)
        """
        from scipy.spatial.distance import cosine, euclidean
        
        n_cells = baseline_embeddings.shape[0]
        shifts = np.zeros(n_cells)
        
        # If goal is single vector, broadcast
        if goal_embeddings.ndim == 1:
            goal_embeddings = goal_embeddings.reshape(1, -1)
            goal_embeddings = np.repeat(goal_embeddings, n_cells, axis=0)
        
        for i in range(n_cells):
            if metric == "cosine":
                baseline_dist = cosine(baseline_embeddings[i], goal_embeddings[i])
                perturbed_dist = cosine(perturbed_embeddings[i], goal_embeddings[i])
            else:  # euclidean
                baseline_dist = euclidean(baseline_embeddings[i], goal_embeddings[i])
                perturbed_dist = euclidean(perturbed_embeddings[i], goal_embeddings[i])
            
            # Shift = improvement (positive = closer to goal)
            shifts[i] = baseline_dist - perturbed_dist
        
        return shifts
