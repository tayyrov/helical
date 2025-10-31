"""Base class for perturbation experiments.

This module provides the abstract base class for implementing
different types of genetic perturbations (overexpression, knockdown, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging
from anndata import AnnData
from datasets import Dataset
import numpy as np

from helical.models.base_models import HelicalRNAModel

LOGGER = logging.getLogger(__name__)


class BasePerturbation(ABC):
    """Abstract base class for genetic perturbation experiments.
    
    This class defines the interface that all perturbation experiments
    should follow. Subclasses should implement the perturbation logic
    specific to their type (overexpression, knockdown, etc.).
    
    Parameters
    ----------
    model : HelicalRNAModel
        The base model to use for computing embeddings.
    perturbation_genes : list[str], optional
        List of genes to perturb. Can be gene symbols or Ensembl IDs.
    perturbation_strength : float, default=1.0
        Strength of the perturbation (interpretation depends on subclass).
    config : dict, optional
        Additional configuration parameters.
    
    Example
    -------
    >>> from helical.models.geneformer import Geneformer, GeneformerConfig
    >>> from cellreprogrammer.src.perturbations import BasePerturbation
    >>> 
    >>> # Load base model
    >>> config = GeneformerConfig(model_name="gf-12L-38M-i4096")
    >>> model = Geneformer(config)
    >>> 
    >>> # Create perturbation
    >>> perturbation = MyPerturbation(
    ...     model=model,
    ...     perturbation_genes=["BRCA1", "TP53"],
    ...     perturbation_strength=2.0
    ... )
    >>> 
    >>> # Apply perturbation
    >>> perturbed_data = perturbation.apply(ann_data)
    """
    
    def __init__(
        self,
        model: HelicalRNAModel,
        perturbation_genes: Optional[List[str]] = None,
        perturbation_strength: float = 1.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize perturbation experiment.
        
        Parameters
        ----------
        model : HelicalRNAModel
            The base model to use for computing embeddings.
        perturbation_genes : list[str], optional
            List of genes to perturb.
        perturbation_strength : float, default=1.0
            Strength of the perturbation.
        config : dict, optional
            Additional configuration parameters.
        """
        self.model = model
        self.perturbation_genes = perturbation_genes or []
        self.perturbation_strength = perturbation_strength
        self.config = config or {}
        
        LOGGER.info(
            f"Initialized {self.__class__.__name__} with "
            f"{len(self.perturbation_genes)} genes, strength={perturbation_strength}"
        )
    
    @abstractmethod
    def apply(self, ann_data: AnnData) -> Dataset:
        """Apply perturbation to data.
        
        Parameters
        ----------
        ann_data : AnnData
            The original (unperturbed) data.
        
        Returns
        -------
        Dataset
            The perturbed dataset, ready for model processing.
        """
        pass
    
    @abstractmethod
    def get_perturbation_type(self) -> str:
        """Get the type of perturbation.
        
        Returns
        -------
        str
            The type of perturbation (e.g., "overexpression", "knockdown").
        """
        pass
    
    def compute_embeddings(self, dataset: Dataset) -> np.ndarray:
        """Compute embeddings using the base model.
        
        Parameters
        ----------
        dataset : Dataset
            The processed dataset.
        
        Returns
        -------
        np.ndarray
            The computed embeddings.
        """
        LOGGER.info("Computing embeddings...")
        embeddings = self.model.get_embeddings(dataset)
        LOGGER.info(f"Computed embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def compare_conditions(
        self, 
        control_dataset: Dataset, 
        perturbed_dataset: Dataset
    ) -> Dict[str, Any]:
        """Compare control vs perturbed conditions.
        
        Parameters
        ----------
        control_dataset : Dataset
            The control condition.
        perturbed_dataset : Dataset
            The perturbed condition.
        
        Returns
        -------
        dict
            Dictionary containing comparison results (embeddings, distances, etc.).
        """
        LOGGER.info("Comparing control vs perturbed conditions...")
        
        # Compute embeddings
        control_embeddings = self.compute_embeddings(control_dataset)
        perturbed_embeddings = self.compute_embeddings(perturbed_dataset)
        
        # Compute mean embeddings
        control_mean = np.mean(control_embeddings, axis=0)
        perturbed_mean = np.mean(perturbed_embeddings, axis=0)
        
        # Compute euclidean distance between means
        distance = np.linalg.norm(perturbed_mean - control_mean)
        
        results = {
            "control_embeddings": control_embeddings,
            "perturbed_embeddings": perturbed_embeddings,
            "control_mean": control_mean,
            "perturbed_mean": perturbed_mean,
            "distance": distance,
        }
        
        LOGGER.info(f"Distance between conditions: {distance:.4f}")
        
        return results
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"genes={len(self.perturbation_genes)}, "
            f"strength={self.perturbation_strength})"
        )


