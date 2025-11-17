"""Model factory for loading and managing models from helical.

This module provides a unified interface for loading different models
from the helical package for use in perturbation experiments.
"""

import logging
from typing import Dict, Any, Type, Optional
from anndata import AnnData
from datasets import Dataset
import numpy as np

# Import helical models
from helical.models.geneformer import Geneformer, GeneformerConfig
from helical.models.scgpt import scGPT, scGPTConfig
from helical.models.c2s import Cell2Sen, Cell2SenConfig
from helical.models.base_models import HelicalRNAModel

# TODO: Add more models as needed
# from helical.models.uce import UCE, UCEConfig
# from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig

LOGGER = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating and managing helical models.
    
    This factory provides a unified interface to load and use different
    models from helical for perturbation experiments.
    
    Example
    -------
    >>> factory = ModelFactory()
    >>> model = factory.load_model("geneformer", model_name="gf-12L-38M-i4096")
    >>> embeddings = model.get_embeddings(dataset)
    """
    
    # Registry of available models
    MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
        "geneformer": {
            "model_class": Geneformer,
            "config_class": GeneformerConfig,
            "default_config": {
                "model_name": "gf-12L-38M-i4096",
                "batch_size": 10,
                "device": "cuda",
            },
            "description": "Geneformer: Transformer-based model for single-cell RNA-seq",
        },
        "scgpt": {
            "model_class": scGPT,
            "config_class": scGPTConfig,
            "default_config": {
                "batch_size": 10,
                "device": "cuda",
            },
            "description": "scGPT: Transformer-based model for single-cell data",
        },
        "c2s": {
            "model_class": Cell2Sen,
            "config_class": Cell2SenConfig,
            "default_config": {
                "batch_size": 16,
                "model_size": "2B",
                "use_quantization": False,
            },
            "description": "Cell2Sen: LLM-based generative model for single-cell data with native perturbation support",
        },
        # Add more models here as needed
    }
    
    def __init__(self):
        """Initialize the model factory."""
        self._loaded_models: Dict[str, Any] = {}
    
    def get_available_models(self) -> list[str]:
        """Get list of available model names.
        
        Returns
        -------
        list[str]
            List of model names available in the factory.
        """
        return list(self.MODEL_REGISTRY.keys())
    
    def load_model(
        self,
        model_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ) -> HelicalRNAModel:
        """Load a model with the given configuration.
        
        Parameters
        ----------
        model_name : str
            Name of the model to load (e.g., "geneformer", "scgpt").
        config_overrides : dict, optional
            Dictionary of configuration parameters to override defaults.
        cache : bool, default=True
            Whether to cache the loaded model.
        
        Returns
        -------
        HelicalRNAModel
            The loaded model instance.
        
        Raises
        ------
        ValueError
            If model_name is not in the registry.
        """
        if model_name not in self.MODEL_REGISTRY:
            available = ", ".join(self.get_available_models())
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available}"
            )
        
        # Check if already loaded and cached
        cache_key = f"{model_name}_{id(config_overrides)}"
        if cache and cache_key in self._loaded_models:
            LOGGER.info(f"Using cached model: {model_name}")
            return self._loaded_models[cache_key]
        
        # Get model metadata
        model_info = self.MODEL_REGISTRY[model_name]
        model_class = model_info["model_class"]
        config_class = model_info["config_class"]
        default_config = model_info["default_config"].copy()
        
        # Apply config overrides
        if config_overrides:
            default_config.update(config_overrides)
        
        # Create config and model
        LOGGER.info(f"Loading model: {model_name} with config: {default_config}")
        config = config_class(**default_config)
        model = model_class(configurer=config)
        
        # Cache if requested
        if cache:
            self._loaded_models[cache_key] = model
        
        LOGGER.info(f"Successfully loaded model: {model_name}")
        return model
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model.
        
        Parameters
        ----------
        model_name : str
            Name of the model.
        
        Returns
        -------
        dict
            Dictionary containing model information.
        """
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.MODEL_REGISTRY[model_name].copy()
    
    def clear_cache(self):
        """Clear the model cache."""
        self._loaded_models.clear()
        LOGGER.info("Model cache cleared")


def get_available_models() -> list[str]:
    """Convenience function to get available models.
    
    Returns
    -------
    list[str]
        List of available model names.
    """
    factory = ModelFactory()
    return factory.get_available_models()


