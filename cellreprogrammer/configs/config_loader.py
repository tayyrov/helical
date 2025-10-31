"""Configuration loader for CellReprogrammer experiments.

This module provides utilities for loading and validating
configuration files for experiments.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    yaml.YAMLError
        If config file is invalid YAML.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    LOGGER.info(f"Loading configuration from: {config_path}")
    
    # Use OmegaConf for better config management
    config = OmegaConf.load(config_path)
    
    LOGGER.info("Configuration loaded successfully")
    return dict(config)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary to validate.
    
    Returns
    -------
    bool
        True if valid, raises ValueError if invalid.
    
    Raises
    ------
    ValueError
        If required keys are missing.
    """
    required_keys = ["model", "perturbation", "data"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate model config
    if "name" not in config["model"]:
        raise ValueError("Model config must specify 'name'")
    
    # Validate perturbation config
    if "type" not in config["perturbation"]:
        raise ValueError("Perturbation config must specify 'type'")
    
    if "genes" not in config["perturbation"]:
        raise ValueError("Perturbation config must specify 'genes'")
    
    LOGGER.info("Configuration validation passed")
    return True


def get_config_template() -> Dict[str, Any]:
    """Get a template configuration dictionary.
    
    Returns
    -------
    dict
        Template configuration.
    """
    return {
        "model": {
            "name": "geneformer",
            "config": {
                "model_name": "gf-12L-38M-i4096",
                "batch_size": 10,
                "device": "cuda"
            }
        },
        "perturbation": {
            "type": "overexpression",
            "genes": ["BRCA1", "TP53"],
            "strength": 2.0,
            "use_ensembl": False
        },
        "data": {
            "path": "path/to/data.h5ad",
            "subsample_cells": None,
            "gene_column": "index"
        },
        "experiment": {
            "name": "my_experiment",
            "output_dir": "results/",
            "save_embeddings": True
        }
    }


