"""
Geneformer Perturbation Adapter

Wraps the original Geneformer InSilicoPerturber utilities
to provide a unified interface with other models.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from datasets import Dataset
import anndata as ad

from .base_adapter import PerturbationAdapter

# Import original Geneformer utilities
# Note: Assumes geneformer is installed in the active venv (e.g., pip install -e ../Geneformer)
from geneformer import InSilicoPerturber, EmbExtractor, InSilicoPerturberStats


class GeneformerAdapter(PerturbationAdapter):
    """
    Adapter for Geneformer model using original InSilicoPerturber.
    
    This adapter uses the original Geneformer perturbation utilities
    which modify tokenized sequences by moving gene tokens to the front.
    """
    
    def __init__(self, model, model_config):
        super().__init__(model, model_config)
        if InSilicoPerturber is None:
            raise ImportError(
                "Geneformer utilities not available. "
                "Install original Geneformer package: pip install -e /path/to/Geneformer/"
            )
        
        # Get model version from config
        raw_version = model_config.model_map[model_config.config["model_name"]]["model_version"].upper()
        self.model_version = "V2" if raw_version == "V3" else raw_version
        
        # Store tokenizer for gene mapping
        self.tokenizer = model.tk if hasattr(model, 'tk') else None
    
    def process_data(self, adata: ad.AnnData, **kwargs) -> Dataset:
        """Process AnnData using Geneformer tokenizer."""
        return self.model.process_data(adata, **kwargs)
    
    def extract_embeddings(
        self, 
        dataset: Dataset, 
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Extract embeddings using model's get_embeddings method."""
        return self.model.get_embeddings(dataset, **kwargs)
    
    def apply_perturbation(
        self,
        dataset: Dataset,
        genes_to_perturb: List[str],
        perturbation_type: str = "overexpress",
        **kwargs
    ) -> Dataset:
        """
        Apply perturbation using Geneformer's InSilicoPerturber.
        
        Note: This actually runs the full perturbation pipeline (not just dataset modification).
        For Geneformer, we use the original utilities which handle everything.
        """
        # Geneformer's InSilicoPerturber works differently - it takes full pipeline
        # This method is kept for interface compatibility but actual usage
        # should go through run_perturbation_experiment
        raise NotImplementedError(
            "Geneformer uses original InSilicoPerturber pipeline. "
            "Use run_perturbation_experiment() instead."
        )
    
    def get_gene_mapping(self) -> Dict[str, str]:
        """Get gene token mapping from tokenizer."""
        if self.tokenizer is None:
            return {}
        
        # Reverse mapping: token_id -> ensembl_id
        token_to_ensembl = self.tokenizer.token_to_ensembl_dict
        # Forward mapping: ensembl_id -> token_id  
        ensembl_to_token = {v: k for k, v in token_to_ensembl.items()}
        
        return ensembl_to_token
    
    def run_perturbation_experiment(
        self,
        model_path: str,
        input_data_path: str,
        output_dir: str,
        genes_to_perturb: List[str],
        cell_states: Dict,
        filter_data: Dict,
        max_ncells: Optional[int] = None,  # None = use all available cells
        forward_batch_size: int = 50,
        nproc: int = 1,
        state_embs_dict: Optional[Dict] = None,
    ) -> Dict:
        """
        Run full Geneformer perturbation experiment using original utilities.
        
        This is the recommended way to use Geneformer adapter as it leverages
        the original InSilicoPerturber pipeline.
        """
        import os
        os.environ['DATASETS_NUM_PROC'] = str(nproc)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Initialize perturber
        isp = InSilicoPerturber(
            perturb_type="overexpress",
            perturb_rank_shift=None,
            genes_to_perturb=genes_to_perturb,
            combos=0,
            anchor_gene=None,
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cls",
            cell_emb_style="mean_pool",
            filter_data=filter_data,
            cell_states_to_model=cell_states,
            state_embs_dict=state_embs_dict,
            max_ncells=max_ncells,
            emb_layer=-1,
            forward_batch_size=forward_batch_size,
            model_version=self.model_version,
            nproc=nproc
        )
        
        # Run perturbation
        output_name = "perturbation"
        isp.perturb_data(
            model_path,
            input_data_path,
            output_dir,
            output_name
        )
        
        return {
            "perturber": isp,
            "output_name": output_name,
            "output_dir": output_dir
        }
