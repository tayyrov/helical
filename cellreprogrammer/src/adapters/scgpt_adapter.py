"""
scGPT Perturbation Adapter

Implements generic perturbation for scGPT by modifying expression values
in AnnData before processing.
"""

from typing import List, Dict, Optional
import numpy as np
from datasets import Dataset
import anndata as ad
import pandas as pd

from .base_adapter import PerturbationAdapter


class scGPTAdapter(PerturbationAdapter):
    """
    Adapter for scGPT model with generic perturbation.
    
    This adapter modifies expression values in AnnData to simulate
    gene overexpression/knockdown, then re-processes through model.
    """
    
    def __init__(self, model, model_config):
        super().__init__(model, model_config)
        self.original_adata = None
    
    def process_data(self, adata: ad.AnnData, **kwargs) -> Dataset:
        """Process AnnData using scGPT tokenizer."""
        # Store original for perturbation
        self.original_adata = adata.copy()
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
        fold_change: float = 2.0,
        **kwargs
    ) -> Dataset:
        """
        Apply perturbation by modifying AnnData expression values.
        
        Parameters
        ----------
        dataset : Dataset
            Original processed dataset (not used, we need original AnnData)
        genes_to_perturb : List[str]
            List of gene symbols or Ensembl IDs to perturb
        perturbation_type : str
            "overexpress" (multiply expression) or "knockdown" (divide expression)
        fold_change : float
            Fold change to apply (default 2.0 for overexpression)
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dataset
            Perturbed dataset
        """
        if self.original_adata is None:
            raise ValueError(
                "Original AnnData not stored. Call process_data() first."
            )
        
        # Create perturbed copy
        perturbed_adata = self.original_adata.copy()
        
        # Find genes in AnnData (try both var_names and gene columns)
        gene_names = perturbed_adata.var_names.tolist()
        
        # Try to map Ensembl IDs to gene symbols if needed
        genes_to_modify = []
        for gene in genes_to_perturb:
            # Direct match
            if gene in gene_names:
                genes_to_modify.append(gene)
            # Try case-insensitive
            elif gene.upper() in [g.upper() for g in gene_names]:
                idx = [g.upper() for g in gene_names].index(gene.upper())
                genes_to_modify.append(gene_names[idx])
            # Try matching in var columns (gene_name, gene_symbol, etc.)
            else:
                for col in perturbed_adata.var.columns:
                    if gene in perturbed_adata.var[col].values:
                        matched_genes = perturbed_adata.var[perturbed_adata.var[col] == gene].index
                        genes_to_modify.extend(matched_genes.tolist())
                        break
        
        if not genes_to_modify:
            raise ValueError(
                f"None of the genes {genes_to_perturb} found in AnnData. "
                f"Available: {gene_names[:10]}..."
            )
        
        # Apply perturbation to expression matrix
        gene_indices = [perturbed_adata.var_names.get_loc(g) for g in genes_to_modify]
        
        if perturbation_type == "overexpress":
            # Multiply expression by fold_change
            perturbed_adata.X[:, gene_indices] *= fold_change
        elif perturbation_type == "knockdown":
            # Divide expression by fold_change
            perturbed_adata.X[:, gene_indices] /= fold_change
        else:
            raise ValueError(f"Unknown perturbation_type: {perturbation_type}")
        
        # Re-process perturbed data
        perturbed_dataset = self.model.process_data(perturbed_adata, **kwargs)
        
        return perturbed_dataset
    
    def get_gene_mapping(self) -> Dict[str, str]:
        """
        Get gene mapping (scGPT uses gene symbols or Ensembl IDs directly).
        
        Returns mapping from gene identifiers to themselves.
        """
        if self.original_adata is None:
            return {}
        
        # scGPT typically uses var_names as gene identifiers
        gene_names = self.original_adata.var_names.tolist()
        return {gene: gene for gene in gene_names}
