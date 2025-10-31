"""Overexpression perturbation implementation.

This module implements gene overexpression perturbations by modifying
expression values in the data.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
from anndata import AnnData
from datasets import Dataset
import copy

from helical.models.base_models import HelicalRNAModel
from helical.utils.mapping import map_gene_symbols_to_ensembl_ids
from cellreprogrammer.src.perturbations.base_perturbation import BasePerturbation

LOGGER = logging.getLogger(__name__)


class OverexpressionPerturbation(BasePerturbation):
    """Gene overexpression perturbation.
    
    This class implements overexpression by multiplying gene expression
    values by a factor. It works with both AnnData and processed datasets.
    
    Parameters
    ----------
    model : HelicalRNAModel
        The base model to use for computing embeddings.
    perturbation_genes : list[str]
        List of genes to overexpress (gene symbols or Ensembl IDs).
    perturbation_strength : float, default=2.0
        Multiplicative factor for overexpression (e.g., 2.0 = 2x overexpression).
    use_ensembl : bool, default=False
        Whether to assume genes are provided as Ensembl IDs or convert from symbols.
    
    Example
    -------
    >>> from helical.models.geneformer import Geneformer, GeneformerConfig
    >>> import anndata as ad
    >>> 
    >>> # Load model and data
    >>> config = GeneformerConfig(model_name="gf-12L-38M-i4096")
    >>> model = Geneformer(config)
    >>> ann_data = ad.read_h5ad("data.h5ad")
    >>> 
    >>> # Create overexpression perturbation
    >>> oe = OverexpressionPerturbation(
    ...     model=model,
    ...     perturbation_genes=["BRCA1", "TP53", "MYC"],
    ...     perturbation_strength=2.5
    ... )
    >>> 
    >>> # Apply perturbation
    >>> perturbed_dataset = oe.apply(ann_data)
    >>> 
    >>> # Compute embeddings
    >>> embeddings = oe.compute_embeddings(perturbed_dataset)
    """
    
    def __init__(
        self,
        model: HelicalRNAModel,
        perturbation_genes: List[str],
        perturbation_strength: float = 2.0,
        use_ensembl: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize overexpression perturbation.
        
        Parameters
        ----------
        model : HelicalRNAModel
            The base model to use.
        perturbation_genes : list[str]
            List of genes to overexpress.
        perturbation_strength : float, default=2.0
            Multiplicative factor for overexpression.
        use_ensembl : bool, default=False
            Whether genes are Ensembl IDs.
        config : dict, optional
            Additional configuration.
        """
        super().__init__(
            model=model,
            perturbation_genes=perturbation_genes,
            perturbation_strength=perturbation_strength,
            config=config
        )
        self.use_ensembl = use_ensembl
        
        LOGGER.info(
            f"Overexpression perturbation: {len(perturbation_genes)} genes "
            f"at {perturbation_strength}x"
        )
    
    def apply(self, ann_data: AnnData) -> Dataset:
        """Apply overexpression to the data.
        
        Parameters
        ----------
        ann_data : AnnData
            The original data.
        
        Returns
        -------
        Dataset
            The perturbed dataset, processed and ready for model inference.
        
        Notes
        -----
        This method:
        1. Creates a copy of the data
        2. Identifies gene indices to perturb
        3. Multiplies expression values by perturbation_strength
        4. Returns processed dataset
        """
        LOGGER.info("Applying overexpression perturbation...")
        
        # Create a copy to avoid modifying original data
        perturbed_ann_data = copy.deepcopy(ann_data)
        
        # Convert gene symbols to Ensembl IDs if needed
        if not self.use_ensembl:
            # Map symbols to Ensembl IDs
            mapped_genes = []
            for gene in self.perturbation_genes:
                try:
                    from helical.utils.mapping import convert_list_gene_symbols_to_ensembl_ids
                    ensembl_ids = convert_list_gene_symbols_to_ensembl_ids([gene])
                    if ensembl_ids[0]:
                        mapped_genes.append(ensembl_ids[0])
                except Exception as e:
                    LOGGER.warning(f"Could not map gene {gene} to Ensembl: {e}")
                    mapped_genes.append(gene)  # Try using as-is
            self.perturbation_genes = mapped_genes
            LOGGER.info(f"Mapped to Ensembl IDs: {mapped_genes}")
        
        # Find gene indices
        # Try to match genes in var.index or var columns
        gene_indices = []
        for gene in self.perturbation_genes:
            if gene in perturbed_ann_data.var.index:
                gene_indices.append(gene)
            elif "ensembl_id" in perturbed_ann_data.var.columns:
                matches = perturbed_ann_data.var[perturbed_ann_data.var["ensembl_id"] == gene]
                if len(matches) > 0:
                    gene_indices.append(matches.index[0])
            elif gene in perturbed_ann_data.var.index:
                gene_indices.append(gene)
            else:
                LOGGER.warning(f"Gene {gene} not found in data, skipping...")
        
        if not gene_indices:
            LOGGER.error("No valid genes found for perturbation!")
            raise ValueError("No valid genes found for perturbation")
        
        LOGGER.info(f"Found {len(gene_indices)} genes to perturb: {gene_indices}")
        
        # Apply overexpression: multiply expression by perturbation strength
        # Handle both sparse and dense matrices
        from scipy.sparse import issparse
        
        if issparse(perturbed_ann_data.X):
            # For sparse matrices, convert to dense for perturbation
            perturbed_ann_data.X = perturbed_ann_data.X.todense()
        
        # Apply perturbation to all genes in gene_indices
        for gene_idx in gene_indices:
            if gene_idx in perturbed_ann_data.var.index:
                col_idx = perturbed_ann_data.var.index.get_loc(gene_idx)
                perturbed_ann_data.X[:, col_idx] *= self.perturbation_strength
        
        LOGGER.info(
            f"Applied {self.perturbation_strength}x overexpression to "
            f"{len(gene_indices)} genes"
        )
        
        # Process data for the model
        perturbed_dataset = self.model.process_data(perturbed_ann_data)
        
        return perturbed_dataset
    
    def get_perturbation_type(self) -> str:
        """Get the type of perturbation.
        
        Returns
        -------
        str
            "overexpression"
        """
        return "overexpression"
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"genes={self.perturbation_genes}, "
            f"strength={self.perturbation_strength}x)"
        )


