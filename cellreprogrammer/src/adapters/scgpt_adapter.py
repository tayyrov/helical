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
from helical.utils.mapping import map_ensembl_ids_to_gene_symbols


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
        # Check if we need to convert Ensembl IDs to gene symbols
        # scGPT expects gene symbols, but data prepared for Geneformer uses Ensembl IDs
        adata_work = adata.copy()
        
        # Check if var_names look like Ensembl IDs
        if adata_work.var_names.str.startswith("ENSG").any() or "ensembl_id" in adata_work.var.columns:
            # Need to convert Ensembl IDs to gene symbols
            if "ensembl_id" in adata_work.var.columns:
                # Use existing ensembl_id column
                ensembl_key = "ensembl_id"
            elif adata_work.var_names.str.startswith("ENSG").all():
                # var_names are Ensembl IDs
                adata_work.var["ensembl_id"] = adata_work.var_names
                ensembl_key = "ensembl_id"
            else:
                # Try to find ensembl_id column or create from var_names
                ensembl_key = "ensembl_id"
                if ensembl_key not in adata_work.var.columns:
                    # Assume var_names are Ensembl IDs if they start with ENSG
                    adata_work.var[ensembl_key] = adata_work.var_names
            
            # Convert Ensembl IDs to gene symbols
            print("Converting Ensembl IDs to gene symbols for scGPT...")
            adata_work = map_ensembl_ids_to_gene_symbols(adata_work, ensembl_id_key=ensembl_key)
            
            # Use gene_names column for scGPT
            if "gene_names" in adata_work.var.columns:
                # Keep ensembl_id for mapping, but use gene_names for var_names
                # Store mapping: ensembl_id -> gene_symbol
                self.ensembl_to_symbol = {}
                for idx in adata_work.var.index:
                    if pd.notna(adata_work.var.loc[idx, "gene_names"]):
                        ensembl_id = adata_work.var.loc[idx, ensembl_key]
                        symbol = adata_work.var.loc[idx, "gene_names"]
                        self.ensembl_to_symbol[ensembl_id] = symbol
                
                # Filter out any None gene names first
                adata_work = adata_work[:, adata_work.var["gene_names"].notna()]
                
                # Set gene_names as var_names for scGPT
                # Fill NaN values with original var_names (convert Index to Series)
                gene_names_series = adata_work.var["gene_names"].copy()
                # Create a Series from the index to use as fill values
                index_series = pd.Series(adata_work.var.index, index=adata_work.var.index)
                gene_names_series = gene_names_series.fillna(index_series)
                
                # Before setting var_names, aggregate duplicate gene symbols
                # (Multiple Ensembl IDs can map to same gene symbol)
                if gene_names_series.duplicated().any():
                    print(f"  Aggregating {gene_names_series.duplicated().sum()} duplicate gene symbols...")
                    import scanpy as sc
                    # Temporarily set gene_names as var_names for aggregation
                    adata_work.var["temp_gene_names"] = gene_names_series.values
                    # Aggregate by summing expression values for duplicate gene symbols
                    adata_work.var_names = gene_names_series.values
                    adata_work = adata_work.aggregate_by(adata_work.var_names, func="sum")
                    # Clean up temp column if it exists
                    if "temp_gene_names" in adata_work.var.columns:
                        del adata_work.var["temp_gene_names"]
                else:
                    # No duplicates, just set var_names
                    adata_work.var_names = gene_names_series.values
                
                print(f"✓ Converted to gene symbols: {adata_work.n_vars} genes with valid symbols")
            else:
                print("⚠ Warning: No gene_names column created. Using original var_names.")
                self.ensembl_to_symbol = {}
        else:
            self.ensembl_to_symbol = {}
        
        # Store original for perturbation (keep Ensembl IDs for gene mapping)
        self.original_adata = adata.copy()
        self.processed_adata = adata_work.copy()  # Store processed version
        
        # Process with scGPT (use gene_names if available, else index)
        gene_names_param = "gene_names" if "gene_names" in adata_work.var.columns else "index"
        return self.model.process_data(adata_work, gene_names=gene_names_param, **kwargs)
    
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
        
        # Create perturbed copy from processed version (has gene symbols)
        if hasattr(self, 'processed_adata'):
            perturbed_adata = self.processed_adata.copy()
        else:
            # Fallback: create from original and convert
            perturbed_adata = self.original_adata.copy()
            if perturbed_adata.var_names.str.startswith("ENSG").any():
                if "ensembl_id" not in perturbed_adata.var.columns:
                    perturbed_adata.var["ensembl_id"] = perturbed_adata.var_names
                perturbed_adata = map_ensembl_ids_to_gene_symbols(perturbed_adata, ensembl_id_key="ensembl_id")
                if "gene_names" in perturbed_adata.var.columns:
                    perturbed_adata.var_names = perturbed_adata.var["gene_names"].fillna(perturbed_adata.var_names)
                    perturbed_adata = perturbed_adata[:, perturbed_adata.var["gene_names"].notna()]
        
        # Find genes in AnnData (use var_names which should be gene symbols now)
        gene_names = perturbed_adata.var_names.tolist()
        
        # Try to map genes (handle both Ensembl IDs and gene symbols)
        genes_to_modify = []
        
        # If original data has Ensembl IDs, we might need to convert input genes
        # Check if input looks like Ensembl IDs
        input_are_ensembl = all(g.startswith("ENSG") for g in genes_to_perturb)
        
        if input_are_ensembl and hasattr(self, 'ensembl_to_symbol') and self.ensembl_to_symbol:
            # Convert Ensembl IDs to gene symbols using the mapping we already created
            for gene in genes_to_perturb:
                if gene in self.ensembl_to_symbol:
                    symbol = self.ensembl_to_symbol[gene]
                    if symbol in gene_names:
                        genes_to_modify.append(symbol)
                    elif symbol.upper() in [g.upper() for g in gene_names]:
                        idx = [g.upper() for g in gene_names].index(symbol.upper())
                        genes_to_modify.append(gene_names[idx])
        else:
            # Input are gene symbols - match directly
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
        gene_names_param = "gene_names" if "gene_names" in perturbed_adata.var.columns else "index"
        perturbed_dataset = self.model.process_data(perturbed_adata, gene_names=gene_names_param, **kwargs)
        
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
