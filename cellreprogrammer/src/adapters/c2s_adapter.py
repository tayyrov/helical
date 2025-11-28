"""
Cell2Sen (C2S) Perturbation Adapter

Implements perturbation for Cell2Sen using its native generative perturbation method.
Cell2Sen uses text-based perturbations to generate perturbed cell sentences.
"""

from typing import List, Dict, Optional
import numpy as np
from datasets import Dataset
import anndata as ad
import pandas as pd

from .base_adapter import PerturbationAdapter


class Cell2SenAdapter(PerturbationAdapter):
    """
    Adapter for Cell2Sen model with native generative perturbation.
    
    Cell2Sen uses a text-based approach where perturbations are described
    in natural language (e.g., "overexpress POU5F1") and the model generates
    a new rank-ordered gene list representing the perturbed state.
    """
    
    def __init__(self, model, model_config):
        super().__init__(model, model_config)
        self.original_adata = None
        self.processed_dataset = None
    
    def process_data(self, adata: ad.AnnData, **kwargs) -> Dataset:
        """Process AnnData using Cell2Sen's process_data method."""
        # Store original for perturbation
        self.original_adata = adata.copy()
        
        # Cell2Sen expects gene symbols in var_names
        # Check if we need to convert Ensembl IDs to gene symbols
        if adata.var_names.str.startswith("ENSG").any() or "ensembl_id" in adata.var.columns:
            # Need to convert Ensembl IDs to gene symbols
            from helical.utils.mapping import map_ensembl_ids_to_gene_symbols
            
            if "ensembl_id" not in adata.var.columns:
                adata.var["ensembl_id"] = adata.var_names
            
            print("Converting Ensembl IDs to gene symbols for Cell2Sen...")
            adata_work = map_ensembl_ids_to_gene_symbols(adata, ensembl_id_key="ensembl_id")
            
            if "gene_names" in adata_work.var.columns:
                # Use gene_names as var_names
                # Convert Index to Series for fillna compatibility
                gene_names_series = adata_work.var["gene_names"].copy()
                original_names_series = pd.Series(adata_work.var_names, index=adata_work.var.index)
                gene_names_series = gene_names_series.fillna(original_names_series)
                adata_work.var_names = gene_names_series.values
                adata_work = adata_work[:, adata_work.var["gene_names"].notna()]
                # Make var_names unique to avoid warnings (do this before filtering genes)
                if adata_work.var_names.duplicated().any():
                    adata_work.var_names_make_unique()
                print(f"✓ Converted to gene symbols: {adata_work.n_vars} genes")
            else:
                adata_work = adata
        else:
            adata_work = adata
        
        # Filter to top N genes per cell to avoid OOM from extremely long sequences
        # Cell2Sen creates cell sentences from all genes, which can be 20k+ tokens
        # This causes quadratic memory in attention masks
        max_genes_per_cell = kwargs.pop('max_genes_per_cell', 5000)  # Remove from kwargs
        if adata_work.n_vars > max_genes_per_cell:
            print(f"⚠ Filtering to top {max_genes_per_cell} genes per cell (to avoid OOM)")
            print(f"  Original: {adata_work.n_vars} genes")
            # Filter to top N most variable genes
            import scanpy as sc
            # Make a copy to avoid view issues
            adata_work = adata_work.copy()
            sc.pp.highly_variable_genes(adata_work, n_top_genes=max_genes_per_cell, flavor='seurat_v3')
            adata_work = adata_work[:, adata_work.var['highly_variable']]
            # Make var_names unique if needed
            if adata_work.var_names.duplicated().any():
                adata_work.var_names_make_unique()
            print(f"  Filtered: {adata_work.n_vars} genes")
        
        # Process with Cell2Sen (max_genes_per_cell already removed from kwargs)
        dataset = self.model.process_data(adata_work, **kwargs)
        self.processed_dataset = dataset
        return dataset
    
    def extract_embeddings(
        self, 
        dataset: Dataset, 
        batch_size: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """Extract embeddings using Cell2Sen's get_embeddings method."""
        return self.model.get_embeddings(dataset, **kwargs)
    
    def apply_perturbation(
        self,
        dataset: Dataset,
        genes_to_perturb: List[str],
        perturbation_type: str = "overexpress",
        fold_change: Optional[float] = None,
        **kwargs
    ) -> Dataset:
        """
        Apply perturbation using Cell2Sen's native generative method.
        
        Cell2Sen uses text-based perturbations. This method converts gene lists
        to perturbation descriptions and uses get_perturbations() to generate
        perturbed cell sentences.
        
        Parameters
        ----------
        dataset : Dataset
            Original processed dataset
        genes_to_perturb : List[str]
            List of gene symbols to perturb
        perturbation_type : str
            "overexpress" or "knockdown"
        fold_change : float, optional
            Fold change (for display in perturbation text, Cell2Sen doesn't use this directly)
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dataset
            Dataset with perturbed_cell_sentence column added
        """
        if self.processed_dataset is None:
            raise ValueError(
                "Processed dataset not stored. Call process_data() first."
            )
        
        # Convert genes to perturbation description
        # Based on the tutorial, Cell2Sen expects simple perturbation names like "IFNg"
        # The model may not understand "overexpress" format, so let's try just the gene names
        # or a very simple format that the model might recognize
        if perturbation_type == "overexpress":
            if len(genes_to_perturb) == 1:
                # Single gene: try just the gene name first (like "IFNg" in tutorial)
                pert_text = genes_to_perturb[0]
            else:
                # Multiple genes: try comma-separated gene names
                # This is closer to natural language that the model might understand
                pert_text = ", ".join(genes_to_perturb)
        elif perturbation_type == "knockdown":
            if len(genes_to_perturb) == 1:
                pert_text = genes_to_perturb[0]
            else:
                pert_text = ", ".join(genes_to_perturb)
        else:
            raise ValueError(f"Unknown perturbation_type: {perturbation_type}")
        
        # Add fold change info if provided
        if fold_change:
            pert_text += f" by {fold_change}x"
        
        # Remove any existing perturbed_cell_sentence column to ensure fresh perturbations
        # This is important when applying multiple perturbations to the same dataset
        if 'perturbed_cell_sentence' in dataset.column_names:
            dataset = dataset.remove_columns(['perturbed_cell_sentence'])
        
        # Create perturbation list (one per cell)
        perturbations_list = [pert_text] * len(dataset)
        
        # Debug: Print first prompt to verify format
        if len(dataset) > 0:
            from helical.models.c2s.config import PERTURBATION_PROMPT
            first_cell_sentence = dataset['cell_sentence'][0] if 'cell_sentence' in dataset.column_names else 'N/A'
            first_organism = dataset['organism'][0] if 'organism' in dataset.column_names else 'human'
            first_prompt = PERTURBATION_PROMPT.format(
                organism=first_organism,
                perturbation=pert_text,
                cell_sentence=first_cell_sentence[:100] + "..." if len(first_cell_sentence) > 100 else first_cell_sentence
            )
            print(f"  DEBUG: First perturbation prompt (first 300 chars):\n{first_prompt[:300]}...")
        
        # Use Cell2Sen's native get_perturbations method
        perturbed_dataset, perturbed_sentences = self.model.get_perturbations(
            dataset,
            perturbations_list=perturbations_list
        )
        
        # Validate generated perturbations
        # Check if perturbations are reasonable (not just special tokens or very short)
        if perturbed_sentences:
            # Find first non-None sentence for validation
            first_valid_sentence = None
            for s in perturbed_sentences:
                if s is not None and len(s.strip()) > 0:
                    first_valid_sentence = s
                    break
            
            if first_valid_sentence:
                # Count genes in the sentence (split by whitespace)
                genes = first_valid_sentence.strip().split()
                gene_count = len(genes)
                sentence_length = len(first_valid_sentence)
                
                # Check if the sentence is suspiciously short or contains only special tokens
                # Valid cell sentences should have many genes (typically 100+)
                # If we only have 1-2 tokens, it's likely just a special token
                is_suspicious = (
                    gene_count <= 2 or  # Too few genes
                    sentence_length < 20 or  # Too short
                    first_valid_sentence.startswith('<') and first_valid_sentence.endswith('>')  # Looks like special token
                )
                
                if is_suspicious:
                    print("\n" + "=" * 80)
                    print("ERROR: Invalid perturbation generation detected")
                    print("=" * 80)
                    print(f"Generated sentence: '{first_valid_sentence}'")
                    print(f"Length: {gene_count} genes, {sentence_length} chars")
                    print()
                    print("The base Cell2Sen model is not generating proper perturbations.")
                    print("This is because the base model is not finetuned for perturbation prediction.")
                    print()
                    print("SOLUTION:")
                    print("You need to finetune the Cell2Sen model on perturbation data first.")
                    print("See the tutorial for details:")
                    print("  https://github.com/vandijklab/cell2sentence/blob/main/tutorials/c2s_tutorial_10_perturbation_response_prediction.ipynb")
                    print()
                    print("Alternatively:")
                    print("  1. Use a pre-finetuned model if available")
                    print("  2. Use Geneformer or scGPT for perturbation experiments (they work out-of-the-box)")
                    print("=" * 80)
                    print()
                    raise ValueError(
                        "Cell2Sen model generated invalid perturbations. "
                        "The base model needs to be finetuned for perturbation prediction. "
                        "See the error message above for details."
                    )
        
        return perturbed_dataset
    
    def get_gene_mapping(self) -> Dict[str, str]:
        """
        Get gene mapping for Cell2Sen.
        
        Cell2Sen uses gene symbols directly, so returns identity mapping.
        """
        if self.original_adata is None:
            return {}
        
        gene_names = self.original_adata.var_names.tolist()
        return {gene: gene for gene in gene_names}
    
    def extract_perturbed_embeddings(
        self,
        perturbed_dataset: Dataset,
        **kwargs
    ) -> np.ndarray:
        """
        Extract embeddings from perturbed cell sentences.
        
        This is a helper method specific to Cell2Sen that processes
        the generated perturbed sentences back into embeddings.
        
        Parameters
        ----------
        perturbed_dataset : Dataset
            Dataset with perturbed_cell_sentence column (from apply_perturbation)
            
        Returns
        -------
        np.ndarray
            Embeddings from perturbed sentences (NaN for cells without valid perturbations)
        """
        # Get perturbed sentences and organisms
        perturbed_sentences = perturbed_dataset['perturbed_cell_sentence']
        organisms = perturbed_dataset['organism']
        
        # Filter out None values (cells without valid perturbations)
        valid_indices = [i for i, s in enumerate(perturbed_sentences) if s is not None]
        
        if len(valid_indices) == 0:
            raise ValueError("No valid perturbed sentences found in dataset")
        
        # Create a minimal dataset for embedding extraction
        # Cell2Sen's get_embeddings expects 'cell_sentence' and 'organism'
        valid_sentences = [perturbed_sentences[i] for i in valid_indices]
        valid_organisms = [organisms[i] for i in valid_indices]
        
        # Create temporary dataset with required fields
        from datasets import Dataset as HFDataset
        temp_dataset = HFDataset.from_dict({
            'cell_sentence': valid_sentences,
            'organism': valid_organisms,
        })
        
        # Extract embeddings using Cell2Sen's native method
        embeddings = self.model.get_embeddings(temp_dataset, **kwargs)
        
        # Create full embedding array with NaN for invalid cells
        # This allows downstream code to filter out invalid cells
        full_embeddings = np.full((len(perturbed_sentences), embeddings.shape[1]), np.nan)
        for idx, valid_idx in enumerate(valid_indices):
            full_embeddings[valid_idx] = embeddings[idx]
        
        return full_embeddings

