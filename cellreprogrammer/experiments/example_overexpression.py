"""Example script for gene overexpression perturbation experiments.

This script demonstrates how to use the CellReprogrammer framework
to perform overexpression experiments with Geneformer and other models.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path to import cellreprogrammer
sys.path.insert(0, str(Path(__file__).parent.parent))

from cellreprogrammer.src.models.model_factory import ModelFactory
from cellreprogrammer.src.perturbations.overexpression import OverexpressionPerturbation
from helical.utils.mapping import map_gene_symbols_to_ensembl_ids
from helical.utils import get_anndata_from_hf_dataset
from datasets import load_dataset
import anndata as ad

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def main():
    """Run overexpression perturbation experiment."""
    
    print("=" * 80)
    print("CellReprogrammer: Gene Overexpression Example")
    print("=" * 80)
    
    # Initialize model factory
    factory = ModelFactory()
    print(f"\nAvailable models: {factory.get_available_models()}")
    
    # Load Geneformer model
    print("\n1. Loading Geneformer model...")
    model = factory.load_model(
        "geneformer",
        config_overrides={
            "model_name": "gf-12L-38M-i4096",
            "batch_size": 8,
            "device": "cuda"  # or "cpu"
        }
    )
    
    # Load data
    print("\n2. Loading data...")
    # Option 1: Load from HuggingFace dataset
    # hf_dataset = load_dataset(
    #     "helical-ai/yolksac_human",
    #     split="train[:10]",
    #     trust_remote_code=True,
    #     download_mode="reuse_cache_if_exists",
    # )
    # ann_data = get_anndata_from_hf_dataset(hf_dataset)
    
    # Option 2: Load from local file
    # ann_data = ad.read_h5ad("path/to/your/data.h5ad")
    
    # For demonstration, we'll use a sample approach
    # Note: Replace this with your actual data loading
    print("  ⚠️  Please replace this with your actual data loading code")
    print("  Example: ann_data = ad.read_h5ad('your_data.h5ad')")
    
    # Example perturbation genes (replace with your genes of interest)
    perturbation_genes = ["BRCA1", "TP53", "MYC"]
    print(f"\n3. Setting up overexpression for: {perturbation_genes}")
    
    # Create overexpression perturbation
    oe = OverexpressionPerturbation(
        model=model,
        perturbation_genes=perturbation_genes,
        perturbation_strength=2.5,  # 2.5x overexpression
        use_ensembl=False  # Using gene symbols
    )
    
    # Note: The actual perturbation application would be:
    # perturbed_dataset = oe.apply(ann_data)
    # embeddings = oe.compute_embeddings(perturbed_dataset)
    # print(f"\n4. Computed embeddings shape: {embeddings.shape}")
    
    print("\n✓ Setup complete!")
    print("\nTo run the full experiment:")
    print("1. Load your AnnData object")
    print("2. Call oe.apply(ann_data) to apply perturbation")
    print("3. Call oe.compute_embeddings() to get embeddings")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()


