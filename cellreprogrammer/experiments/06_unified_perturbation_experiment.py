"""
Unified Perturbation Experiment Runner

Runs in-silico perturbation experiments across multiple models
using the adapter framework for consistent comparison.

Usage:
    python 06_unified_perturbation_experiment.py --model scgpt --genes OCT4 SOX2 KLF4 MYC
    python 06_unified_perturbation_experiment.py --model geneformer --genes ENSG00000204531 ...
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# Path setup
script_dir = Path(__file__).resolve().parent
BASE_DIR = script_dir.parent.parent
if not BASE_DIR.exists():
    BASE_DIR = Path("/home/ubuntu/data-at-virginia/helical")
CELLREPROGRAMMER_DIR = BASE_DIR / "cellreprogrammer"

# Add to path
sys.path.insert(0, str(CELLREPROGRAMMER_DIR))

# Import adapters
from src.adapters import GeneformerAdapter, scGPTAdapter

# Model imports
from helical.models.geneformer import Geneformer, GeneformerConfig
from helical.models.scgpt import scGPT, scGPTConfig
from helical.utils.downloader import Downloader
import torch

# Detect available device
def get_device():
    """Detect and return the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠ No GPU detected, using CPU")
    return device


# Model factory
# Note: device detection happens at runtime in run_generic_perturbation_experiment
MODEL_REGISTRY = {
    "geneformer": {
        "model_class": Geneformer,
        "config_class": GeneformerConfig,
        "adapter_class": GeneformerAdapter,
        "default_config": {"model_name": "gf-20L-151M-i4096", "batch_size": 50},
    },
    "scgpt": {
        "model_class": scGPT,
        "config_class": scGPTConfig,
        "adapter_class": scGPTAdapter,
        "default_config": {"batch_size": 50},  # device will be added dynamically
    },
}


def load_data(data_path: Path):
    """Load AnnData from prepared data."""
    import anndata as ad
    return ad.read_h5ad(data_path)


def run_generic_perturbation_experiment(
    model_name: str,
    data_path: Path,
    genes_to_perturb: List[str],
    random_genes: List[str],
    output_dir: Path,
    start_state: str = "Fibroblast",
    goal_state: str = "iPSC",
    max_cells: Optional[int] = None,  # None = use all cells (recommended for accuracy)
    fold_change: float = 2.0,
):
    """
    Run perturbation experiment with a model using the adapter framework.
    
    This works for models with generic perturbation (scGPT, TranscriptFormer, etc.)
    but NOT for Geneformer (which uses original utilities).
    """
    print("=" * 80)
    print(f"Unified Perturbation Experiment: {model_name.upper()}")
    print("=" * 80)
    print()
    
    # Get model config
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    registry = MODEL_REGISTRY[model_name]
    config_class = registry["config_class"]
    model_class = registry["model_class"]
    adapter_class = registry["adapter_class"]
    default_config = registry["default_config"].copy()  # Copy to avoid modifying original
    
    # Auto-detect device for scGPT
    if model_name == "scgpt" and "device" not in default_config:
        device = get_device()
        default_config["device"] = device
        print(f"Using device: {device}")
    
    # Create config and model
    print(f"Initializing {model_name}...")
    config = config_class(**default_config)
    
    # Download model files if needed
    if hasattr(config, 'list_of_files_to_download'):
        print("Downloading model files...")
        downloader = Downloader()
        for file in config.list_of_files_to_download:
            downloader.download_via_name(file)
    
    model = model_class(config)
    adapter = adapter_class(model, config)
    print("✓ Model initialized")
    print()
    
    # Load data
    print("Loading data...")
    adata = load_data(data_path)
    print(f"✓ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Filter to start state
    if start_state:
        adata = adata[adata.obs['cell_type'] == start_state].copy()
        print(f"✓ Filtered to {start_state}: {adata.n_obs} cells")
    
    # Limit cells if requested (using all cells is recommended for accuracy)
    if max_cells and adata.n_obs > max_cells:
        import scanpy as sc
        print(f"⚠ Limiting to {max_cells} cells (for faster testing)")
        print(f"  Note: Using all {adata.n_obs} cells would be more accurate")
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
        print(f"✓ Subsampled to {max_cells} cells")
    elif max_cells is None:
        print(f"✓ Using all {adata.n_obs} cells (recommended for accuracy)")
    print()
    
    # Extract goal state embeddings (for comparison)
    print(f"Extracting goal state embeddings ({goal_state})...")
    goal_adata = load_data(data_path)
    goal_adata = goal_adata[goal_adata.obs['cell_type'] == goal_state].copy()
    if max_cells and goal_adata.n_obs > max_cells:
        import scanpy as sc
        sc.pp.subsample(goal_adata, n_obs=max_cells, random_state=42)
    
    goal_dataset = adapter.process_data(goal_adata)
    goal_embeddings = adapter.extract_embeddings(goal_dataset)
    goal_centroid = np.mean(goal_embeddings, axis=0)
    print(f"✓ Goal state embeddings: {goal_embeddings.shape}")
    print()
    
    # Process baseline data
    print("Processing baseline data...")
    baseline_dataset = adapter.process_data(adata)
    baseline_embeddings = adapter.extract_embeddings(baseline_dataset)
    print(f"✓ Baseline embeddings: {baseline_embeddings.shape}")
    print()
    
    # Test target genes
    print(f"Testing perturbation: {', '.join(genes_to_perturb)}")
    perturbed_dataset = adapter.apply_perturbation(
        baseline_dataset,
        genes_to_perturb,
        perturbation_type="overexpress",
        fold_change=fold_change
    )
    perturbed_embeddings = adapter.extract_embeddings(perturbed_dataset)
    print(f"✓ Perturbed embeddings: {perturbed_embeddings.shape}")
    
    # Calculate shifts
    shifts = adapter.compute_shift(
        baseline_embeddings,
        perturbed_embeddings,
        goal_centroid,
        metric="cosine"
    )
    print(f"✓ Mean shift: {np.mean(shifts):.6f} ± {np.std(shifts):.6f}")
    print()
    
    # Test random genes
    print(f"Testing random control: {', '.join(random_genes)}")
    random_dataset = adapter.apply_perturbation(
        baseline_dataset,
        random_genes,
        perturbation_type="overexpress",
        fold_change=fold_change
    )
    random_embeddings = adapter.extract_embeddings(random_dataset)
    random_shifts = adapter.compute_shift(
        baseline_embeddings,
        random_embeddings,
        goal_centroid,
        metric="cosine"
    )
    print(f"✓ Random mean shift: {np.mean(random_shifts):.6f} ± {np.std(random_shifts):.6f}")
    print()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results = pd.DataFrame({
        'cell_idx': range(len(shifts)),
        'target_shift': shifts,
        'random_shift': random_shifts,
    })
    results.to_csv(output_dir / f"{model_name}_perturbation_results.csv", index=False)
    
    # Calculate fold improvement for CSV
    csv_fold_improvement = None
    if abs(np.mean(random_shifts)) > 1e-8:  # Use smaller threshold to catch very small shifts
        csv_fold_improvement = abs(np.mean(shifts)) / abs(np.mean(random_shifts))
    
    summary = pd.DataFrame([{
        'model': model_name,
        'target_genes': ', '.join(genes_to_perturb),
        'random_genes': ', '.join(random_genes),
        'fold_change': fold_change,
        'target_mean_shift': np.mean(shifts),
        'target_std_shift': np.std(shifts),
        'random_mean_shift': np.mean(random_shifts),
        'random_std_shift': np.std(random_shifts),
        'improvement': np.mean(shifts) - np.mean(random_shifts),
        'fold_improvement': csv_fold_improvement if csv_fold_improvement else None,
    }])
    summary.to_csv(output_dir / f"{model_name}_summary.csv", index=False)
    
    # Calculate summary statistics
    target_mean = np.mean(shifts)
    target_std = np.std(shifts)
    random_mean = np.mean(random_shifts)
    random_std = np.std(random_shifts)
    improvement = target_mean - random_mean
    
    # Calculate fold-change improvement if meaningful
    fold_improvement = None
    if abs(random_mean) > 1e-8:  # Use smaller threshold to catch very small shifts
        fold_improvement = abs(target_mean) / abs(random_mean)
    
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"When {', '.join(genes_to_perturb)} were overexpressed {fold_change}x:")
    print(f"  • Mean shift toward {goal_state}: {target_mean:+.6f} ± {target_std:.6f}")
    if random_mean > 0:
        print(f"  • Random controls shift: {random_mean:+.6f} ± {random_std:.6f}")
    else:
        print(f"  • Random controls shift: {random_mean:.6f} ± {random_std:.6f}")
    print()
    
    if fold_improvement and fold_improvement > 1.0:
        print(f"✓ Target genes showed {fold_improvement:.2f}x better shift toward {goal_state}")
        print(f"  compared to random controls ({', '.join(random_genes)})")
    elif improvement > 0:
        print(f"✓ Target genes shifted cells {improvement:+.6f} closer to {goal_state}")
        print(f"  compared to random controls ({', '.join(random_genes)})")
    else:
        print(f"✗ Target genes did not show improvement over random controls")
    print()
    print(f"Results saved to: {output_dir}")
    print()
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run unified perturbation experiment")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()),
                       help="Model to use")
    parser.add_argument("--data", type=Path,
                       default=CELLREPROGRAMMER_DIR / "data" / "prepared" / "fibroblast_ipsc_prepared.h5ad",
                       help="Path to prepared AnnData file")
    parser.add_argument("--genes", nargs="+", required=True,
                       help="Genes to perturb (symbols or Ensembl IDs)")
    parser.add_argument("--random", nargs="+",
                       default=["GAPDH", "ACTB", "B2M", "MT-ATP6"],
                       help="Random control genes (default: 4 genes to match typical OSKM perturbations)")
    parser.add_argument("--output", type=Path,
                       default=None,
                       help="Output directory")
    parser.add_argument("--max-cells", type=int, default=None,
                       help="Maximum number of cells to use (default: None = use all cells). "
                            "Limit cells only if you need faster testing or hit memory limits.")
    parser.add_argument("--fold-change", type=float, default=2.0,
                       help="Fold change for overexpression")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output is None:
        args.output = CELLREPROGRAMMER_DIR / "results" / "unified_perturbation" / args.model
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Check if Geneformer (needs special handling)
    if args.model == "geneformer":
        print("ERROR: Geneformer requires original InSilicoPerturber utilities.")
        print("Please use 03_reproduce_reprogramming.py or 05_test_all_oskm_combinations.py instead.")
        sys.exit(1)
    
    # Run experiment
    summary = run_generic_perturbation_experiment(
        model_name=args.model,
        data_path=args.data,
        genes_to_perturb=args.genes,
        random_genes=args.random,
        output_dir=args.output,
        max_cells=args.max_cells,
        fold_change=args.fold_change,
    )
    
    print("Experiment complete!")


if __name__ == "__main__":
    main()
