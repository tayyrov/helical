"""
Run scGPT perturbation experiments.

This module provides a standalone function for running scGPT-specific
perturbation experiments using the adapter framework.
"""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import anndata as ad

from helical.models.scgpt import scGPT, scGPTConfig
from cellreprogrammer.src.adapters import scGPTAdapter
from helical.utils.downloader import Downloader
from cellreprogrammer.src.utils import calculate_fold_improvement, format_perturbation_results


def get_device():
    """Detect and return the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
    return device


def load_data(data_path: Path) -> ad.AnnData:
    """Load AnnData from prepared data."""
    return ad.read_h5ad(data_path)


def run_perturbation_experiment(
    model_name: str,
    data_path: Path,
    genes_to_perturb: List[str],
    random_genes: List[str],
    output_dir: Path,
    start_state: str = "Fibroblast",
    goal_state: str = "iPSC",
    max_cells: Optional[int] = None,  # None = use all cells (recommended for accuracy)
    fold_change: float = 2.0,
) -> pd.DataFrame:
    """
    Run scGPT perturbation experiment using the adapter framework.
    
    Parameters
    ----------
    model_name : str
        Model name (should be "scgpt")
    data_path : Path
        Path to prepared AnnData file
    genes_to_perturb : List[str]
        Gene symbols to perturb
    random_genes : List[str]
        Random control gene symbols
    output_dir : Path
        Output directory for results
    start_state : str
        Starting cell state (default: "Fibroblast")
    goal_state : str
        Goal cell state (default: "iPSC")
    max_cells : Optional[int]
        Maximum number of cells to use (None = use all cells, recommended for accuracy)
    fold_change : float
        Fold change for overexpression (default: 2.0)
        
    Returns
    -------
    pd.DataFrame
        Summary of perturbation results
    """
    print("=" * 80)
    print(f"scGPT Perturbation Experiment")
    print("=" * 80)
    print()
    
    # Auto-detect device
    device = get_device()
    
    # Create config and model
    print(f"Initializing scGPT...")
    config = scGPTConfig(device=device)
    
    # Download model files if needed
    if hasattr(config, 'list_of_files_to_download'):
        print("Downloading model files...")
        downloader = Downloader()
        for file in config.list_of_files_to_download:
            downloader.download_via_name(file)
    
    model = scGPT(config)
    adapter = scGPTAdapter(model, config)
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
    
    # Calculate summary statistics
    target_mean = np.mean(shifts)
    target_std = np.std(shifts)
    random_mean = np.mean(random_shifts)
    random_std = np.std(random_shifts)
    improvement = target_mean - random_mean
    fold_improvement = calculate_fold_improvement(target_mean, random_mean)
    
    summary = pd.DataFrame([{
        'model': model_name,
        'target_genes': ', '.join(genes_to_perturb),
        'random_genes': ', '.join(random_genes),
        'fold_change': fold_change,
        'target_mean_shift': target_mean,
        'target_std_shift': target_std,
        'random_mean_shift': random_mean,
        'random_std_shift': random_std,
        'improvement': improvement,
        'fold_improvement': fold_improvement if fold_improvement else None,
    }])
    summary.to_csv(output_dir / f"{model_name}_summary.csv", index=False)
    
    # Format and print results
    format_perturbation_results(
        target_mean=target_mean,
        target_std=target_std,
        random_mean=random_mean,
        random_std=random_std,
        target_genes=genes_to_perturb,
        random_genes=random_genes,
        goal_state=goal_state,
        fold_change=fold_change,
        perturbation_type="overexpressed",
    )
    print(f"Results saved to: {output_dir}")
    print()
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run scGPT perturbation experiment")
    parser.add_argument("--data", type=Path, required=True,
                       help="Path to prepared AnnData file")
    parser.add_argument("--genes", nargs="+", required=True,
                       help="Genes to perturb (gene symbols)")
    parser.add_argument("--random", nargs="+",
                       default=["GAPDH", "ACTB", "B2M", "MT-ATP6"],
                       help="Random control genes (default: GAPDH, ACTB, B2M, MT-ATP6)")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("--start-state", default="Fibroblast",
                       help="Starting cell state (default: Fibroblast)")
    parser.add_argument("--goal-state", default="iPSC",
                       help="Goal cell state (default: iPSC)")
    parser.add_argument("--max-cells", type=int, default=None,
                       help="Maximum number of cells to use (default: None = all available)")
    parser.add_argument("--fold-change", type=float, default=2.0,
                       help="Fold change for overexpression (default: 2.0)")
    
    args = parser.parse_args()
    
    run_perturbation_experiment(
        model_name="scgpt",
        data_path=args.data,
        genes_to_perturb=args.genes,
        random_genes=args.random,
        output_dir=args.output,
        start_state=args.start_state,
        goal_state=args.goal_state,
        max_cells=args.max_cells,
        fold_change=args.fold_change,
    )
