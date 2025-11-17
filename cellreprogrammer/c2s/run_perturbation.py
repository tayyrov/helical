"""
Run Cell2Sen (C2S) perturbation experiments.

This module provides a standalone function for running Cell2Sen-specific
perturbation experiments using the adapter framework.

Cell2Sen uses a generative, text-based approach where perturbations are
described in natural language and the model generates new cell sentences.
"""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import anndata as ad

from helical.models.c2s import Cell2Sen, Cell2SenConfig

# Import from src (cellreprogrammer directory should be in sys.path)
import sys
from pathlib import Path
# Ensure cellreprogrammer directory is in sys.path
cellreprogrammer_dir = Path(__file__).resolve().parent.parent
if str(cellreprogrammer_dir) not in sys.path:
    sys.path.insert(0, str(cellreprogrammer_dir))

from src.adapters import Cell2SenAdapter
from src.utils import calculate_fold_improvement, format_perturbation_results


def get_device():
    """Detect and return the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠ No GPU detected, using CPU (Cell2Sen will be slower)")
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
    max_cells: Optional[int] = None,
    perturbation_type: str = "overexpress",
    fold_change: Optional[float] = None,
    model_size: str = "2B",
    use_quantization: bool = False,
    max_genes_per_cell: int = 5000,
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Run Cell2Sen perturbation experiment using the adapter framework.
    
    Parameters
    ----------
    model_name : str
        Model name (should be "c2s")
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
        Maximum number of cells to use (None = use all cells)
    perturbation_type : str
        "overexpress" or "knockdown" (default: "overexpress")
    fold_change : Optional[float]
        Fold change for display (Cell2Sen doesn't use this directly, but includes it in text)
    model_size : str
        Model size: "2B" or "27B" (default: "2B")
    use_quantization : bool
        Whether to use 4-bit quantization (default: False)
    max_genes_per_cell : int
        Maximum number of genes to use per cell (default: 5000)
        Lower values (e.g., 2000-3000) use less memory but may lose information
        This is critical for avoiding OOM errors with long cell sentences
    batch_size : int
        Batch size for processing (default: 1)
        Processing one cell at a time avoids OOM. Increase only if you have sufficient GPU memory.
        
    Returns
    -------
    pd.DataFrame
        Summary of perturbation results
    """
    print("=" * 80)
    print(f"Cell2Sen (C2S) Perturbation Experiment")
    print("=" * 80)
    print()
    print(f"Model: Cell2Sen-{model_size}")
    print(f"Perturbation type: {perturbation_type}")
    if fold_change:
        print(f"Fold change: {fold_change}x")
    print()
    
    # Auto-detect device
    device = get_device()
    
    # Create config and model
    # Use small batch_size to avoid OOM - Cell2Sen processes long sequences (cell sentences)
    # Even with 1000 genes, sequences are 4000+ tokens, so batch processing causes OOM
    # Default to batch_size=1, but allow user to override via parameter
    print(f"Initializing Cell2Sen-{model_size}...")
    print(f"  Using batch_size={batch_size} (processing {batch_size} cell(s) at a time)")
    if batch_size > 1:
        print(f"  ⚠ Warning: batch_size > 1 may cause OOM with long sequences")
    if use_quantization:
        print("  Using 4-bit quantization to reduce memory")
    config = Cell2SenConfig(
        batch_size=batch_size,
        model_size=model_size,
        use_quantization=use_quantization,
    )
    
    model = Cell2Sen(config)
    adapter = Cell2SenAdapter(model, config)
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
    
    # Limit cells if requested
    if max_cells and adata.n_obs > max_cells:
        import scanpy as sc
        print(f"⚠ Limiting to {max_cells} cells")
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
        print(f"✓ Subsampled to {max_cells} cells")
    elif max_cells is None:
        print(f"✓ Using all {adata.n_obs} cells")
    print()
    
    # Extract goal state embeddings (for comparison)
    print(f"Extracting goal state embeddings ({goal_state})...")
    goal_adata = load_data(data_path)
    goal_adata = goal_adata[goal_adata.obs['cell_type'] == goal_state].copy()
    if max_cells and goal_adata.n_obs > max_cells:
        import scanpy as sc
        sc.pp.subsample(goal_adata, n_obs=max_cells, random_state=42)
    
    goal_dataset = adapter.process_data(goal_adata, max_genes_per_cell=max_genes_per_cell)
    goal_embeddings = adapter.extract_embeddings(goal_dataset)
    goal_centroid = np.mean(goal_embeddings, axis=0)
    print(f"✓ Goal state embeddings: {goal_embeddings.shape}")
    print()
    
    # Process baseline data
    print("Processing baseline data...")
    baseline_dataset = adapter.process_data(adata, max_genes_per_cell=max_genes_per_cell)
    baseline_embeddings = adapter.extract_embeddings(baseline_dataset)
    print(f"✓ Baseline embeddings: {baseline_embeddings.shape}")
    print()
    
    # Test target genes
    print(f"Generating perturbed cell sentences for: {', '.join(genes_to_perturb)}")
    print("  (This uses Cell2Sen's generative LLM to predict perturbed states)")
    perturbed_dataset = adapter.apply_perturbation(
        baseline_dataset,
        genes_to_perturb,
        perturbation_type=perturbation_type,
        fold_change=fold_change
    )
    
    # Extract embeddings from perturbed sentences
    print("Extracting embeddings from perturbed sentences...")
    perturbed_embeddings = adapter.extract_perturbed_embeddings(perturbed_dataset)
    
    # Filter to valid embeddings (non-NaN)
    valid_mask = ~np.isnan(perturbed_embeddings).any(axis=1)
    if valid_mask.sum() == 0:
        raise ValueError("No valid perturbed embeddings generated")
    
    valid_perturbed_embeddings = perturbed_embeddings[valid_mask]
    valid_baseline_embeddings = baseline_embeddings[valid_mask]
    valid_goal_centroid = goal_centroid  # Single vector, no need to filter
    
    print(f"✓ Valid perturbed embeddings: {valid_perturbed_embeddings.shape[0]} / {len(perturbed_embeddings)}")
    
    # Calculate shifts
    shifts = adapter.compute_shift(
        valid_baseline_embeddings,
        valid_perturbed_embeddings,
        valid_goal_centroid,
        metric="cosine"
    )
    print(f"✓ Mean shift: {np.mean(shifts):.6f} ± {np.std(shifts):.6f}")
    print()
    
    # Test random genes
    print(f"Generating perturbed cell sentences for random control: {', '.join(random_genes)}")
    random_dataset = adapter.apply_perturbation(
        baseline_dataset,
        random_genes,
        perturbation_type=perturbation_type,
        fold_change=fold_change
    )
    
    print("Extracting embeddings from random control perturbed sentences...")
    random_embeddings = adapter.extract_perturbed_embeddings(random_dataset)
    
    # Filter to valid embeddings
    valid_random_mask = ~np.isnan(random_embeddings).any(axis=1)
    if valid_random_mask.sum() == 0:
        raise ValueError("No valid random perturbed embeddings generated")
    
    valid_random_embeddings = random_embeddings[valid_random_mask]
    valid_random_baseline_embeddings = baseline_embeddings[valid_random_mask]
    
    random_shifts = adapter.compute_shift(
        valid_random_baseline_embeddings,
        valid_random_embeddings,
        valid_goal_centroid,
        metric="cosine"
    )
    print(f"✓ Random mean shift: {np.mean(random_shifts):.6f} ± {np.std(random_shifts):.6f}")
    print()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Create full results array (with NaN for invalid cells)
    full_shifts = np.full(len(baseline_embeddings), np.nan)
    full_shifts[valid_mask] = shifts
    
    full_random_shifts = np.full(len(baseline_embeddings), np.nan)
    full_random_shifts[valid_random_mask] = random_shifts
    
    results = pd.DataFrame({
        'cell_idx': range(len(full_shifts)),
        'target_shift': full_shifts,
        'random_shift': full_random_shifts,
        'valid_target': valid_mask,
        'valid_random': valid_random_mask,
    })
    results.to_csv(output_dir / f"{model_name}_perturbation_results.csv", index=False)
    
    # Calculate summary statistics (only on valid cells)
    target_mean = np.mean(shifts)
    target_std = np.std(shifts)
    random_mean = np.mean(random_shifts)
    random_std = np.std(random_shifts)
    improvement = target_mean - random_mean
    fold_improvement = calculate_fold_improvement(target_mean, random_mean)
    
    summary = pd.DataFrame([{
        'model': model_name,
        'model_size': model_size,
        'target_genes': ', '.join(genes_to_perturb),
        'random_genes': ', '.join(random_genes),
        'perturbation_type': perturbation_type,
        'fold_change': fold_change if fold_change else None,
        'n_valid_target': valid_mask.sum(),
        'n_valid_random': valid_random_mask.sum(),
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
        perturbation_type=perturbation_type,
    )
    print(f"Results saved to: {output_dir}")
    print()
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Cell2Sen perturbation experiment")
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
    parser.add_argument("--perturbation-type", default="overexpress",
                       choices=["overexpress", "knockdown"],
                       help="Type of perturbation (default: overexpress)")
    parser.add_argument("--fold-change", type=float, default=None,
                       help="Fold change for display (optional)")
    parser.add_argument("--model-size", default="2B",
                       choices=["2B", "27B"],
                       help="Model size: 2B or 27B (default: 2B)")
    parser.add_argument("--use-quantization", action="store_true",
                       help="Use 4-bit quantization (reduces memory usage, recommended for large datasets)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for processing (default: 1, increase only if you have enough GPU memory)")
    parser.add_argument("--max-genes-per-cell", type=int, default=5000,
                       help="Maximum genes per cell (default: 5000, lower values use less memory)")
    
    args = parser.parse_args()
    
    run_perturbation_experiment(
        model_name="c2s",
        data_path=args.data,
        genes_to_perturb=args.genes,
        random_genes=args.random,
        output_dir=args.output,
        start_state=args.start_state,
        goal_state=args.goal_state,
        max_cells=args.max_cells,
        perturbation_type=args.perturbation_type,
        fold_change=args.fold_change,
        model_size=args.model_size,
        use_quantization=args.use_quantization,
        max_genes_per_cell=args.max_genes_per_cell,
        batch_size=args.batch_size,
    )

