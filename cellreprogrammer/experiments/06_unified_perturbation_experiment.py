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
from helical.utils.mapping import convert_list_gene_symbols_to_ensembl_ids, convert_list_ensembl_ids_to_gene_symbols
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


def convert_genes_to_ensembl_ids(genes: List[str]) -> List[str]:
    """
    Convert gene symbols to Ensembl IDs if needed.
    Handles mixed inputs (some symbols, some Ensembl IDs).
    
    Parameters
    ----------
    genes : List[str]
        List of gene identifiers (can be symbols or Ensembl IDs)
        
    Returns
    -------
    List[str]
        List of Ensembl IDs (converted if needed)
    """
    # Check if already all Ensembl IDs
    if all(g.startswith("ENSG") for g in genes):
        return genes
    
    # Separate Ensembl IDs and symbols
    ensembl_ids = []
    symbols_to_convert = []
    indices_to_convert = []
    
    for i, gene in enumerate(genes):
        if gene.startswith("ENSG"):
            ensembl_ids.append(gene)
        else:
            ensembl_ids.append(None)  # Placeholder
            symbols_to_convert.append(gene)
            indices_to_convert.append(i)
    
    # Convert only the symbols
    if symbols_to_convert:
        print(f"Converting {len(symbols_to_convert)} gene symbol(s) to Ensembl IDs...")
        converted_ids = convert_list_gene_symbols_to_ensembl_ids(symbols_to_convert)
        
        # Fill in the converted IDs
        for idx, eid in zip(indices_to_convert, converted_ids):
            ensembl_ids[idx] = eid
        
        # Check for failures
        failed = [g for g, eid in zip(symbols_to_convert, converted_ids) if eid is None]
        if failed:
            raise ValueError(
                f"Could not map the following genes to Ensembl IDs: {failed}\n"
                f"Please check spelling or provide Ensembl IDs directly (e.g., ENSG00000204531)"
            )
        
        print(f"✓ Converted {len(symbols_to_convert)} gene symbol(s) to Ensembl IDs")
    
    return ensembl_ids


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


def run_geneformer_perturbation_experiment(
    model_name: str,
    data_path: Path,
    genes_to_perturb: List[str],
    random_genes: List[str],
    output_dir: Path,
    start_state: str = "Fibroblast",
    goal_state: str = "iPSC",
    max_cells: Optional[int] = None,
    fold_change: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run Geneformer perturbation experiment using original InSilicoPerturber utilities.
    
    Note: Geneformer doesn't support fold_change parameter - perturbation is done
    by moving genes to the front of tokenized sequences.
    """
    print("=" * 80)
    print(f"Unified Perturbation Experiment: {model_name.upper()}")
    print("=" * 80)
    print()
    
    # Warn about fold_change if provided
    if fold_change is not None:
        print("⚠ WARNING: Geneformer doesn't support fold_change parameter.")
        print("  Geneformer perturbs genes by moving them to the front of tokenized sequences.")
        print("  The --fold-change parameter will be ignored.")
        print()
    
    # Get model config
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    registry = MODEL_REGISTRY[model_name]
    config_class = registry["config_class"]
    model_class = registry["model_class"]
    adapter_class = registry["adapter_class"]
    default_config = registry["default_config"].copy()
    
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
    
    # Determine which tokenized dataset to use
    DATA_DIR = data_path.parent.parent  # Go up from prepared/ to data/
    if hasattr(config, 'model_map'):
        raw_version = config.model_map[config.config["model_name"]]["model_version"].upper()
        if raw_version == "V1":
            input_data_path = DATA_DIR / "tokenized" / "fibroblast_ipsc_v1.dataset"
        else:
            input_data_path = DATA_DIR / "tokenized" / "fibroblast_ipsc.dataset"
    else:
        # Default to V2/V3 dataset
        input_data_path = DATA_DIR / "tokenized" / "fibroblast_ipsc.dataset"
    
    if not input_data_path.exists():
        raise FileNotFoundError(
            f"Tokenized dataset not found: {input_data_path}\n"
            f"Please run 02_prepare_reprogramming_data.py first."
        )
    
    print(f"Using tokenized dataset: {input_data_path}")
    model_path = config.files_config["model_files_dir"]
    print(f"Model path: {model_path}")
    print()
    
    # Set up cell states
    cell_states = {
        "state_key": "cell_type",
        "start_state": start_state,
        "goal_state": goal_state,
        "alt_states": []
    }
    
    filter_data = {"cell_type": [start_state, goal_state]}
    
    # Store original gene names for display
    original_target_genes = genes_to_perturb.copy()
    original_random_genes = random_genes.copy()
    
    # Convert gene symbols to Ensembl IDs (Geneformer requires Ensembl IDs)
    genes_to_perturb = convert_genes_to_ensembl_ids(genes_to_perturb)
    random_genes = convert_genes_to_ensembl_ids(random_genes)
    print()
    
    max_ncells = max_cells if max_cells else 500  # Geneformer default
    
    # Extract state embeddings first (if not already done)
    print("=" * 80)
    print("Extracting goal state embeddings")
    print("=" * 80)
    
    try:
        from geneformer import EmbExtractor
    except ImportError:
        GENEFORMER_REPO = Path("/home/ubuntu/data-at-virginia/Geneformer")
        if GENEFORMER_REPO.exists():
            import sys
            sys.path.insert(0, str(GENEFORMER_REPO))
            from geneformer import EmbExtractor
        else:
            raise ImportError("Original Geneformer package required. Install: pip install -e /path/to/Geneformer/")
    
    os.makedirs(output_dir, exist_ok=True)
    
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        filter_data=filter_data,
        max_ncells=max_ncells,
        emb_layer=-1,
        summary_stat="exact_mean",
        forward_batch_size=50,
        model_version=adapter.model_version,
        nproc=1
    )
    
    state_embs_dict = embex.get_state_embs(
        cell_states,
        str(model_path),
        str(input_data_path),
        str(output_dir),
        "state_embs"
    )
    print("✓ State embeddings extracted")
    print()
    
    # Run target genes perturbation
    print("=" * 80)
    print(f"Testing target genes: {', '.join(genes_to_perturb)}")
    print("=" * 80)
    
    target_output_dir = output_dir / "target"
    os.makedirs(target_output_dir, exist_ok=True)
    
    adapter.run_perturbation_experiment(
        model_path=str(model_path),
        input_data_path=str(input_data_path),
        output_dir=str(target_output_dir),
        genes_to_perturb=genes_to_perturb,
        cell_states=cell_states,
        filter_data={"cell_type": [start_state]},
        max_ncells=max_ncells,
        forward_batch_size=50,
        nproc=1,
        state_embs_dict=state_embs_dict,
    )
    
    # Generate stats for target genes
    try:
        from geneformer import InSilicoPerturberStats
    except ImportError:
        GENEFORMER_REPO = Path("/home/ubuntu/data-at-virginia/Geneformer")
        if GENEFORMER_REPO.exists():
            import sys
            sys.path.insert(0, str(GENEFORMER_REPO))
            from geneformer import InSilicoPerturberStats
        else:
            raise ImportError("Original Geneformer package required.")
    
    ispstats_target = InSilicoPerturberStats(
        mode="goal_state_shift",
        genes_perturbed=genes_to_perturb,
        combos=0,
        anchor_gene=None,
        cell_states_to_model=cell_states,
        model_version=adapter.model_version
    )
    
    ispstats_target.get_stats(
        str(target_output_dir),
        None,
        str(target_output_dir),
        "perturbation_stats"
    )
    
    print("✓ Target perturbation complete")
    print()
    
    # Run random control genes perturbation
    print("=" * 80)
    print(f"Testing random control genes: {', '.join(random_genes)}")
    print("=" * 80)
    
    random_output_dir = output_dir / "random"
    os.makedirs(random_output_dir, exist_ok=True)
    
    adapter.run_perturbation_experiment(
        model_path=str(model_path),
        input_data_path=str(input_data_path),
        output_dir=str(random_output_dir),
        genes_to_perturb=random_genes,
        cell_states=cell_states,
        filter_data={"cell_type": [start_state]},
        max_ncells=max_ncells,
        forward_batch_size=50,
        nproc=1,
        state_embs_dict=state_embs_dict,
    )
    
    # Generate stats for random genes
    ispstats_random = InSilicoPerturberStats(
        mode="goal_state_shift",
        genes_perturbed=random_genes,
        combos=0,
        anchor_gene=None,
        cell_states_to_model=cell_states,
        model_version=adapter.model_version
    )
    
    ispstats_random.get_stats(
        str(random_output_dir),
        None,
        str(random_output_dir),
        "perturbation_stats"
    )
    
    print("✓ Random control perturbation complete")
    print()
    
    # Read and compare results
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    target_stats_file = target_output_dir / "perturbation_stats.csv"
    random_stats_file = random_output_dir / "perturbation_stats.csv"
    
    if not target_stats_file.exists():
        raise FileNotFoundError(f"Target stats file not found: {target_stats_file}")
    if not random_stats_file.exists():
        raise FileNotFoundError(f"Random stats file not found: {random_stats_file}")
    
    target_df = pd.read_csv(target_stats_file)
    random_df = pd.read_csv(random_stats_file)
    
    if 'Shift_to_goal_end' not in target_df.columns or 'Shift_to_goal_end' not in random_df.columns:
        raise ValueError("Shift_to_goal_end column not found in stats files")
    
    target_mean = target_df['Shift_to_goal_end'].mean()
    target_std = target_df['Shift_to_goal_end'].std()
    random_mean = random_df['Shift_to_goal_end'].mean()
    random_std = random_df['Shift_to_goal_end'].std()
    improvement = target_mean - random_mean
    
    # Calculate fold improvement
    fold_improvement = None
    if abs(random_mean) > 1e-8:
        fold_improvement = abs(target_mean) / abs(random_mean)
    
    # Format results (same as generic perturbation)
    # Use original gene names for display (more readable)
    print(f"When {', '.join(original_target_genes)} were overexpressed (Geneformer perturbation):")
    print(f"  • Mean shift toward {goal_state}: {target_mean:+.6f} ± {target_std:.6f}")
    if random_mean > 0:
        print(f"  • Random controls shift: {random_mean:+.6f} ± {random_std:.6f}")
    else:
        print(f"  • Random controls shift: {random_mean:.6f} ± {random_std:.6f}")
    print()
    
    if fold_improvement and fold_improvement > 1.0:
        print(f"✓ Target genes showed {fold_improvement:.2f}x better shift toward {goal_state}")
        print(f"  compared to random controls ({', '.join(original_random_genes)})")
    elif improvement > 0:
        print(f"✓ Target genes shifted cells {improvement:+.6f} closer to {goal_state}")
        print(f"  compared to random controls ({', '.join(original_random_genes)})")
    else:
        print(f"✗ Target genes did not show improvement over random controls")
    print()
    
    # Save summary CSV (include both original names and Ensembl IDs)
    summary = pd.DataFrame([{
        'model': model_name,
        'target_genes': ', '.join(original_target_genes),
        'target_genes_ensembl': ', '.join(genes_to_perturb),
        'random_genes': ', '.join(original_random_genes),
        'random_genes_ensembl': ', '.join(random_genes),
        'fold_change': None,  # Geneformer doesn't use fold_change
        'target_mean_shift': target_mean,
        'target_std_shift': target_std,
        'random_mean_shift': random_mean,
        'random_std_shift': random_std,
        'improvement': improvement,
        'fold_improvement': fold_improvement if fold_improvement else None,
    }])
    summary.to_csv(output_dir / f"{model_name}_summary.csv", index=False)
    
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
                       help="Genes to perturb as gene symbols (e.g., OCT4, SOX2) or Ensembl IDs. "
                            "The script automatically converts symbols to Ensembl IDs when needed.")
    parser.add_argument("--random", nargs="+",
                       default=["GAPDH", "ACTB", "B2M", "MT-ATP6"],  # Will be converted to Ensembl IDs automatically
                       help="Random control genes as gene symbols or Ensembl IDs (default: 4 genes to match typical OSKM perturbations)")
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
    
    # Run experiment (Geneformer handled separately)
    if args.model == "geneformer":
        summary = run_geneformer_perturbation_experiment(
            model_name=args.model,
            data_path=args.data,
            genes_to_perturb=args.genes,
            random_genes=args.random,
            output_dir=args.output,
            max_cells=args.max_cells,
            fold_change=args.fold_change,  # Warn if provided
        )
    else:
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
