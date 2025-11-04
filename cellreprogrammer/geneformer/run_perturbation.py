"""
Run Geneformer perturbation experiments.

This module provides a standalone function for running Geneformer-specific
perturbation experiments using the original InSilicoPerturber utilities.
"""

import os
from pathlib import Path
from typing import List, Optional
import pandas as pd

from helical.models.geneformer import GeneformerConfig, Geneformer
from cellreprogrammer.src.adapters import GeneformerAdapter
from helical.utils.downloader import Downloader
from helical.utils.mapping import convert_list_gene_symbols_to_ensembl_ids

from cellreprogrammer.src.utils import calculate_fold_improvement, format_perturbation_results


# Standard control genes with explicit Ensembl ID mappings for consistency
STANDARD_CONTROL_GENES = {
    "GAPDH": "ENSG00000111640",
    "ACTB": "ENSG00000075624",
    "B2M": "ENSG00000204525",
    "MT-ATP6": "ENSG00000198899",
}

# Gene aliases for common gene names
GENE_ALIASES = {
    "OCT4": "POU5F1",
    "M2B": "B2M",
}


def convert_genes_to_ensembl_ids(genes: List[str]) -> List[str]:
    """
    Convert gene symbols to Ensembl IDs, handling aliases and standard controls.
    
    Parameters
    ----------
    genes : List[str]
        List of gene symbols or Ensembl IDs
        
    Returns
    -------
    List[str]
        List of Ensembl IDs
    """
    ensembl_ids = []
    for gene in genes:
        gene_upper = gene.upper()
        
        # Check if it's already an Ensembl ID
        if gene.startswith("ENSG"):
            ensembl_ids.append(gene)
            continue
        
        # Check standard control genes first (explicit mapping)
        if gene_upper in STANDARD_CONTROL_GENES:
            ensembl_ids.append(STANDARD_CONTROL_GENES[gene_upper])
            continue
        
        # Resolve aliases
        if gene_upper in GENE_ALIASES:
            gene = GENE_ALIASES[gene_upper]
        
        # Convert to Ensembl ID
        try:
            converted = convert_list_gene_symbols_to_ensembl_ids([gene])
            if converted[0] is None:
                raise ValueError(f"Could not map gene '{gene}' to Ensembl ID")
            ensembl_ids.append(converted[0])
        except Exception as e:
            raise ValueError(f"Could not map gene '{gene}' to Ensembl ID: {e}")
    
    return ensembl_ids


def format_gene_list_for_display(original_genes: List[str], ensembl_ids: List[str]) -> str:
    """
    Format gene list for display, showing both symbols and Ensembl IDs.
    
    Parameters
    ----------
    original_genes : List[str]
        Original gene symbols
    ensembl_ids : List[str]
        Corresponding Ensembl IDs
        
    Returns
    -------
    str
        Formatted string for display
    """
    parts = []
    for orig, ens_id in zip(original_genes, ensembl_ids):
        if orig.startswith("ENSG"):
            parts.append(ens_id)
        else:
            parts.append(f"{orig} ({ens_id})")
    return ", ".join(parts)


def run_perturbation_experiment(
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
    
    Parameters
    ----------
    model_name : str
        Geneformer model name (e.g., "gf-20L-151M-i4096")
    data_path : Path
        Path to prepared AnnData file (for determining data directory structure)
    genes_to_perturb : List[str]
        Gene symbols or Ensembl IDs to perturb (will be converted to Ensembl IDs)
    random_genes : List[str]
        Random control gene symbols or Ensembl IDs (will be converted to Ensembl IDs)
    output_dir : Path
        Output directory for results
    start_state : str
        Starting cell state (default: "Fibroblast")
    goal_state : str
        Goal cell state (default: "iPSC")
    max_cells : Optional[int]
        Maximum number of cells to use (None = use all available cells)
    fold_change : Optional[float]
        Not used by Geneformer (will issue warning if provided)
        
    Returns
    -------
    pd.DataFrame
        Summary of perturbation results
    """
    print("=" * 80)
    print(f"Geneformer Perturbation Experiment: {model_name.upper()}")
    print("=" * 80)
    print()
    
    # Warn about unsupported parameters
    if fold_change is not None:
        print(f"⚠ Warning: Geneformer does not support 'fold_change' parameter.")
        print(f"  Perturbation is done by moving genes to the front of tokenized sequences.")
        print()
    
    # Create config and model
    print(f"Initializing {model_name}...")
    config = GeneformerConfig(model_name=model_name, batch_size=50)
    
    # Download model files if needed
    if hasattr(config, 'list_of_files_to_download'):
        print("Downloading model files...")
        downloader = Downloader()
        for file in config.list_of_files_to_download:
            downloader.download_via_name(file)
    
    model = Geneformer(config)
    adapter = GeneformerAdapter(model, config)
    print("✓ Model initialized")
    print()
    
    # Determine which tokenized dataset to use
    DATA_DIR = data_path.parent.parent  # Go up from prepared/ to data/
    if hasattr(config, 'model_map'):
        raw_version = config.model_map[model_name]["model_version"].upper()
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
    
    # Use all cells by default for accuracy (Geneformer default is 1000, but we use None = all)
    max_ncells = max_cells  # None means use all available cells
    if max_ncells is None:
        print(f"Using all available cells (max_ncells=None)")
    else:
        print(f"Limiting to {max_ncells} cells")
    print()
    
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
    print(f"Testing target genes: {format_gene_list_for_display(original_target_genes, genes_to_perturb)}")
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
    print(f"Testing random control genes: {format_gene_list_for_display(original_random_genes, random_genes)}")
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
    fold_improvement = calculate_fold_improvement(target_mean, random_mean)
    
    # Format and print results using original gene names for display (more readable)
    format_perturbation_results(
        target_mean=target_mean,
        target_std=target_std,
        random_mean=random_mean,
        random_std=random_std,
        target_genes=original_target_genes,
        random_genes=original_random_genes,
        goal_state=goal_state,
        fold_change=None,  # Geneformer doesn't use fold_change
        perturbation_type="overexpressed",
    )
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Geneformer perturbation experiment")
    parser.add_argument("--model", required=True,
                       help="Geneformer model name (e.g., gf-20L-151M-i4096)")
    parser.add_argument("--data", type=Path, required=True,
                       help="Path to prepared AnnData file")
    parser.add_argument("--genes", nargs="+", required=True,
                       help="Genes to perturb (gene symbols or Ensembl IDs)")
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
    
    args = parser.parse_args()
    
    run_perturbation_experiment(
        model_name=args.model,
        data_path=args.data,
        genes_to_perturb=args.genes,
        random_genes=args.random,
        output_dir=args.output,
        start_state=args.start_state,
        goal_state=args.goal_state,
        max_cells=args.max_cells,
    )
