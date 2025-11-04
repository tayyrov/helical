"""
Unified Perturbation Experiment Runner

Runs in-silico perturbation experiments across multiple models
using model-specific scripts for each model.

Usage:
    python 06_unified_perturbation_experiment.py --model scgpt --genes OCT4 SOX2 KLF4 MYC
    python 06_unified_perturbation_experiment.py --model geneformer --genes OCT4 SOX2 KLF4 MYC
"""

import argparse
import sys
from pathlib import Path

# Path setup
script_dir = Path(__file__).resolve().parent
BASE_DIR = script_dir.parent.parent
if not BASE_DIR.exists():
    BASE_DIR = Path("/home/ubuntu/data-at-virginia/helical")
CELLREPROGRAMMER_DIR = BASE_DIR / "cellreprogrammer"

# Add to path
sys.path.insert(0, str(CELLREPROGRAMMER_DIR))

# Import model-specific perturbation functions
from geneformer.run_perturbation import run_perturbation_experiment as run_geneformer_perturbation
from scgpt.run_perturbation import run_perturbation_experiment as run_scgpt_perturbation

# Model routing map
MODEL_ROUTER = {
    "geneformer": run_geneformer_perturbation,
    "scgpt": run_scgpt_perturbation,
}


def main():
    """
    Main entry point for unified perturbation experiments.
    
    Routes to model-specific perturbation scripts based on the --model argument.
    """
    parser = argparse.ArgumentParser(description="Run unified perturbation experiment")
    parser.add_argument("--model", required=True, choices=list(MODEL_ROUTER.keys()),
                       help="Model to use")
    parser.add_argument("--data", type=Path,
                       default=CELLREPROGRAMMER_DIR / "data" / "prepared" / "fibroblast_ipsc_prepared.h5ad",
                       help="Path to prepared AnnData file")
    parser.add_argument("--genes", nargs="+", required=True,
                       help="Genes to perturb as gene symbols (e.g., OCT4, SOX2) or Ensembl IDs. "
                            "The script automatically converts symbols to Ensembl IDs when needed.")
    parser.add_argument("--random", nargs="+",
                       default=["GAPDH", "ACTB", "B2M", "MT-ATP6"],
                       help="Random control genes as gene symbols or Ensembl IDs (default: 4 genes to match typical OSKM perturbations)")
    parser.add_argument("--output", type=Path,
                       default=None,
                       help="Output directory")
    parser.add_argument("--max-cells", type=int, default=None,
                       help="Maximum number of cells to use (default: None = use all cells for accuracy). "
                            "Limit cells only if you need faster testing or hit memory limits.")
    parser.add_argument("--fold-change", type=float, default=2.0,
                       help="Fold change for overexpression (not used by Geneformer)")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output is None:
        args.output = CELLREPROGRAMMER_DIR / "results" / "unified_perturbation" / args.model
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Route to model-specific perturbation function
    if args.model not in MODEL_ROUTER:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_ROUTER.keys())}")
    
    run_function = MODEL_ROUTER[args.model]
    
    # Prepare arguments for model-specific function
    kwargs = {
        "model_name": args.model if args.model == "geneformer" else "scgpt",
        "data_path": args.data,
        "genes_to_perturb": args.genes,
        "random_genes": args.random,
        "output_dir": args.output,
        "max_cells": args.max_cells,
    }
    
    # Only pass fold_change for models that support it
    if args.model != "geneformer":
        kwargs["fold_change"] = args.fold_change
    
    # Call model-specific function
    summary = run_function(**kwargs)
    
    print("Experiment complete!")


if __name__ == "__main__":
    main()
