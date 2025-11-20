"""
Prepare Norman Dataset for Geneformer
=====================================

This script prepares the Norman 2019 dataset (GSE118258) for Geneformer tokenization.
The Norman dataset contains CRISPR-activated and non-activated K562 cells.

Uses helical's TranscriptomeTokenizer for compatibility.
"""

import os
import sys
from pathlib import Path

# Add helical to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import scanpy as sc
import pandas as pd
import numpy as np
import logging
import anndata as ad

# Import helical components
from helical.models.geneformer import GeneformerConfig
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
from helical.utils.downloader import Downloader
from helical.utils.mapping import map_gene_symbols_to_ensembl_ids

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

print("=" * 80)
print("Prepare Norman Dataset for Geneformer")
print("=" * 80)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths - works both locally and on remote server
script_dir = Path(__file__).resolve().parent
# Go up from experiments/ -> geneformer/ -> cellreprogrammer/
CELLREPROGRAMMER_DIR = script_dir.parent.parent

# Verify we're in the right place by checking for data directory
if not (CELLREPROGRAMMER_DIR / "data").exists():
    # Try going up one more level to find helical/
    potential_helical = CELLREPROGRAMMER_DIR.parent
    if (potential_helical / "cellreprogrammer" / "data").exists():
        CELLREPROGRAMMER_DIR = potential_helical / "cellreprogrammer"
    else:
        # Try common remote server paths
        for base_path in [
            Path("/home/ubuntu/data-at-virginia/helical"),
            Path("/lambda/nfs/data-at-virginia/helical"),
        ]:
            test_cellreprogrammer = base_path / "cellreprogrammer"
            if (test_cellreprogrammer / "data").exists():
                CELLREPROGRAMMER_DIR = test_cellreprogrammer
                break

BASE_DIR = CELLREPROGRAMMER_DIR.parent
DATA_DIR = CELLREPROGRAMMER_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"
TOKENIZED_DIR = DATA_DIR / "tokenized"

# Input/output files
INPUT_FILE = RAW_DIR / "norman_2019_adata.h5ad"
OUTPUT_PREFIX = "norman_k562"

# Ensure directories exist
os.makedirs(PREPARED_DIR, exist_ok=True)
os.makedirs(TOKENIZED_DIR, exist_ok=True)

print(f"Working directory: {CELLREPROGRAMMER_DIR}")
print(f"Input: {INPUT_FILE}")
print(f"Output: {TOKENIZED_DIR}/{OUTPUT_PREFIX}.dataset")
print()

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("=" * 80)
print("Step 1: Loading Norman dataset")
print("=" * 80)

if not INPUT_FILE.exists():
    print(f"ERROR: Input file not found: {INPUT_FILE}")
    print("Please ensure norman_2019_adata.h5ad is in the raw data directory")
    exit(1)

print(f"Loading: {INPUT_FILE}")
adata = sc.read_h5ad(INPUT_FILE)

print(f"✓ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"  Cell metadata columns: {list(adata.obs.columns)}")
print(f"  Gene metadata columns: {list(adata.var.columns)}")
print()

# =============================================================================
# STEP 2: INSPECT DATA STRUCTURE
# =============================================================================

print("=" * 80)
print("Step 2: Inspecting data structure")
print("=" * 80)

# Check for perturbation/activation information
print("\nAvailable metadata columns:")
for col in adata.obs.columns:
    unique_vals = adata.obs[col].unique()
    if len(unique_vals) <= 20:
        print(f"  {col}: {list(unique_vals)}")
    else:
        print(f"  {col}: {len(unique_vals)} unique values")

# Check for common Norman dataset columns
# Typical columns: 'perturbation', 'perturbation_type', 'target_gene', 'is_activated', etc.
perturbation_cols = [col for col in adata.obs.columns 
                     if any(x in col.lower() for x in ['pert', 'target', 'crispr', 'activated', 'condition'])]

if perturbation_cols:
    print(f"\nFound potential perturbation columns: {perturbation_cols}")
    for col in perturbation_cols:
        print(f"  {col}: {adata.obs[col].value_counts().head(10).to_dict()}")
else:
    print("\n⚠ Warning: No obvious perturbation columns found")
    print("  Will need to inspect manually or use default assumptions")

print()

# =============================================================================
# STEP 3: PREPARE GENE ANNOTATIONS
# =============================================================================

print("=" * 80)
print("Step 3: Preparing gene annotations (Ensembl IDs)")
print("=" * 80)

# Check if Ensembl IDs already exist
has_ensembl = 'ensembl_id' in adata.var.columns

if has_ensembl:
    print("✓ Ensembl IDs already present in var['ensembl_id']")
    # Verify they're valid
    valid_ensembl = adata.var['ensembl_id'].str.startswith('ENSG', na=False).sum()
    print(f"  Valid Ensembl IDs: {valid_ensembl}/{len(adata.var)}")
else:
    print("Mapping gene symbols to Ensembl IDs...")
    
    # Try to find gene symbol column
    gene_symbol_col = None
    if 'gene_symbols' in adata.var.columns:
        gene_symbol_col = 'gene_symbols'
    elif 'gene_name' in adata.var.columns:
        gene_symbol_col = 'gene_name'
    elif 'Symbol' in adata.var.columns:
        gene_symbol_col = 'Symbol'
    else:
        # Use index as gene symbols
        gene_symbol_col = 'index'
        print(f"  Using var.index as gene symbols")
    
    # Save original gene symbols before mapping
    if gene_symbol_col == 'index':
        adata.var['original_gene_symbol'] = adata.var.index.values
    else:
        adata.var['original_gene_symbol'] = adata.var[gene_symbol_col].values
    
    print(f"  Mapping {len(adata.var)} genes...")
    # map_gene_symbols_to_ensembl_ids modifies adata in place
    adata = map_gene_symbols_to_ensembl_ids(adata, gene_names=gene_symbol_col if gene_symbol_col != 'index' else None)
    
    mapped_count = (~adata.var['ensembl_id'].isna()).sum()
    print(f"  ✓ Mapped {mapped_count}/{len(adata.var)} genes to Ensembl IDs")
    
    # Filter out genes without Ensembl IDs
    if mapped_count < len(adata.var):
        print(f"  Filtering out {len(adata.var) - mapped_count} unmapped genes...")
        adata = adata[:, ~adata.var['ensembl_id'].isna()].copy()
        print(f"  ✓ Remaining: {adata.n_vars} genes")

print()

# =============================================================================
# STEP 4: PREPARE CELL METADATA
# =============================================================================

print("=" * 80)
print("Step 4: Preparing cell metadata")
print("=" * 80)

# Ensure total_counts exists (required for tokenization)
if 'n_counts' not in adata.obs.columns and 'total_counts' not in adata.obs.columns:
    print("Calculating total counts per cell...")
    if hasattr(adata.X, 'toarray'):
        adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    else:
        adata.obs['n_counts'] = adata.X.sum(axis=1)
    print("✓ Added n_counts")
else:
    if 'total_counts' in adata.obs.columns and 'n_counts' not in adata.obs.columns:
        adata.obs['n_counts'] = adata.obs['total_counts']
    print("✓ n_counts already present")

# Create perturbation/activation labels if needed
# Try to identify activated vs non-activated cells
if 'is_activated' not in adata.obs.columns:
    # Look for common patterns
    if 'perturbation' in adata.obs.columns:
        # Assume non-empty perturbation = activated
        adata.obs['is_activated'] = (~adata.obs['perturbation'].isna() & 
                                      (adata.obs['perturbation'] != '') &
                                      (adata.obs['perturbation'] != 'control')).astype(int)
        print("✓ Created is_activated from 'perturbation' column")
    elif 'perturbation_type' in adata.obs.columns:
        adata.obs['is_activated'] = (adata.obs['perturbation_type'] != 'control').astype(int)
        print("✓ Created is_activated from 'perturbation_type' column")
    else:
        # Default: mark all as activated (user can adjust)
        adata.obs['is_activated'] = 1
        print("⚠ Warning: Could not determine activation status, marking all as activated")
        print("  Please review and adjust 'is_activated' column if needed")

# Create cell_type column if missing (use K562 as default)
if 'cell_type' not in adata.obs.columns:
    adata.obs['cell_type'] = 'K562'
    print("✓ Added cell_type='K562'")

# Create perturbation_label for easier filtering
if 'perturbation_label' not in adata.obs.columns:
    if 'perturbation' in adata.obs.columns:
        adata.obs['perturbation_label'] = adata.obs['perturbation'].fillna('control')
    elif 'target_gene' in adata.obs.columns:
        adata.obs['perturbation_label'] = adata.obs['target_gene'].fillna('control')
    else:
        adata.obs['perturbation_label'] = adata.obs['is_activated'].map({1: 'activated', 0: 'control'})
    print("✓ Created perturbation_label")

print(f"\nCell state summary:")
if 'is_activated' in adata.obs.columns:
    print(f"  Activated: {(adata.obs['is_activated'] == 1).sum()} cells")
    print(f"  Non-activated: {(adata.obs['is_activated'] == 0).sum()} cells")
if 'perturbation_label' in adata.obs.columns:
    print(f"\nPerturbation labels:")
    for label, count in adata.obs['perturbation_label'].value_counts().head(10).items():
        print(f"  {label}: {count} cells")
print()

# =============================================================================
# STEP 5: QUALITY CONTROL
# =============================================================================

print("=" * 80)
print("Step 5: Quality control filtering")
print("=" * 80)

# Ensure filter_pass exists
if 'filter_pass' not in adata.obs.columns:
    adata.obs['filter_pass'] = 1

# Calculate n_genes if not present
if 'n_genes' not in adata.obs.columns:
    if hasattr(adata.X, 'toarray'):
        adata.obs['n_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten()
    else:
        adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)

# Apply QC criteria
min_genes = 200
max_genes = 5000
min_counts = 1000

# Update filter_pass based on QC criteria
adata.obs['filter_pass'] = (
    (adata.obs['n_genes'] >= min_genes) &
    (adata.obs['n_genes'] <= max_genes) &
    (adata.obs['n_counts'] >= min_counts)
).astype(int)

n_pass = adata.obs['filter_pass'].sum()
print(f"✓ Cells passing QC: {n_pass}/{adata.n_obs} ({n_pass/adata.n_obs*100:.1f}%)")
print(f"  Criteria: {min_genes}-{max_genes} genes per cell, ≥{min_counts} counts")
print()

# =============================================================================
# STEP 6: SAVE PREPARED DATA
# =============================================================================

print("=" * 80)
print("Step 6: Saving prepared data")
print("=" * 80)

output_h5ad = PREPARED_DIR / f"{OUTPUT_PREFIX}_prepared.h5ad"
adata.write_h5ad(output_h5ad, compression='gzip')
print(f"✓ Saved to: {output_h5ad}")
print()

# =============================================================================
# STEP 7: TOKENIZE FOR GENEFORMER
# =============================================================================

print("=" * 80)
print("Step 7: Tokenizing data for Geneformer")
print("=" * 80)

# Create custom attribute dictionary for tokenizer
custom_attrs = {
    "cell_type": "cell_type",
    "is_activated": "is_activated",
    "perturbation_label": "perturbation_label",
}

# Add other columns if present
for col in ["perturbation", "target_gene", "perturbation_type"]:
    if col in adata.obs.columns:
        custom_attrs[col] = col

print(f"Metadata columns to preserve: {list(custom_attrs.keys())}")
print()

# Tokenize for V2/V3 models (4096 tokens) - most common for newer models
model_name = "gf-12L-38M-i4096"  # V2 model
print(f"Tokenizing for {model_name} (V2/V3 models use 4096 tokens)")

# Initialize tokenizer with helical
print("  Initializing Geneformer tokenizer...")

# Create Geneformer config
config = GeneformerConfig(model_name=model_name)

# Get the correct paths
model_version = config.model_map[model_name]['model_version']
input_size = config.model_map[model_name]['input_size']
special_token = config.model_map[model_name]['special_token']

print(f"  Model version: {model_version}, input_size: {input_size}, special_token: {special_token}")

# Download tokenizer files if needed
print("  Downloading tokenizer files if needed...")
downloader = Downloader()
tokenizer_files = [
    file for file in config.list_of_files_to_download 
    if 'gene_median' in file or 'token_dictionary' in file or 'ensembl_mapping' in file
]
for file in tokenizer_files:
    downloader.download_via_name(file)
print("  ✓ Tokenizer files ready")

# Use tokenizer
tk = TranscriptomeTokenizer(
    custom_attr_name_dict=custom_attrs if custom_attrs else None,
    nproc=16,
    model_input_size=input_size,
    special_token=special_token,
    gene_median_file=config.files_config['gene_median_path'],
    token_dictionary_file=config.files_config['token_path'],
    gene_mapping_file=config.files_config['ensembl_dict_path'],
)

# Tokenize the data
print("  Tokenizing cells...")
tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)

print("  Creating dataset...")
tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata, use_generator=False)

# Save tokenized dataset
tokenized_path = TOKENIZED_DIR / f"{OUTPUT_PREFIX}.dataset"
tokenized_dataset.save_to_disk(str(tokenized_path))
print(f"  ✓ Tokenized data saved to: {tokenized_path}")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("DATA PREPARATION COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  • Total cells: {adata.n_obs}")
print(f"  • Genes: {adata.n_vars}")
print(f"  • Cells passing QC: {n_pass}")
print(f"  • Prepared file: {output_h5ad}")
print(f"  • Tokenized file: {tokenized_path}")
print()
print("Next steps:")
print(f"  1. Verify the tokenized dataset exists")
print(f"  2. Run: python test_norman_dataset.py")
print()
print("=" * 80)

