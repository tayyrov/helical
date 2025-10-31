"""
Prepare Reprogramming Data for Geneformer
==========================================

This script prepares GSE118258 converted data for Geneformer tokenization
and runs the tokenization process.

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

# Import helical's tokenizer
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

print("=" * 80)
print("Prepare Reprogramming Data for Geneformer")
print("=" * 80)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = Path("/home/ubuntu/data-at-virginia/helical")
CELLREPROGRAMMER_DIR = BASE_DIR / "cellreprogrammer"
DATA_DIR = CELLREPROGRAMMER_DIR / "data"
PREPARED_DIR = DATA_DIR / "prepared"
TOKENIZED_DIR = DATA_DIR / "tokenized"

# Input/output files
INPUT_FILE = PREPARED_DIR / "GSE118258_converted.h5ad"
OUTPUT_PREFIX = "fibroblast_ipsc"

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
print("Step 1: Loading converted data")
print("=" * 80)

if not INPUT_FILE.exists():
    print(f"ERROR: Input file not found: {INPUT_FILE}")
    print("Please run 01_convert_geo_data.py first")
    exit(1)

print(f"Loading: {INPUT_FILE}")
adata = sc.read_h5ad(INPUT_FILE)

print(f"✓ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"  Cell metadata columns: {list(adata.obs.columns)}")
print()

# =============================================================================
# STEP 2: INSPECT AND VERIFY DATA
# =============================================================================

print("=" * 80)
print("Step 2: Verifying data structure")
print("=" * 80)

# Check required fields
has_ensembl = 'ensembl_id' in adata.var.columns
has_counts = 'n_counts' in adata.obs.columns

print(f"Has Ensembl IDs: {has_ensembl}")
print(f"Has total counts: {has_counts}")

if not has_ensembl:
    print("ERROR: Data missing Ensembl IDs. Please check input file.")
    exit(1)

if not has_counts:
    print("ERROR: Data missing total counts. Please check input file.")
    exit(1)

# Display cell type summary
if 'cell_type' in adata.obs.columns:
    print("\nCell types:")
    for cell_type, count in adata.obs['cell_type'].value_counts().items():
        print(f"  {cell_type}: {count} cells")
print()

# =============================================================================
# STEP 3: ADD METADATA ATTRIBUTES
# =============================================================================

print("=" * 80)
print("Step 3: Setting up metadata attributes")
print("=" * 80)

# Create custom attribute dictionary for tokenizer
# This maps original column names to what will be stored in the tokenized dataset
custom_attrs = {
    "cell_type": "cell_type",
}

# Add other columns if present
for col in ["Time-point", "filter_pass"]:
    if col in adata.obs.columns:
        custom_attrs[col] = col

print(f"Metadata columns to preserve: {list(custom_attrs.keys())}")
print()

# =============================================================================
# STEP 4: QUALITY CONTROL
# =============================================================================

print("=" * 80)
print("Step 4: Quality control filtering")
print("=" * 80)

# Ensure filter_pass exists
if 'filter_pass' not in adata.obs.columns:
    adata.obs['filter_pass'] = 1

# Apply QC criteria if needed
min_genes = 200
max_genes = 5000

# Calculate n_genes if not present
if 'n_genes' not in adata.obs.columns:
    adata.obs['n_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten()

# Update filter_pass based on QC criteria
adata.obs['filter_pass'] = (
    (adata.obs['n_genes'] >= min_genes) &
    (adata.obs['n_genes'] <= max_genes)
).astype(int)

n_pass = adata.obs['filter_pass'].sum()
print(f"✓ Cells passing QC: {n_pass}/{adata.n_obs} ({n_pass/adata.n_obs*100:.1f}%)")
print(f"  Criteria: {min_genes}-{max_genes} genes per cell")
print()

# =============================================================================
# STEP 5: SAVE PREPARED DATA
# =============================================================================

print("=" * 80)
print("Step 5: Saving prepared data")
print("=" * 80)

output_h5ad = PREPARED_DIR / f"{OUTPUT_PREFIX}_prepared.h5ad"
adata.write_h5ad(output_h5ad, compression='gzip')
print(f"✓ Saved to: {output_h5ad}")
print()

# =============================================================================
# STEP 6: TOKENIZE FOR GENEFORMER
# =============================================================================

print("=" * 80)
print("Step 6: Tokenizing data for Geneformer")
print("=" * 80)

# Initialize tokenizer with helical
print("Initializing Geneformer tokenizer...")
print(f"  Preserving metadata: {list(custom_attrs.values())}")

# Note: Model version V2 for compatibility with newest models
tk = TranscriptomeTokenizer(
    custom_attr_name_dict=custom_attrs if custom_attrs else None,
    nproc=16,  # Adjust based on available CPUs
)

print("Tokenizing... (this may take several minutes)")
print(f"  Input: {PREPARED_DIR}")
print(f"  Output: {TOKENIZED_DIR}")

# Tokenize the specific file
tk.tokenize_data(
    str(PREPARED_DIR),  # Directory containing h5ad files
    str(TOKENIZED_DIR),  # Output directory
    OUTPUT_PREFIX,  # Output prefix
    file_format="h5ad",
    input_identifier=output_h5ad.stem  # Only tokenize this specific file
)

tokenized_path = TOKENIZED_DIR / f"{OUTPUT_PREFIX}.dataset"
print(f"✓ Tokenized data saved to: {tokenized_path}")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("DATA PREPARATION COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  • Prepared cells: {adata.n_obs}")
print(f"  • Genes: {adata.n_vars}")
print(f"  • Cells passing QC: {n_pass}")
print(f"  • Prepared file: {output_h5ad}")
print(f"  • Tokenized file: {tokenized_path}")
print()
print("Next steps:")
print("  1. Verify the tokenized dataset exists")
print(f"  2. Update INPUT_DATA_PATH in 03_reproduce_reprogramming.py")
print(f"  3. Run: python 03_reproduce_reprogramming.py")
print()
print("=" * 80)

