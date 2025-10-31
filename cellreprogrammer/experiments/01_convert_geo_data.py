"""
Convert GSE118258 Data for Geneformer
======================================

This script converts the GSE118258 (Xing et al. 2020) dataset from raw files
to processed h5ad format ready for Geneformer experiments.

Key features:
- Handles gzipped UMI matrices and annotations
- Converts genes × cells to cells × genes format
- Maps timepoints to cell types for reprogramming experiments
- Creates sparse matrices for memory efficiency
"""

import os
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse
import gzip
from pathlib import Path

print("=" * 80)
print("Converting GSE118258 (Xing et al. 2020) for Geneformer")
print("=" * 80)
print()

# =============================================================================
# CONFIGURATION - UPDATE FOR YOUR ENVIRONMENT
# =============================================================================

# Base paths
BASE_DIR = Path("/home/ubuntu/data-at-virginia/helical")
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "prepared"

# Ensure directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input files (adjust if your files are named differently)
umi_file = RAW_DIR / "GSE118258_UMI.csv.gz"
annotation_file = RAW_DIR / "GSE118258_Annotation.txt.gz"

# Output file
output_file = OUTPUT_DIR / "GSE118258_converted.h5ad"

print(f"Working directory: {BASE_DIR}")
print(f"Input: {RAW_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

# =============================================================================
# STEP 1: LOAD ANNOTATION DATA
# =============================================================================

print("=" * 80)
print("Step 1: Loading cell annotations")
print("=" * 80)

if not annotation_file.exists():
    print(f"ERROR: Annotation file not found: {annotation_file}")
    print("Please download GSE118258_Annotation.txt.gz to this location")
    exit(1)

with gzip.open(annotation_file, 'rt') as f:
    annotation_df = pd.read_csv(f, sep='\t', index_col=0)

print(f"✓ Loaded {len(annotation_df)} cells")
print(f"  Annotation columns: {list(annotation_df.columns)}")
print()

# =============================================================================
# STEP 2: LOAD UMI MATRIX
# =============================================================================

print("=" * 80)
print("Step 2: Loading UMI matrix")
print("=" * 80)

if not umi_file.exists():
    print(f"ERROR: UMI file not found: {umi_file}")
    print("Please download GSE118258_UMI.csv.gz to this location")
    exit(1)

print("Loading UMI matrix (genes × cells)...")
print("  This is a large file, please wait...")

with gzip.open(umi_file, 'rt') as f:
    # Load as genes × cells
    umi_df = pd.read_csv(f, index_col=0, low_memory=False)

print(f"✓ Loaded: {umi_df.shape[0]} genes × {umi_df.shape[1]} cells")

# Transpose to cells × genes
print("\nTransposing to cells × genes...")
umi_df = umi_df.T
print(f"✓ Shape after transpose: {umi_df.shape[0]} cells × {umi_df.shape[1]} genes")
print()

# =============================================================================
# STEP 3: ALIGN WITH ANNOTATIONS
# =============================================================================

print("=" * 80)
print("Step 3: Aligning with cell annotations")
print("=" * 80)

# Filter to cells that have annotations
umi_df = umi_df.loc[annotation_df.index]
print(f"✓ Filtered to {len(umi_df)} cells with annotations")
print()

# =============================================================================
# STEP 4: CREATE ANNDATA OBJECT
# =============================================================================

print("=" * 80)
print("Step 4: Creating AnnData object (sparse matrix)")
print("=" * 80)

# Convert to sparse format for memory efficiency
sparse_X = scipy.sparse.csr_matrix(umi_df.values.astype(np.float32))
adata = sc.AnnData(X=sparse_X)

# Add cell annotations
adata.obs = annotation_df.loc[umi_df.index].copy()

# Add gene information
adata.var = pd.DataFrame(index=umi_df.columns)
adata.var['gene_name'] = umi_df.columns.values
adata.var['gene_symbol'] = umi_df.columns.values
adata.var['ensembl_id'] = umi_df.columns.values  # These are already Ensembl IDs!

print(f"✓ Created AnnData with {adata.n_obs} cells × {adata.n_vars} genes")
print(f"  Cell annotations: {list(adata.obs.columns)}")
print()

# =============================================================================
# STEP 5: ADD TOTAL COUNTS AND METADATA
# =============================================================================

print("=" * 80)
print("Step 5: Adding metadata")
print("=" * 80)

# Add total counts per cell
print("Calculating total counts per cell...")
adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
print(f"  Mean counts: {adata.obs['n_counts'].mean():.0f}")
print(f"  Median counts: {adata.obs['n_counts'].median():.0f}")
print()

# =============================================================================
# STEP 6: MAP TIMEPOINTS TO CELL TYPES
# =============================================================================

print("=" * 80)
print("Step 6: Mapping timepoints to cell types")
print("=" * 80)

cell_type_mapping = {
    'D0': 'Fibroblast',
    'D2': 'Early_reprogramming',
    'D8': 'Mid_reprogramming', 
    'D12': 'Late_reprogramming',
    'D16_negative': 'Failed_reprogramming',
    'D16_positive': 'iPSC'
}

adata.obs['cell_type'] = adata.obs['Time-point'].map(cell_type_mapping).fillna(adata.obs['Time-point'])

print("Time-point -> Cell type mapping:")
for tp, ct in cell_type_mapping.items():
    count = (adata.obs['Time-point'] == tp).sum()
    if count > 0:
        print(f"  {tp} -> {ct}: {count} cells")

print(f"\n✓ Cell types:")
for cell_type, count in adata.obs['cell_type'].value_counts().items():
    print(f"  {cell_type}: {count} cells")
print()

# Add filter_pass column (all cells pass by default)
adata.obs['filter_pass'] = 1

# =============================================================================
# STEP 7: SAVE OUTPUT
# =============================================================================

print("=" * 80)
print("Step 7: Saving converted data")
print("=" * 80)

print(f"Saving to: {output_file} (with compression)...")
adata.write_h5ad(output_file, compression='gzip')
print("✓ Saved successfully!")

# Check file size
size_mb = os.path.getsize(output_file) / (1024**2)
print(f"  File size: {size_mb:.2f} MB")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("CONVERSION COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  • Input cells: {len(annotation_df)}")
print(f"  • Output cells: {adata.n_obs}")
print(f"  • Genes: {adata.n_vars}")
print(f"  • File: {output_file}")
print()
print("Cell type distribution:")
for cell_type, count in adata.obs['cell_type'].value_counts().items():
    print(f"  {cell_type}: {count} cells")
print()
print("Next steps:")
print("  1. Run: python 02_prepare_reprogramming_data.py")
print(f"     to tokenize this data for Geneformer")
print()

