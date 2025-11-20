"""
Test Geneformer on Norman Dataset - Single and Double Perturbations
===================================================================

This script evaluates Geneformer's ability to predict perturbation effects
in the Norman et al. 2019 dataset, which contains:
- Control cells (ctrl)
- Single-gene perturbations (Gene+ctrl or ctrl+Gene)
- Double-gene perturbations (Gene1+Gene2)

We test:
1. Can Geneformer distinguish perturbed from control cells?
2. Can it predict single perturbation effects?
3. Can it predict double perturbation effects?
4. Can it detect genetic interactions (synergy)?

Based on Norman et al. 2019 Science paper and Pert-VALA evaluation approach.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Path setup
script_dir = Path(__file__).resolve().parent
CELLREPROGRAMMER_DIR = script_dir.parent.parent

# Verify we're in the right place
if not (CELLREPROGRAMMER_DIR / "data").exists():
    potential_helical = CELLREPROGRAMMER_DIR.parent
    if (potential_helical / "cellreprogrammer" / "data").exists():
        CELLREPROGRAMMER_DIR = potential_helical / "cellreprogrammer"
    else:
        for base_path in [
            Path("/home/ubuntu/data-at-virginia/helical"),
            Path("/lambda/nfs/data-at-virginia/helical"),
        ]:
            test_cellreprogrammer = base_path / "cellreprogrammer"
            if (test_cellreprogrammer / "data").exists():
                CELLREPROGRAMMER_DIR = test_cellreprogrammer
                break

BASE_DIR = CELLREPROGRAMMER_DIR.parent

# Import dataset tools
from datasets import load_from_disk
import scanpy as sc

print("=" * 80)
print("Norman Dataset Perturbation Analysis")
print("=" * 80)
print()
print("Loading Norman dataset to analyze perturbation structure...")
print()

# =============================================================================
# STEP 1: ANALYZE DATASET STRUCTURE
# =============================================================================

DATA_DIR = CELLREPROGRAMMER_DIR / "data"
INPUT_DATA_PATH = DATA_DIR / "tokenized" / "norman_k562.dataset"
PREPARED_H5AD = DATA_DIR / "prepared" / "norman_k562_prepared.h5ad"
OUTPUT_DIR = CELLREPROGRAMMER_DIR / "results" / "norman_perturbations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not PREPARED_H5AD.exists():
    print(f"ERROR: Prepared data not found: {PREPARED_H5AD}")
    print("Please run prepare_norman_data.py first")
    sys.exit(1)

print(f"Loading: {PREPARED_H5AD}")
adata = sc.read_h5ad(PREPARED_H5AD)

print(f"✓ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
print()

# =============================================================================
# STEP 2: CATEGORIZE PERTURBATIONS
# =============================================================================

print("=" * 80)
print("STEP 1: Categorizing perturbations")
print("=" * 80)
print()

# Parse guide_merged to identify single vs double perturbations
perturbation_types = defaultdict(list)

for guide in adata.obs['guide_merged'].unique():
    guide_str = str(guide)
    
    if guide_str == 'ctrl':
        perturbation_types['control'].append(guide_str)
    elif '+' in guide_str:
        parts = guide_str.split('+')
        if 'ctrl' in parts:
            # Single perturbation (Gene+ctrl or ctrl+Gene)
            perturbation_types['single'].append(guide_str)
        else:
            # Double perturbation (Gene1+Gene2)
            perturbation_types['double'].append(guide_str)
    else:
        perturbation_types['unknown'].append(guide_str)

print("Perturbation breakdown:")
print(f"  Control: {len(perturbation_types['control'])} guides")
print(f"    Example: {perturbation_types['control'][:3]}")
print()
print(f"  Single perturbations: {len(perturbation_types['single'])} guides")
print(f"    Example: {perturbation_types['single'][:5]}")
print()
print(f"  Double perturbations: {len(perturbation_types['double'])} guides")
print(f"    Example: {perturbation_types['double'][:5]}")
print()

# Count cells per category
ctrl_cells = (adata.obs['guide_merged'] == 'ctrl').sum()
single_cells = adata.obs['guide_merged'].isin(perturbation_types['single']).sum()
double_cells = adata.obs['guide_merged'].isin(perturbation_types['double']).sum()

print("Cell counts:")
print(f"  Control cells: {ctrl_cells}")
print(f"  Single perturbation cells: {single_cells}")
print(f"  Double perturbation cells: {double_cells}")
print()

# =============================================================================
# STEP 3: IDENTIFY KEY PERTURBATIONS FROM NORMAN PAPER
# =============================================================================

print("=" * 80)
print("STEP 2: Identifying key perturbations from Norman et al. 2019")
print("=" * 80)
print()

# Key interactions from the paper
KEY_INTERACTIONS = {
    # Erythroid differentiation
    "CBL+CNN1": {"type": "erythroid", "expected": "synergistic", "program": "Erythroid"},
    "CBL+ctrl": {"type": "erythroid_single", "program": "Erythroid"},
    "ctrl+CNN1": {"type": "erythroid_single", "program": "Erythroid"},
    
    # MAPK pathway
    "DUSP9+MAPK1": {"type": "mapk", "expected": "buffering", "program": "Megakaryocyte"},
    "DUSP9+ETS2": {"type": "mapk", "expected": "epistasis", "program": "Megakaryocyte"},
    "ETS2+MAPK1": {"type": "mapk", "expected": "synergistic", "program": "Megakaryocyte"},
    
    # Granulocyte differentiation
    "CEBPE+CEBPA": {"type": "granulocyte", "program": "Granulocyte/apoptosis"},
    "CEBPE+ctrl": {"type": "granulocyte_single", "program": "Granulocyte/apoptosis"},
    "ctrl+CEBPA": {"type": "granulocyte_single", "program": "Granulocyte/apoptosis"},
    
    # Pro-growth
    "KLF1+ctrl": {"type": "growth_single", "program": "Pro-growth"},
    "CEBPE+KLF1": {"type": "growth_double", "program": "Pro-growth"},
}

# Check which perturbations are in our dataset
available_perturbations = []
missing_perturbations = []

for pert, info in KEY_INTERACTIONS.items():
    n_cells = (adata.obs['guide_merged'] == pert).sum()
    if n_cells > 0:
        available_perturbations.append(pert)
        print(f"✓ {pert}: {n_cells} cells ({info.get('expected', 'N/A')})")
    else:
        missing_perturbations.append(pert)
        print(f"✗ {pert}: NOT FOUND")

print()
print(f"Available: {len(available_perturbations)}/{len(KEY_INTERACTIONS)} key perturbations")
print()

# =============================================================================
# STEP 4: ANALYZE GENE PROGRAMS
# =============================================================================

print("=" * 80)
print("STEP 3: Analyzing gene programs (cell states)")
print("=" * 80)
print()

program_counts = adata.obs['gene_program'].value_counts()
print("Gene programs in dataset:")
for program, count in program_counts.items():
    pct = count / len(adata) * 100
    print(f"  {program}: {count} cells ({pct:.1f}%)")
print()

# =============================================================================
# STEP 5: EXTRACT SINGLE GENES FROM DATASET
# =============================================================================

print("=" * 80)
print("STEP 4: Extracting single genes from perturbations")
print("=" * 80)
print()

# Extract unique genes from all perturbations
unique_genes = set()
for guide in adata.obs['guide_merged'].unique():
    if guide != 'ctrl' and '+' in str(guide):
        parts = str(guide).split('+')
        for part in parts:
            if part != 'ctrl':
                unique_genes.add(part)

print(f"Found {len(unique_genes)} unique genes in perturbations:")
sorted_genes = sorted(list(unique_genes))
print(f"  {', '.join(sorted_genes[:20])}")
if len(sorted_genes) > 20:
    print(f"  ... and {len(sorted_genes) - 20} more")
print()

# =============================================================================
# STEP 6: SUMMARY AND RECOMMENDATIONS
# =============================================================================

print("=" * 80)
print("SUMMARY AND TEST STRATEGY")
print("=" * 80)
print()

print("Norman Dataset Characteristics:")
print(f"  • {len(unique_genes)} unique genes perturbed")
print(f"  • {len(perturbation_types['single'])} single perturbations")
print(f"  • {len(perturbation_types['double'])} double perturbations")
print(f"  • 7 cell state programs identified")
print()

print("Recommended Tests:")
print()
print("1. EMBEDDING SEPARATION")
print("   Test: Can Geneformer separate different cell states?")
print("   Method: Extract embeddings for Ctrl vs Erythroid vs Granulocyte")
print("   Metric: Cosine similarity, clustering quality")
print()

print("2. SINGLE PERTURBATION PREDICTION")
print("   Test: Can Geneformer predict single gene activation effects?")
print("   Method: InSilicoPerturber on control cells → predict CBL+ctrl")
print("   Metric: Compare predicted vs actual CBL+ctrl cell embeddings")
print()

print("3. DOUBLE PERTURBATION PREDICTION")
print("   Test: Can Geneformer predict double perturbation effects?")
print("   Method: InSilicoPerturber on control → predict CBL+CNN1")
print("   Metric: Compare predicted vs actual CBL+CNN1 embeddings")
print()

print("4. GENETIC INTERACTION DETECTION")
print("   Test: Can Geneformer detect synergy (CBL+CNN1 > CBL + CNN1)?")
print("   Method: Compare (CBL+CNN1) vs (CBL+ctrl) + (ctrl+CNN1)")
print("   Metric: Additivity score, interaction strength")
print()

print("5. PATHWAY ORDERING")
print("   Test: Can Geneformer order genes in pathways (DUSP9 → MAPK1 → ETS2)?")
print("   Method: Analyze perturbation effects and epistasis patterns")
print("   Metric: Correct ordering of known pathways")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("This analysis shows the Norman dataset structure.")
print()
print("To properly test Geneformer, we need to:")
print("  1. Extract embeddings for all perturbation conditions")
print("  2. Use InSilicoPerturber to predict perturbation effects")
print("  3. Compare predicted vs actual transcriptional changes")
print("  4. Measure synergy for known interacting pairs")
print()
print("The current test_norman_dataset.py tests only state separation.")
print("A full evaluation requires comparing:")
print("  • Predicted perturbation effects vs actual measured effects")
print("  • Model's ability to predict unseen double perturbations")
print("  • Detection of synergistic interactions")
print()
print("This matches the Pert-VALA evaluation approach for foundation models.")
print()
print("=" * 80)

# Save analysis results
analysis_results = {
    'total_cells': len(adata),
    'n_genes': len(unique_genes),
    'n_control_cells': ctrl_cells,
    'n_single_pert_cells': single_cells,
    'n_double_pert_cells': double_cells,
    'unique_genes': sorted_genes,
    'gene_programs': program_counts.to_dict(),
    'available_key_perturbations': available_perturbations,
    'missing_key_perturbations': missing_perturbations,
}

output_json = OUTPUT_DIR / "norman_dataset_analysis.pkl"
with open(output_json, 'wb') as f:
    pickle.dump(analysis_results, f)

print(f"Analysis saved to: {output_json}")
print()

