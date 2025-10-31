"""
Reproduce Geneformer In Silico Reprogramming Experiments
=========================================================

This script reproduces the fibroblast → iPSC reprogramming experiments
from Theodoris et al. Nature 2023 using the OSKM (OCT4, SOX2, KLF4, MYC)
Yamanaka factors.

Based on: Xing, Q. R. et al. (2020) Science Advances dataset

IMPORTANT NOTE:
---------------
This script uses InSilicoPerturber, EmbExtractor, and InSilicoPerturberStats
from the ORIGINAL Geneformer package (not helical). These advanced perturbation
utilities are not yet wrapped in helical.

To run this script, you need BOTH packages installed:
  1. helical - for the core model
  2. geneformer - for InSilicoPerturber and related utilities

Install geneformer with: pip install geneformer
"""

import os
import sys
from pathlib import Path

# Try to import from original geneformer package
try:
    from geneformer import InSilicoPerturber, EmbExtractor, InSilicoPerturberStats
except ImportError:
    print("=" * 80)
    print("ERROR: Missing geneformer package")
    print("=" * 80)
    print()
    print("This script requires the original Geneformer package for")
    print("InSilicoPerturber, EmbExtractor, and InSilicoPerturberStats.")
    print()
    print("Install it with:")
    print("  pip install geneformer")
    print()
    print("Alternatively, use CellReprogrammer's overexpression framework:")
    print("  See: src/perturbations/overexpression.py")
    print()
    sys.exit(1)

# Add helical to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = Path("/home/ubuntu/data-at-virginia/helical")
CELLREPROGRAMMER_DIR = BASE_DIR / "cellreprogrammer"
DATA_DIR = CELLREPROGRAMMER_DIR / "data"

# Paths - UPDATE THESE FOR YOUR SETUP
# For V2 model, use HuggingFace model ID - Geneformer will download it
MODEL_NAME = "gf-12L-38M-i4096"  # V2 model for reprogramming
MODEL_PATH = MODEL_NAME  # Will be downloaded from HuggingFace
INPUT_DATA_PATH = DATA_DIR / "tokenized" / "fibroblast_ipsc.dataset"
OUTPUT_DIR = CELLREPROGRAMMER_DIR / "results" / "oskm_experiment"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("Geneformer Reprogramming Experiment")
print("=" * 80)
print()
print("Paths:")
print(f"  Working directory: {CELLREPROGRAMMER_DIR}")
print(f"  Model: {MODEL_PATH}")
print(f"  Data: {INPUT_DATA_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print()

# Verify inputs (MODEL_PATH is a HuggingFace ID, not a local path)
print(f"Model will be loaded from HuggingFace: {MODEL_PATH}")
print()

if not INPUT_DATA_PATH.exists():
    print(f"ERROR: Data not found: {INPUT_DATA_PATH}")
    print("Please run 02_prepare_reprogramming_data.py first")
    sys.exit(1)

# =============================================================================
# OSKM FACTORS (YAMANAKA FACTORS)
# =============================================================================

OSKM_FACTORS = [
    "ENSG00000204531",  # POU5F1 (OCT4)
    "ENSG00000181449",  # SOX2
    "ENSG00000136826",  # KLF4
    "ENSG00000136997"   # MYC
]

RANDOM_CONTROL_GENES = [
    "ENSG00000111640",  # GAPDH
    "ENSG00000075624",  # ACTB
    "ENSG00000204525",  # B2M
    "ENSG00000198899"   # MT-ATP6
]

print("Perturbation genes:")
print(f"  OSKM factors: {len(OSKM_FACTORS)} genes")
print(f"  Random controls: {len(RANDOM_CONTROL_GENES)} genes")
print()

# =============================================================================
# CELL STATE DEFINITIONS
# =============================================================================

# For GSE118258 data
# D0 -> 'Fibroblast' (starting cells)
# D16+ -> 'iPSC' (successfully reprogrammed)
# D16- -> 'Failed_reprogramming'

CELL_STATES = {
    "state_key": "cell_type",
    "start_state": "Fibroblast",
    "goal_state": "iPSC",
    "alt_states": []
}

FILTER_DATA = {
    "cell_type": ["Fibroblast", "iPSC", "Failed_reprogramming"]
}

print("Cell states:")
print(f"  Start: {CELL_STATES['start_state']}")
print(f"  Goal: {CELL_STATES['goal_state']}")
print()

# =============================================================================
# COMPUTATION SETTINGS
# =============================================================================

MAX_NCELLS = 500  # Reduced for memory
NPROC = 1  # Use 1 process for data filtering (Geneformer requires > 0)
FORWARD_BATCH_SIZE = 50  # Reduced for A100 memory constraints
MODEL_VERSION = "V2"  # Using V2 model

# Disable multiprocessing for CUDA compatibility
os.environ['DATASETS_NUM_PROC'] = '1'  # Geneformer requires at least 1
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Computational settings:")
print(f"  Max cells: {MAX_NCELLS}")
print(f"  Batch size: {FORWARD_BATCH_SIZE}")
print(f"  Model version: {MODEL_VERSION}")
print()

# =============================================================================
# STEP 1: EXTRACT STATE EMBEDDINGS
# =============================================================================

print("=" * 80)
print("STEP 1: Extracting state embeddings")
print("=" * 80)

embex = EmbExtractor(
    model_type="Pretrained",
    num_classes=0,
    filter_data=FILTER_DATA,
    max_ncells=MAX_NCELLS,
    emb_layer=-1,
    summary_stat="exact_mean",
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version=MODEL_VERSION,
    nproc=NPROC
)

state_embs_dict = embex.get_state_embs(
    CELL_STATES,
    str(MODEL_PATH),
    str(INPUT_DATA_PATH),
    str(OUTPUT_DIR),
    "reprogramming_state_embs"
)

print("✓ State embeddings extracted")
print()

# =============================================================================
# STEP 2: TEST OSKM FACTORS
# =============================================================================

print("=" * 80)
print("STEP 2: Testing OSKM factors")
print("=" * 80)

OSKM_OUTPUT_DIR = OUTPUT_DIR / "oskm_results"
os.makedirs(OSKM_OUTPUT_DIR, exist_ok=True)

isp_individual = InSilicoPerturber(
    perturb_type="overexpress",
    perturb_rank_shift=None,
    genes_to_perturb=OSKM_FACTORS,
    combos=0,
    anchor_gene=None,
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cls",
    cell_emb_style="mean_pool",
    filter_data={"cell_type": [CELL_STATES["start_state"]]},
    cell_states_to_model=CELL_STATES,
    state_embs_dict=state_embs_dict,
    max_ncells=MAX_NCELLS,
    emb_layer=-1,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version=MODEL_VERSION,
    nproc=NPROC
)

print("Running in silico overexpression of OSKM factors...")
isp_individual.perturb_data(
    str(MODEL_PATH),
    str(INPUT_DATA_PATH),
    str(OSKM_OUTPUT_DIR),
    "oskm_individual"
)

print("✓ OSKM factors perturbation complete")
print()

# =============================================================================
# STEP 3: ANALYZE OSKM FACTORS RESULTS
# =============================================================================

print("=" * 80)
print("STEP 3: Analyzing OSKM factors")
print("=" * 80)

ispstats_individual = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed=OSKM_FACTORS,
    combos=0,
    anchor_gene=None,
    cell_states_to_model=CELL_STATES,
    model_version=MODEL_VERSION
)

ispstats_individual.get_stats(
    str(OSKM_OUTPUT_DIR),
    None,
    str(OUTPUT_DIR),
    "oskm_individual_stats"
)

print("✓ OSKM factors analysis complete")
print(f"  Results: {OUTPUT_DIR}/oskm_individual_stats.csv")
print()

# =============================================================================
# STEP 4: TEST RANDOM CONTROL GENES
# =============================================================================

print("=" * 80)
print("STEP 4: Testing random control genes")
print("=" * 80)

RANDOM_OUTPUT_DIR = OUTPUT_DIR / "random_results"
os.makedirs(RANDOM_OUTPUT_DIR, exist_ok=True)

isp_random = InSilicoPerturber(
    perturb_type="overexpress",
    perturb_rank_shift=None,
    genes_to_perturb=RANDOM_CONTROL_GENES,
    combos=0,
    anchor_gene=None,
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cls",
    cell_emb_style="mean_pool",
    filter_data={"cell_type": [CELL_STATES["start_state"]]},
    cell_states_to_model=CELL_STATES,
    state_embs_dict=state_embs_dict,
    max_ncells=MAX_NCELLS,
    emb_layer=-1,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version=MODEL_VERSION,
    nproc=NPROC
)

print("Running in silico overexpression of random control genes...")
isp_random.perturb_data(
    str(MODEL_PATH),
    str(INPUT_DATA_PATH),
    str(RANDOM_OUTPUT_DIR),
    "random_control"
)

print("✓ Random control perturbation complete")
print()

# =============================================================================
# STEP 5: ANALYZE RANDOM CONTROL RESULTS
# =============================================================================

print("=" * 80)
print("STEP 5: Analyzing random controls")
print("=" * 80)

ispstats_random = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed=RANDOM_CONTROL_GENES,
    combos=0,
    anchor_gene=None,
    cell_states_to_model=CELL_STATES,
    model_version=MODEL_VERSION
)

ispstats_random.get_stats(
    str(RANDOM_OUTPUT_DIR),
    None,
    str(OUTPUT_DIR),
    "random_control_stats"
)

print("✓ Random control analysis complete")
print(f"  Results: {OUTPUT_DIR}/random_control_stats.csv")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("EXPERIMENT COMPLETE!")
print("=" * 80)
print()
print("Results:")
print(f"  1. OSKM: {OUTPUT_DIR}/oskm_individual_stats.csv")
print(f"  2. Random: {OUTPUT_DIR}/random_control_stats.csv")
print()
print("Expected findings:")
print("  • OSKM factors should show shift toward iPSC state")
print("  • Random genes should show minimal/no shift")
print()

# =============================================================================
# QUICK COMPARISON
# =============================================================================

try:
    import pandas as pd
    
    print("=" * 80)
    print("Quick comparison")
    print("=" * 80)
    
    oskm_df = pd.read_csv(f"{OUTPUT_DIR}/oskm_individual_stats.csv")
    random_df = pd.read_csv(f"{OUTPUT_DIR}/random_control_stats.csv")
    
    if 'Shift_to_goal_end' in oskm_df.columns:
        oskm_mean = oskm_df['Shift_to_goal_end'].mean()
        print(f"OSKM mean shift: {oskm_mean:.4f}")
    
    if 'Shift_to_goal_end' in random_df.columns:
        random_mean = random_df['Shift_to_goal_end'].mean()
        print(f"Random mean shift: {random_mean:.4f}")
    
    if 'Shift_to_goal_end' in oskm_df.columns and 'Shift_to_goal_end' in random_df.columns:
        fold_change = oskm_mean / random_mean if random_mean != 0 else float('inf')
        print(f"\nFold change: {fold_change:.2f}x")
        print(f"OSKM factors are {fold_change:.1f}x more effective")
    
except Exception as e:
    print(f"Could not compare results: {e}")

print()
print("=" * 80)

