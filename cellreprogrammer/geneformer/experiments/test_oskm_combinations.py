"""
Test All OSKM Factor Combinations and Rank Results
=================================================

Tests all 15 possible combinations of OSKM factors (1, 2, 3, and 4-gene combinations)
and ranks them by their effectiveness in shifting cells toward iPSC state.

This will test:
- 4 single genes
- 6 pairs  
- 4 triplets
- 1 quadruplet (all 4 together)

Total: 15 combinations

Adapted to use helical's APIs and the gf-20L-151M-i4096 model.
"""

import os
import sys
import pickle
import glob
import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path

# Path setup - works both locally and on remote server
# Assuming structure: helical/cellreprogrammer/experiments/script.py
script_dir = Path(__file__).resolve().parent
BASE_DIR = script_dir.parent.parent  # Go up to helical/
# Fallback to remote server path if BASE_DIR doesn't exist
if not BASE_DIR.exists():
    BASE_DIR = Path("/home/ubuntu/data-at-virginia/helical")

CELLREPROGRAMMER_DIR = BASE_DIR / "cellreprogrammer"

# Import helical's GeneformerConfig and Downloader
from helical.models.geneformer.geneformer_config import GeneformerConfig
from helical.utils.downloader import Downloader

# Try to import original Geneformer utilities
try:
    from geneformer import InSilicoPerturber, EmbExtractor, InSilicoPerturberStats
except ImportError:
    # Try local Geneformer repo
    GENEFORMER_REPO = Path("/home/ubuntu/data-at-virginia/Geneformer")
    if GENEFORMER_REPO.exists():
        sys.path.insert(0, str(GENEFORMER_REPO))
        from geneformer import InSilicoPerturber, EmbExtractor, InSilicoPerturberStats
    else:
        print("=" * 80)
        print("ERROR: Missing geneformer package")
        print("=" * 80)
        print("\nThis script requires the original Geneformer package for")
        print("InSilicoPerturber, EmbExtractor, and InSilicoPerturberStats.\n")
        print("Install it with:")
        print("  pip install -e /path/to/Geneformer/\n")
        sys.exit(1)

# =============================================================================
# CONFIGURATION  
# =============================================================================

# Model configuration
MODEL_NAME = "gf-20L-151M-i4096"
GENEFORMER_CONFIG = GeneformerConfig(model_name=MODEL_NAME, batch_size=50)

# Ensure model is downloaded
print("Ensuring model is downloaded via helical...")
downloader = Downloader()
for file in GENEFORMER_CONFIG.list_of_files_to_download:
    downloader.download_via_name(file)
print("✓ Model files downloaded")

MODEL_PATH = GENEFORMER_CONFIG.files_config["model_files_dir"]
print(f"✓ Model path: {MODEL_PATH}")

# Verify model path exists and contains required files
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model directory does not exist: {MODEL_PATH}")

required_files = ["config.json"]
if (MODEL_PATH / "model.safetensors").exists():
    required_files.append("model.safetensors")
elif (MODEL_PATH / "pytorch_model.bin").exists():
    required_files.append("pytorch_model.bin")
else:
    raise FileNotFoundError(f"Model weights file not found in {MODEL_PATH}")

print(f"✓ Model files verified: {', '.join(required_files)}")
print()

# Paths - relative to cellreprogrammer directory
DATA_DIR = CELLREPROGRAMMER_DIR / "data"
INPUT_DATA_PATH = DATA_DIR / "tokenized" / "fibroblast_ipsc.dataset"  # V2/V3 models use 4096 tokens
OUTPUT_DIR = CELLREPROGRAMMER_DIR / "results" / "oskm_combinations"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OSKM factors
OSKM_GENES = {
    "ENSG00000204531": "OCT4 (POU5F1)",
    "ENSG00000181449": "SOX2",
    "ENSG00000136826": "KLF4",
    "ENSG00000136997": "MYC"
}

OSKM_FACTORS = list(OSKM_GENES.keys())

# Random control genes (for reference, not used in this script)
RANDOM_CONTROL_GENES = [
    "ENSG00000111640",  # GAPDH
    "ENSG00000075624",  # ACTB
]

# Cell state definitions
CELL_STATES = {
    "state_key": "cell_type",
    "start_state": "Fibroblast",
    "goal_state": "iPSC",
    "alt_states": []
}

FILTER_DATA = {
    "cell_type": ["Fibroblast", "iPSC"]
}

# Computation settings
MAX_NCELLS = None  # Use all available cells for accuracy
NPROC = 1  # Must be > 0 for datasets library
FORWARD_BATCH_SIZE = 50

# Map helical V3 models to original Geneformer V2
raw_version = GENEFORMER_CONFIG.model_map[MODEL_NAME]["model_version"].upper()
MODEL_VERSION = "V2" if raw_version == "V3" else raw_version

os.environ['DATASETS_NUM_PROC'] = str(NPROC)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

print("=" * 80)
print("OSKM Combinations Test")
print("=" * 80)
print()
print("Paths:")
print(f"  Working directory: {CELLREPROGRAMMER_DIR}")
print(f"  Model: {MODEL_NAME}")
print(f"  Model path: {MODEL_PATH}")
print(f"  Data: {INPUT_DATA_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print()
print("OSKM factors:")
for ensembl_id, name in OSKM_GENES.items():
    print(f"  • {name}: {ensembl_id}")
print()
print("Cell states:")
print(f"  Start: {CELL_STATES['start_state']}")
print(f"  Goal: {CELL_STATES['goal_state']}")
print()
print("Computational settings:")
print(f"  Max cells: {MAX_NCELLS if MAX_NCELLS is not None else 'All available'}")
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
    "state_embs"
)

print("✓ State embeddings extracted successfully")
print()

# =============================================================================
# STEP 2: GENERATE ALL COMBINATIONS
# =============================================================================

print("=" * 80)
print("STEP 2: Generating all OSKM combinations")
print("=" * 80)

all_combinations = []
for r in range(1, 5):  # 1, 2, 3, 4 genes
    all_combinations.extend(list(combinations(OSKM_FACTORS, r)))

print(f"Total combinations: {len(all_combinations)}")
print("Breakdown:")
print(f"  • 1-gene: 4 combinations")
print(f"  • 2-gene: 6 combinations")
print(f"  • 3-gene: 4 combinations")
print(f"  • 4-gene: 1 combination")
print()

# =============================================================================
# STEP 3: TEST EACH COMBINATION
# =============================================================================

print("=" * 80)
print("STEP 3: Testing each combination")
print("=" * 80)

results_list = []

for i, combo in enumerate(all_combinations, 1):
    combo_list = list(combo)
    combo_size = len(combo_list)
    
    # Create readable name
    combo_name = "+".join([OSKM_GENES[g] for g in combo_list])
    
    print(f"\n[{i}/{len(all_combinations)}] Testing {combo_size}-gene: {combo_name}")
    
    try:
        # Ensure output subdirectory exists
        combo_output_dir = OUTPUT_DIR / f"combo_{i:02d}"
        os.makedirs(combo_output_dir, exist_ok=True)
        
        isp = InSilicoPerturber(
            perturb_type="overexpress",
            perturb_rank_shift=None,
            genes_to_perturb=combo_list,
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
        
        output_name = f"combo_{i:02d}"
        
        # Run perturbation
        isp.perturb_data(
            str(MODEL_PATH),
            str(INPUT_DATA_PATH),
            str(combo_output_dir),
            output_name
        )
        
        # Check if stats CSV already exists
        stats_csv = combo_output_dir / f"{output_name}_stats.csv"
        
        if stats_csv.exists():
            # Stats already generated, just read it
            try:
                df = pd.read_csv(stats_csv)
                if 'Shift_to_goal_end' in df.columns:
                    shift_value = df['Shift_to_goal_end'].iloc[0]
                    results_list.append({
                        'Rank': i,
                        'Combo_Size': combo_size,
                        'Genes': combo_name,
                        'Ensembl_IDs': str(combo_list),
                        'Shift_to_iPSC': shift_value
                    })
                    print(f"  ✓ Shift: {shift_value:.6f} (from existing stats)")
                    continue
            except Exception as e:
                print(f"  ⚠ Could not read existing stats: {e}")
        
        # Try to run stats
        try:
            ispstats = InSilicoPerturberStats(
                mode="goal_state_shift",
                genes_perturbed=combo_list,
                combos=0,
                anchor_gene=None,
                cell_states_to_model=CELL_STATES,
                model_version=MODEL_VERSION
            )
            
            ispstats.get_stats(
                str(combo_output_dir),
                None,
                str(combo_output_dir),
                f"{output_name}_stats"
            )
            
            # Read the results
            df = pd.read_csv(stats_csv)
            if 'Shift_to_goal_end' in df.columns:
                shift_value = df['Shift_to_goal_end'].iloc[0]
                results_list.append({
                    'Rank': i,
                    'Combo_Size': combo_size,
                    'Genes': combo_name,
                    'Ensembl_IDs': str(combo_list),
                    'Shift_to_iPSC': shift_value
                })
                print(f"  ✓ Shift: {shift_value:.6f}")
            else:
                print(f"  ✗ No shift data found")
        except Exception as stats_error:
            # Fallback: Extract shift directly from pickle
            try:
                # Find the actual pickle file (they have complex names)
                pickle_pattern = str(combo_output_dir / f"in_silico_overexpress_{output_name}_cell_embs_dict_*_raw.pickle")
                pickle_files = glob.glob(pickle_pattern)
                
                if not pickle_files:
                    print(f"  ✗ No pickle file found matching pattern")
                    continue
                
                pickle_file = pickle_files[0]
                with open(pickle_file, 'rb') as f:
                    perturbation_data = pickle.load(f)
                
                shift_values = []
                if 'iPSC' in perturbation_data:
                    for key, values in perturbation_data['iPSC'].items():
                        if isinstance(values, list):
                            shift_values.extend(values)
                        elif isinstance(values, np.ndarray):
                            shift_values.extend(values.flatten().tolist())
                
                if shift_values:
                    shift_value = np.mean(shift_values)
                    results_list.append({
                        'Rank': i,
                        'Combo_Size': combo_size,
                        'Genes': combo_name,
                        'Ensembl_IDs': str(combo_list),
                        'Shift_to_iPSC': shift_value
                    })
                    print(f"  ✓ Shift: {shift_value:.6f} (from pickle)")
                else:
                    print(f"  ✗ No shift data found")
            except Exception as e:
                print(f"  ✗ Could not extract shift: {e}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n✓ All {len(all_combinations)} combinations tested")
print()

# =============================================================================
# STEP 4: RANK RESULTS
# =============================================================================

print("=" * 80)
print("STEP 4: Ranking all combinations by iPSC shift")
print("=" * 80)

results_df = pd.DataFrame(results_list)

if len(results_df) > 0:
    # Sort by shift value (descending)
    results_df = results_df.sort_values('Shift_to_iPSC', ascending=False)
    
    # Add final rank
    results_df.insert(0, 'Final_Rank', range(1, len(results_df) + 1))
    
    # Save ranked results
    output_csv = OUTPUT_DIR / "all_oskm_combinations_ranked.csv"
    results_df.to_csv(output_csv, index=False)
    
    print("\nTop 5 combinations:")
    print()
    for idx, row in results_df.head(5).iterrows():
        print(f"  Rank {row['Final_Rank']}: {row['Genes']}")
        print(f"    Shift: {row['Shift_to_iPSC']:.6f}")
        print()
    
    print(f"\n✓ Results saved to: {output_csv}")
    print()
    
    # Summary by combo size
    print("Summary by number of genes:")
    for size in [1, 2, 3, 4]:
        size_df = results_df[results_df['Combo_Size'] == size]
        if len(size_df) > 0:
            best = size_df.iloc[0]
            print(f"  {size}-gene combo: {best['Genes']} (shift: {best['Shift_to_iPSC']:.6f})")
    print()
else:
    print("No results to rank")
    print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("ALL STEPS COMPLETE!")
print("=" * 80)
print()
print(f"Tested: {len(all_combinations)} combinations")
print(f"Results: {len(results_df)} successful")
print()
print("Files generated:")
print(f"  • {OUTPUT_DIR}/all_oskm_combinations_ranked.csv (main results)")
print(f"  • Individual perturbation pickles in: {OUTPUT_DIR}/combo_XX/")
print()
