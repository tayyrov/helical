"""
Test Norman Dataset with Geneformer
===================================

Tests Geneformer's ability to distinguish between CRISPR-activated and 
non-activated K562 cells from the Norman 2019 dataset.

This script:
1. Extracts embeddings for activated vs non-activated cells
2. Tests in-silico perturbations to shift cells from non-activated to activated state
3. Measures the effectiveness of perturbations

Adapted from test_oskm_combinations.py for Norman dataset.
"""

import os
import sys
import pickle
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Path setup - works both locally and on remote server
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
MODEL_NAME = "gf-20L-151M-i4096"  # Large model for best performance
GENEFORMER_CONFIG = GeneformerConfig(model_name=MODEL_NAME, batch_size=50)

# Ensure model is downloaded
print("Ensuring model is downloaded via helical...")
downloader = Downloader()
for file in GENEFORMER_CONFIG.list_of_files_to_download:
    downloader.download_via_name(file)
print("✓ Model files downloaded")

MODEL_PATH = GENEFORMER_CONFIG.files_config["model_files_dir"]
print(f"✓ Model path: {MODEL_PATH}")

# Verify model path exists
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

# Paths
DATA_DIR = CELLREPROGRAMMER_DIR / "data"
INPUT_DATA_PATH = DATA_DIR / "tokenized" / "norman_k562.dataset"
OUTPUT_DIR = CELLREPROGRAMMER_DIR / "results" / "norman_dataset"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cell state definitions for Norman dataset
CELL_STATES = {
    "state_key": "is_activated",  # Use is_activated column
    "start_state": 0,  # Non-activated (control)
    "goal_state": 1,  # Activated (CRISPR)
    "alt_states": []
}

FILTER_DATA = {
    "is_activated": [0, 1]  # Include both activated and non-activated
}

# Computation settings
MAX_NCELLS = None  # Use all available cells
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
print("Norman Dataset Test with Geneformer")
print("=" * 80)
print()
print("Paths:")
print(f"  Working directory: {CELLREPROGRAMMER_DIR}")
print(f"  Model: {MODEL_NAME}")
print(f"  Model path: {MODEL_PATH}")
print(f"  Data: {INPUT_DATA_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print()
print("Cell states:")
print(f"  Start: Non-activated (is_activated=0)")
print(f"  Goal: Activated (is_activated=1)")
print()
print("Computational settings:")
print(f"  Max cells: {MAX_NCELLS if MAX_NCELLS is not None else 'All available'}")
print(f"  Batch size: {FORWARD_BATCH_SIZE}")
print(f"  Model version: {MODEL_VERSION}")
print()

# Verify input data exists
if not INPUT_DATA_PATH.exists():
    print(f"ERROR: Tokenized dataset not found: {INPUT_DATA_PATH}")
    print("Please run prepare_norman_data.py first to prepare and tokenize the data.")
    sys.exit(1)

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
# STEP 2: ANALYZE EMBEDDING SEPARATION
# =============================================================================

print("=" * 80)
print("STEP 2: Analyzing embedding separation")
print("=" * 80)

# Load the embeddings to check separation
emb_pickle = OUTPUT_DIR / "state_embs" / "state_embs_dict.pickle"
if emb_pickle.exists():
    with open(emb_pickle, 'rb') as f:
        embs_data = pickle.load(f)
    
    if 0 in embs_data and 1 in embs_data:
        non_activated_embs = embs_data[0]
        activated_embs = embs_data[1]
        
        # Calculate mean embeddings
        if isinstance(non_activated_embs, dict):
            non_activated_mean = np.mean([np.array(v).flatten() for v in non_activated_embs.values()], axis=0)
        else:
            non_activated_mean = np.array(non_activated_embs).flatten()
        
        if isinstance(activated_embs, dict):
            activated_mean = np.mean([np.array(v).flatten() for v in activated_embs.values()], axis=0)
        else:
            activated_mean = np.array(activated_embs).flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(non_activated_mean, activated_mean)
        norm_non = np.linalg.norm(non_activated_mean)
        norm_act = np.linalg.norm(activated_mean)
        cosine_sim = dot_product / (norm_non * norm_act) if (norm_non * norm_act) > 0 else 0
        
        print(f"✓ Embedding analysis:")
        print(f"  Cosine similarity between states: {cosine_sim:.4f}")
        print(f"  (Lower = better separation)")
        print()
    else:
        print("⚠ Could not find state embeddings in expected format")
        print()
else:
    print("⚠ State embeddings pickle not found, skipping analysis")
    print()

# =============================================================================
# STEP 3: TEST PERTURBATIONS
# =============================================================================

print("=" * 80)
print("STEP 3: Testing perturbations")
print("=" * 80)

# For Norman dataset, we'll test overexpression of top differentially expressed genes
# or known activation markers. Since we don't have specific target genes,
# we'll test a few common activation-related genes

# Common genes that might be involved in activation
# These are examples - user should adjust based on their specific dataset
# You can also load genes from the dataset's perturbation labels if available
TEST_GENES = [
    "ENSG00000139618",  # BRCA1 - often involved in cell cycle
    "ENSG00000157764",  # BRAF - signaling
    "ENSG00000141510",  # TP53 - tumor suppressor
    "ENSG00000136997",  # MYC - proto-oncogene
    "ENSG00000181449",  # SOX2 - transcription factor
]

# Alternative: Load actual perturbed genes from the dataset if available
# Uncomment and modify this section to use genes from the dataset:
"""
from datasets import load_from_disk
dataset = load_from_disk(str(INPUT_DATA_PATH))
if 'perturbation_label' in dataset.column_names:
    # Extract unique perturbed genes (excluding 'control')
    perturbed_genes = set()
    for label in dataset['perturbation_label']:
        if label and label != 'control' and label != '':
            # Try to parse Ensembl ID from label
            # Adjust parsing logic based on your dataset format
            perturbed_genes.add(label)
    if perturbed_genes:
        TEST_GENES = list(perturbed_genes)[:10]  # Limit to top 10
        print(f"Using {len(TEST_GENES)} genes from dataset perturbations")
"""

# Alternative: test all genes from a specific perturbation if available
# For now, we'll test individual genes and combinations

print(f"Testing {len(TEST_GENES)} genes individually...")
print()

results_list = []

for i, gene_id in enumerate(TEST_GENES, 1):
    print(f"\n[{i}/{len(TEST_GENES)}] Testing gene: {gene_id}")
    
    try:
        # Ensure output subdirectory exists
        gene_output_dir = OUTPUT_DIR / f"gene_{i:02d}_{gene_id[:10]}"
        os.makedirs(gene_output_dir, exist_ok=True)
        
        isp = InSilicoPerturber(
            perturb_type="overexpress",
            perturb_rank_shift=None,
            genes_to_perturb=[gene_id],
            combos=0,
            anchor_gene=None,
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cls",
            cell_emb_style="mean_pool",
            filter_data={"is_activated": [0]},  # Start from non-activated cells
            cell_states_to_model=CELL_STATES,
            state_embs_dict=state_embs_dict,
            max_ncells=MAX_NCELLS,
            emb_layer=-1,
            forward_batch_size=FORWARD_BATCH_SIZE,
            model_version=MODEL_VERSION,
            nproc=NPROC
        )
        
        output_name = f"gene_{i:02d}"
        
        # Run perturbation
        isp.perturb_data(
            str(MODEL_PATH),
            str(INPUT_DATA_PATH),
            str(gene_output_dir),
            output_name
        )
        
        # Check if stats CSV already exists
        stats_csv = gene_output_dir / f"{output_name}_stats.csv"
        
        if stats_csv.exists():
            # Stats already generated, just read it
            try:
                df = pd.read_csv(stats_csv)
                if 'Shift_to_goal_end' in df.columns:
                    shift_value = df['Shift_to_goal_end'].iloc[0]
                    results_list.append({
                        'Gene_ID': gene_id,
                        'Shift_to_Activated': shift_value
                    })
                    print(f"  ✓ Shift: {shift_value:.6f} (from existing stats)")
                    continue
            except Exception as e:
                print(f"  ⚠ Could not read existing stats: {e}")
        
        # Try to run stats
        try:
            ispstats = InSilicoPerturberStats(
                mode="goal_state_shift",
                genes_perturbed=[gene_id],
                combos=0,
                anchor_gene=None,
                cell_states_to_model=CELL_STATES,
                model_version=MODEL_VERSION
            )
            
            ispstats.get_stats(
                str(gene_output_dir),
                None,
                str(gene_output_dir),
                f"{output_name}_stats"
            )
            
            # Read the results
            df = pd.read_csv(stats_csv)
            if 'Shift_to_goal_end' in df.columns:
                shift_value = df['Shift_to_goal_end'].iloc[0]
                results_list.append({
                    'Gene_ID': gene_id,
                    'Shift_to_Activated': shift_value
                })
                print(f"  ✓ Shift: {shift_value:.6f}")
            else:
                print(f"  ✗ No shift data found")
        except Exception as stats_error:
            print(f"  ⚠ Stats generation failed: {stats_error}")
            # Try to extract from pickle as fallback
            try:
                pickle_pattern = str(gene_output_dir / f"in_silico_overexpress_{output_name}_cell_embs_dict_*_raw.pickle")
                pickle_files = glob.glob(pickle_pattern)
                
                if pickle_files:
                    pickle_file = pickle_files[0]
                    with open(pickle_file, 'rb') as f:
                        perturbation_data = pickle.load(f)
                    
                    shift_values = []
                    if 1 in perturbation_data:  # Goal state (activated)
                        for key, values in perturbation_data[1].items():
                            if isinstance(values, list):
                                shift_values.extend(values)
                            elif isinstance(values, np.ndarray):
                                shift_values.extend(values.flatten().tolist())
                    
                    if shift_values:
                        shift_value = np.mean(shift_values)
                        results_list.append({
                            'Gene_ID': gene_id,
                            'Shift_to_Activated': shift_value
                        })
                        print(f"  ✓ Shift: {shift_value:.6f} (from pickle)")
                    else:
                        print(f"  ✗ No shift data found in pickle")
                else:
                    print(f"  ✗ No pickle file found")
            except Exception as e:
                print(f"  ✗ Could not extract shift: {e}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n✓ All {len(TEST_GENES)} genes tested")
print()

# =============================================================================
# STEP 4: RANK RESULTS
# =============================================================================

print("=" * 80)
print("STEP 4: Ranking results by activation shift")
print("=" * 80)

if len(results_list) > 0:
    results_df = pd.DataFrame(results_list)
    
    # Sort by shift value (descending - higher shift = better)
    results_df = results_df.sort_values('Shift_to_Activated', ascending=False)
    
    # Add rank
    results_df.insert(0, 'Rank', range(1, len(results_df) + 1))
    
    # Save ranked results
    output_csv = OUTPUT_DIR / "norman_perturbation_results.csv"
    results_df.to_csv(output_csv, index=False)
    
    print("\nTop genes by activation shift:")
    print()
    for idx, row in results_df.head(10).iterrows():
        print(f"  Rank {row['Rank']}: {row['Gene_ID']}")
        print(f"    Shift: {row['Shift_to_Activated']:.6f}")
        print()
    
    print(f"\n✓ Results saved to: {output_csv}")
    print()
else:
    print("No results to rank")
    print()

# =============================================================================
# STEP 5: COMPARE TO BASELINE
# =============================================================================

print("=" * 80)
print("STEP 5: Baseline comparison")
print("=" * 80)

# Calculate baseline: how well can we distinguish states without perturbation?
if emb_pickle.exists():
    try:
        with open(emb_pickle, 'rb') as f:
            embs_data = pickle.load(f)
        
        if 0 in embs_data and 1 in embs_data:
            print("✓ Baseline state separation calculated in Step 2")
            print("  Use embedding cosine similarity as baseline reference")
        else:
            print("⚠ Could not calculate baseline")
    except Exception as e:
        print(f"⚠ Could not load baseline: {e}")
else:
    print("⚠ Baseline embeddings not available")

print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("ALL STEPS COMPLETE!")
print("=" * 80)
print()
print(f"Tested: {len(TEST_GENES)} genes")
print(f"Results: {len(results_list)} successful")
print()
print("Files generated:")
print(f"  • {OUTPUT_DIR}/norman_perturbation_results.csv (main results)")
print(f"  • State embeddings: {OUTPUT_DIR}/state_embs/")
print(f"  • Individual perturbation results: {OUTPUT_DIR}/gene_XX/")
print()
print("Next steps:")
print("  1. Review results to identify top-performing perturbations")
print("  2. Test combinations of top genes if desired")
print("  3. Compare with known biology from Norman et al. 2019 paper")
print()
print("=" * 80)

