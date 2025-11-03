"""
Compare All Geneformer Models for Reprogramming
================================================

This script runs the OSKM reprogramming experiment across all available
Geneformer models to identify the best performing model.

Based on: Xing, Q. R. et al. (2020) Science Advances dataset
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback

# Add helical to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from helical.models.geneformer import GeneformerConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = Path("/home/ubuntu/data-at-virginia/helical")
CELLREPROGRAMMER_DIR = BASE_DIR / "cellreprogrammer"
RESULTS_DIR = CELLREPROGRAMMER_DIR / "results"

# All available Geneformer models from helical
AVAILABLE_MODELS = [
    # Version 1.0 models (use 2048 tokens)
    "gf-6L-10M-i2048",
    "gf-12L-40M-i2048",
    "gf-12L-40M-i2048-CZI-CellxGene",
    
    # Version 2.0 models (use 4096 tokens)
    "gf-12L-38M-i4096",
    "gf-20L-151M-i4096",
    "gf-12L-38M-i4096-CLcancer",
    "gf-12L-104M-i4096",
    "gf-12L-104M-i4096-CLcancer",
    "gf-18L-316M-i4096",
]

# Create comparison results directory
COMPARISON_DIR = RESULTS_DIR / "model_comparison"
os.makedirs(COMPARISON_DIR, exist_ok=True)

print("=" * 80)
print("Geneformer Model Comparison for Reprogramming")
print("=" * 80)
print()
print(f"Comparing {len(AVAILABLE_MODELS)} models:")
for i, model in enumerate(AVAILABLE_MODELS, 1):
    print(f"  {i}. {model}")
print()
print(f"Results will be saved to: {COMPARISON_DIR}")
print()

# =============================================================================
# HELPER FUNCTION TO RUN SINGLE MODEL
# =============================================================================

def run_model_experiment(model_name):
    """
    Run the reprogramming experiment for a single model.
    Returns True if successful, False otherwise.
    """
    print("\n" + "=" * 80)
    print(f"RUNNING: {model_name}")
    print("=" * 80)
    
    try:
        # Check if model is valid
        try:
            config = GeneformerConfig(model_name=model_name)
        except Exception as e:
            print(f"✗ Invalid model: {e}")
            return False
        
        # Create a temporary script for this model
        script_content = f'''
import sys
sys.path.insert(0, "{str(Path(__file__).parent.parent.parent)}")

# Import the original script's logic
import os
from pathlib import Path
from helical.models.geneformer import GeneformerConfig
from helical.utils.downloader import Downloader
from geneformer import InSilicoPerturber, EmbExtractor, InSilicoPerturberStats

# Configuration
BASE_DIR = Path("/home/ubuntu/data-at-virginia/helical")
CELLREPROGRAMMER_DIR = BASE_DIR / "cellreprogrammer"
DATA_DIR = CELLREPROGRAMMER_DIR / "data"

MODEL_NAME = "{model_name}"
GENEFORMER_CONFIG = GeneformerConfig(model_name=MODEL_NAME, batch_size=50)
MODEL_PATH = GENEFORMER_CONFIG.files_config["model_files_dir"]

# Determine which tokenized dataset to use based on model version
raw_version = GENEFORMER_CONFIG.model_map[MODEL_NAME]["model_version"].upper()
if raw_version == "V1":
    INPUT_DATA_PATH = DATA_DIR / "tokenized" / "fibroblast_ipsc_v1.dataset"
else:
    # V2 uses 4096 tokenization
    INPUT_DATA_PATH = DATA_DIR / "tokenized" / "fibroblast_ipsc.dataset"

OUTPUT_DIR = CELLREPROGRAMMER_DIR / "results" / "model_comparison" / MODEL_NAME
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "oskm", exist_ok=True)

# Download model if needed
print("Downloading model if needed...")
downloader = Downloader()
for file in GENEFORMER_CONFIG.list_of_files_to_download:
    downloader.download_via_name(file)

# Setup
OSKM_FACTORS = ["ENSG00000204531", "ENSG00000181449", "ENSG00000136826", "ENSG00000136997"]
CELL_STATES = {{"state_key": "cell_type", "start_state": "Fibroblast", "goal_state": "iPSC", "alt_states": []}}
FILTER_DATA = {{"cell_type": ["Fibroblast", "iPSC", "Failed_reprogramming"]}}
MAX_NCELLS = 500
NPROC = 1
FORWARD_BATCH_SIZE = 50
# Map helical V3 models to original Geneformer V2
raw_version = GENEFORMER_CONFIG.model_map[MODEL_NAME]["model_version"].upper()
MODEL_VERSION = "V2" if raw_version == "V3" else raw_version

# Step 1: Extract state embeddings
print("Extracting state embeddings...")
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

# Step 2: OSKM perturbation
print("Testing OSKM factors...")
isp_oskm = InSilicoPerturber(
    perturb_type="overexpress",
    perturb_rank_shift=None,
    genes_to_perturb=OSKM_FACTORS,
    combos=0,
    anchor_gene=None,
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cls",
    cell_emb_style="mean_pool",
    filter_data={{"cell_type": [CELL_STATES["start_state"]]}},
    cell_states_to_model=CELL_STATES,
    state_embs_dict=state_embs_dict,
    max_ncells=MAX_NCELLS,
    emb_layer=-1,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version=MODEL_VERSION,
    nproc=NPROC
)
isp_oskm.perturb_data(
    str(MODEL_PATH),
    str(INPUT_DATA_PATH),
    str(OUTPUT_DIR / "oskm"),
    "oskm"
)

# Step 3: Analyze OSKM
ispstats_oskm = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed=OSKM_FACTORS,
    combos=0,
    anchor_gene=None,
    cell_states_to_model=CELL_STATES,
    model_version=MODEL_VERSION
)
ispstats_oskm.get_stats(
    str(OUTPUT_DIR / "oskm"),
    None,
    str(OUTPUT_DIR),
    "oskm_stats"
)

# Step 4: Random control perturbation
RANDOM_CONTROL_GENES = ["ENSG00000111640", "ENSG00000075624", "ENSG00000204525", "ENSG00000198899"]
os.makedirs(OUTPUT_DIR / "random", exist_ok=True)
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
    filter_data={{"cell_type": [CELL_STATES["start_state"]]}},
    cell_states_to_model=CELL_STATES,
    state_embs_dict=state_embs_dict,
    max_ncells=MAX_NCELLS,
    emb_layer=-1,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version=MODEL_VERSION,
    nproc=NPROC
)
isp_random.perturb_data(
    str(MODEL_PATH),
    str(INPUT_DATA_PATH),
    str(OUTPUT_DIR / "random"),
    "random"
)

# Step 5: Analyze random controls
ispstats_random = InSilicoPerturberStats(
    mode="goal_state_shift",
    genes_perturbed=RANDOM_CONTROL_GENES,
    combos=0,
    anchor_gene=None,
    cell_states_to_model=CELL_STATES,
    model_version=MODEL_VERSION
)
ispstats_random.get_stats(
    str(OUTPUT_DIR / "random"),
    None,
    str(OUTPUT_DIR),
    "random_stats"
)

print(f"✓ Completed: {{MODEL_NAME}}")
'''
        
        # Write and run the script
        temp_script = COMPARISON_DIR / f"run_{model_name}.py"
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # Run the script in subprocess
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {model_name}")
            # Try to extract and print results
            try:
                model_dir = COMPARISON_DIR / model_name
                oskm_stats_file = model_dir / "oskm_stats.csv"
                random_stats_file = model_dir / "random_stats.csv"
                
                if oskm_stats_file.exists():
                    oskm_df = pd.read_csv(oskm_stats_file)
                    oskm_shift = oskm_df['Shift_to_goal_end'].mean() if 'Shift_to_goal_end' in oskm_df.columns else None
                    
                    random_shift = None
                    if random_stats_file.exists():
                        random_df = pd.read_csv(random_stats_file)
                        random_shift = random_df['Shift_to_goal_end'].mean() if 'Shift_to_goal_end' in random_df.columns else None
                    
                    improvement = None
                    if oskm_shift is not None and random_shift is not None:
                        improvement = oskm_shift - random_shift
                    
                    print(f"  Results:")
                    print(f"    OSKM shift: {oskm_shift:.6f}" if oskm_shift is not None else "    OSKM shift: N/A")
                    print(f"    Random shift: {random_shift:.6f}" if random_shift is not None else "    Random shift: N/A")
                    print(f"    Improvement: {improvement:.6f}" if improvement is not None else "    Improvement: N/A")
            except Exception:
                pass  # Don't fail if we can't parse results
            print()
            return True
        else:
            print(f"✗ FAILED: {model_name}")
            print(result.stderr[-500:])  # Show last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {model_name} (exceeded 1 hour)")
        return False
    except Exception as e:
        print(f"✗ ERROR: {model_name}")
        print(traceback.format_exc())
        return False

# =============================================================================
# RUN EXPERIMENTS FOR ALL MODELS
# =============================================================================

results = {}
print("Starting model comparison...")
print()

for model_name in AVAILABLE_MODELS:
    success = run_model_experiment(model_name)
    results[model_name] = success
    print()

# =============================================================================
# COLLECT AND COMPARE RESULTS
# =============================================================================

print("=" * 80)
print("COMPILING RESULTS")
print("=" * 80)
print()

# Collect results from successful runs
comparison_data = []

for model_name, success in results.items():
    model_dir = COMPARISON_DIR / model_name
    oskm_stats_file = model_dir / "oskm_stats.csv"
    random_stats_file = model_dir / "random_stats.csv"
    
    if success and oskm_stats_file.exists():
        try:
            # Read OSKM stats
            oskm_df = pd.read_csv(oskm_stats_file)
            oskm_shift = oskm_df['Shift_to_goal_end'].mean() if 'Shift_to_goal_end' in oskm_df.columns else None
            
            # Read random stats if available
            random_shift = None
            if random_stats_file.exists():
                random_df = pd.read_csv(random_stats_file)
                random_shift = random_df['Shift_to_goal_end'].mean() if 'Shift_to_goal_end' in random_df.columns else None
            
            # Calculate improvement (difference)
            improvement = None
            if oskm_shift is not None and random_shift is not None:
                improvement = oskm_shift - random_shift
            
            comparison_data.append({
                'Model': model_name,
                'OSKM_Shift': oskm_shift,
                'Random_Shift': random_shift,
                'Improvement': improvement,
                'Status': 'Success'
            })
        except Exception as e:
            comparison_data.append({
                'Model': model_name,
                'OSKM_Shift': None,
                'Random_Shift': None,
                'Improvement': None,
                'Status': f'Error: {e}'
            })
    else:
        comparison_data.append({
            'Model': model_name,
            'OSKM_Shift': None,
            'Random_Shift': None,
            'Improvement': None,
            'Status': 'Failed' if not success else 'Missing results'
        })

# Create comparison dataframe
comparison_df = pd.DataFrame(comparison_data)

# Sort by improvement value (descending)
if 'Improvement' in comparison_df.columns:
    comparison_df['Improvement'] = pd.to_numeric(
        comparison_df['Improvement'], 
        errors='coerce'
    )
    comparison_df = comparison_df.sort_values(
        'Improvement', 
        ascending=False, 
        na_position='last'
    )

# Save comparison results
comparison_file = COMPARISON_DIR / "model_comparison_results.csv"
comparison_df.to_csv(comparison_file, index=False)

# Print summary
print("COMPARISON RESULTS:")
print("=" * 80)
print(comparison_df.to_string(index=False))
print()
print(f"Full results saved to: {comparison_file}")
print()

# Print best model
successful_results = comparison_df[comparison_df['Status'] == 'Success']
if not successful_results.empty:
    best_model = successful_results.iloc[0]
    print("=" * 80)
    print("BEST MODEL:")
    print("=" * 80)
    print(f"Model: {best_model['Model']}")
    print(f"OSKM Shift: {best_model['OSKM_Shift']:.6f}")
    print(f"Random Shift: {best_model['Random_Shift']:.6f}")
    improvement_value = best_model['Improvement']
    print(f"Improvement: {improvement_value:.6f}")
    if improvement_value > 0:
        print("  Interpretation: OSKM factors outperform random controls")
    else:
        print("  Interpretation: OSKM factors do not outperform random controls")
    print()
    
    # Show top 3
    if len(successful_results) > 1:
        print("Top 3 models:")
        for i, row in successful_results.head(3).iterrows():
            print(f"  {list(successful_results.index).index(i) + 1}. {row['Model']}")
            print(f"     OSKM: {row['OSKM_Shift']:.6f}, Random: {row['Random_Shift']:.6f}, Improvement: {row['Improvement']:.6f}")
        print()
else:
    print("⚠️  No models completed successfully")

print("=" * 80)
print("COMPARISON COMPLETE!")
print("=" * 80)

