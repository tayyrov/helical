"""
Comprehensive Evaluation of Geneformer on Norman Dataset
=========================================================

Evaluates Geneformer's ability to capture perturbation effects using the
Norman et al. 2019 dataset (CRISPRa single and double perturbations).

Following PertEval-scFM benchmarking methodology:
1. Extract embeddings for all perturbation conditions
2. Compute perturbation effect vectors (perturbed - control)
3. Evaluate clustering quality and biological coherence
4. Test model's ability to distinguish perturbation types
5. Measure correlation with known biology (gene programs)

This is the proper way to evaluate single-cell foundation models on
perturbation data when you have actual measurements (vs predictions).

Reference: PertEval-scFM: Benchmarking Single-Cell Foundation Models 
for Perturbation Effect Prediction
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
import umap

# Path setup
script_dir = Path(__file__).resolve().parent
CELLREPROGRAMMER_DIR = script_dir.parent.parent

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

# Add helical to path
sys.path.insert(0, str(BASE_DIR))

from helical.models.geneformer import Geneformer, GeneformerConfig
from datasets import load_from_disk
import scanpy as sc
import torch

print("=" * 80)
print("Comprehensive Geneformer Evaluation on Norman Dataset")
print("=" * 80)
print()
print("Following PertEval-scFM benchmarking methodology")
print("Reference: Norman et al. 2019 Science")
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = CELLREPROGRAMMER_DIR / "data"
INPUT_DATA_PATH = DATA_DIR / "tokenized" / "norman_k562.dataset"
PREPARED_H5AD = DATA_DIR / "prepared" / "norman_k562_prepared.h5ad"
OUTPUT_DIR = CELLREPROGRAMMER_DIR / "results" / "norman_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configuration - use a fast model for evaluation
MODEL_NAME = "gf-12L-38M-i4096"  # V2 model, good balance of speed and performance
BATCH_SIZE = 50

print(f"Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Data: {INPUT_DATA_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print()

# =============================================================================
# STEP 1: LOAD DATA AND MODEL
# =============================================================================

print("=" * 80)
print("STEP 1: Loading data and model")
print("=" * 80)
print()

if not INPUT_DATA_PATH.exists():
    print(f"ERROR: Tokenized dataset not found: {INPUT_DATA_PATH}")
    print("Please run prepare_norman_data.py first")
    sys.exit(1)

if not PREPARED_H5AD.exists():
    print(f"ERROR: Prepared h5ad not found: {PREPARED_H5AD}")
    print("Please run prepare_norman_data.py first")
    sys.exit(1)

# Load metadata
print(f"Loading metadata: {PREPARED_H5AD}")
adata = sc.read_h5ad(PREPARED_H5AD)
print(f"✓ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
print()

# Load tokenized dataset
print(f"Loading tokenized dataset: {INPUT_DATA_PATH}")
dataset = load_from_disk(str(INPUT_DATA_PATH))
print(f"✓ Loaded: {len(dataset)} cells")
print()

# Initialize Geneformer
print(f"Initializing Geneformer model: {MODEL_NAME}")
config = GeneformerConfig(model_name=MODEL_NAME, batch_size=BATCH_SIZE)
geneformer = Geneformer(configurer=config)
print(f"✓ Model loaded")
print()

# =============================================================================
# STEP 2: EXTRACT EMBEDDINGS FOR ALL CELLS
# =============================================================================

print("=" * 80)
print("STEP 2: Extracting embeddings for all cells")
print("=" * 80)
print()

embeddings_file = OUTPUT_DIR / "geneformer_embeddings.npy"

if embeddings_file.exists():
    print(f"Loading cached embeddings: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    print(f"✓ Loaded embeddings: {embeddings.shape}")
else:
    print("Extracting embeddings (this may take a while)...")
    print(f"Processing {len(dataset)} cells in batches of {BATCH_SIZE}...")
    
    embeddings = geneformer.get_embeddings(dataset)
    
    print(f"✓ Extracted embeddings: {embeddings.shape}")
    
    # Save embeddings
    np.save(embeddings_file, embeddings)
    print(f"✓ Saved embeddings to: {embeddings_file}")

print()

# =============================================================================
# STEP 3: ORGANIZE EMBEDDINGS BY PERTURBATION
# =============================================================================

print("=" * 80)
print("STEP 3: Organizing embeddings by perturbation condition")
print("=" * 80)
print()

# Create mapping from cell to perturbation
perturbation_to_indices = defaultdict(list)
perturbation_to_embeddings = {}
perturbation_to_program = {}

for idx, guide in enumerate(adata.obs['guide_merged']):
    perturbation_to_indices[guide].append(idx)

print(f"Found {len(perturbation_to_indices)} unique perturbations")
print()

# Compute mean embeddings per perturbation
print("Computing mean embeddings per perturbation...")
for pert, indices in perturbation_to_indices.items():
    pert_embeddings = embeddings[indices]
    perturbation_to_embeddings[pert] = np.mean(pert_embeddings, axis=0)
    
    # Get gene program for this perturbation
    program = adata.obs.iloc[indices[0]]['gene_program']
    perturbation_to_program[pert] = program

print(f"✓ Computed mean embeddings for {len(perturbation_to_embeddings)} perturbations")
print()

# Categorize perturbations
control_perts = []
single_perts = []
double_perts = []

for pert in perturbation_to_embeddings.keys():
    if pert == 'ctrl':
        control_perts.append(pert)
    elif '+' in pert:
        parts = pert.split('+')
        if 'ctrl' in parts:
            single_perts.append(pert)
        else:
            double_perts.append(pert)

print(f"Perturbation breakdown:")
print(f"  Control: {len(control_perts)}")
print(f"  Single: {len(single_perts)}")
print(f"  Double: {len(double_perts)}")
print()

# =============================================================================
# STEP 4: COMPUTE PERTURBATION EFFECT VECTORS
# =============================================================================

print("=" * 80)
print("STEP 4: Computing perturbation effect vectors")
print("=" * 80)
print()

# Perturbation effect = perturbed_embedding - control_embedding
control_embedding = perturbation_to_embeddings['ctrl']

perturbation_effects = {}
for pert, emb in perturbation_to_embeddings.items():
    if pert != 'ctrl':
        perturbation_effects[pert] = emb - control_embedding

print(f"✓ Computed {len(perturbation_effects)} perturbation effect vectors")
print()

# =============================================================================
# STEP 5: EVALUATE EMBEDDING QUALITY
# =============================================================================

print("=" * 80)
print("STEP 5: Evaluating embedding quality")
print("=" * 80)
print()

# 5.1: Silhouette score for gene programs
print("5.1: Clustering quality by gene program")
print("-" * 40)

# Create labels and embeddings array
all_labels = []
all_embeds = []
label_to_int = {prog: i for i, prog in enumerate(adata.obs['gene_program'].unique())}

for idx in range(len(adata)):
    all_labels.append(label_to_int[adata.obs.iloc[idx]['gene_program']])
    all_embeds.append(embeddings[idx])

all_labels = np.array(all_labels)
all_embeds = np.array(all_embeds)

# Subsample for silhouette score (too slow on full dataset)
n_sample = min(5000, len(all_labels))
sample_idx = np.random.choice(len(all_labels), n_sample, replace=False)

silhouette = silhouette_score(all_embeds[sample_idx], all_labels[sample_idx])
print(f"  Silhouette score: {silhouette:.4f}")
print(f"  (Range: [-1, 1], higher is better)")
print()

# 5.2: Separation of perturbation types
print("5.2: Separation of control vs single vs double perturbations")
print("-" * 40)

pert_type_embeddings = {
    'Control': [perturbation_to_embeddings['ctrl']],
    'Single': [perturbation_to_embeddings[p] for p in single_perts],
    'Double': [perturbation_to_embeddings[p] for p in double_perts]
}

# Compute within-group and between-group distances
for type1 in ['Control', 'Single', 'Double']:
    for type2 in ['Control', 'Single', 'Double']:
        if type1 == type2:
            # Within-group distance
            embs = pert_type_embeddings[type1]
            if len(embs) > 1:
                distances = []
                for i in range(len(embs)):
                    for j in range(i+1, len(embs)):
                        distances.append(cosine(embs[i], embs[j]))
                mean_dist = np.mean(distances) if distances else 0
                print(f"  Within {type1}: {mean_dist:.4f}")
        elif type1 < type2:  # Only compute once per pair
            # Between-group distance
            distances = []
            for emb1 in pert_type_embeddings[type1]:
                for emb2 in pert_type_embeddings[type2]:
                    distances.append(cosine(emb1, emb2))
            mean_dist = np.mean(distances)
            print(f"  Between {type1}-{type2}: {mean_dist:.4f}")

print()

# =============================================================================
# STEP 6: ANALYZE KEY PERTURBATIONS FROM NORMAN PAPER
# =============================================================================

print("=" * 80)
print("STEP 6: Analyzing key perturbations from Norman et al. 2019")
print("=" * 80)
print()

# 6.1: CBL+CNN1 erythroid synergy
print("6.1: CBL+CNN1 erythroid synergy (key finding from paper)")
print("-" * 40)

if 'CBL+CNN1' in perturbation_effects:
    cbl_cnn1_effect = perturbation_effects['CBL+CNN1']
    
    # Find if we have singles
    cbl_single = None
    cnn1_single = None
    
    for pert in single_perts:
        if 'CBL' in pert and 'CNN1' not in pert:
            cbl_single = pert
        if 'CNN1' in pert and 'CBL' not in pert:
            cnn1_single = pert
    
    if cbl_single and cnn1_single:
        cbl_effect = perturbation_effects[cbl_single]
        cnn1_effect = perturbation_effects[cnn1_single]
        additive_effect = cbl_effect + cnn1_effect
        
        # Measure deviation from additivity (synergy)
        synergy_magnitude = np.linalg.norm(cbl_cnn1_effect - additive_effect)
        actual_magnitude = np.linalg.norm(cbl_cnn1_effect)
        additive_magnitude = np.linalg.norm(additive_effect)
        
        print(f"  CBL+CNN1 effect magnitude: {actual_magnitude:.4f}")
        print(f"  Additive prediction: {additive_magnitude:.4f}")
        print(f"  Synergy magnitude: {synergy_magnitude:.4f}")
        print(f"  Synergy ratio: {synergy_magnitude/actual_magnitude:.4f}")
    else:
        print(f"  ⚠ Singles not found (CBL: {cbl_single}, CNN1: {cnn1_single})")
        print(f"  CBL+CNN1 effect magnitude: {np.linalg.norm(cbl_cnn1_effect):.4f}")
else:
    print("  ✗ CBL+CNN1 not found in dataset")

print()

# 6.2: Gene program coherence
print("6.2: Gene program coherence")
print("-" * 40)

program_embeddings = defaultdict(list)
for pert, prog in perturbation_to_program.items():
    if pert != 'ctrl':
        program_embeddings[prog].append(perturbation_effects[pert])

print("  Within-program similarity (mean cosine similarity):")
for prog, embs in program_embeddings.items():
    if len(embs) > 1:
        similarities = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                # Cosine similarity = 1 - cosine distance
                similarities.append(1 - cosine(embs[i], embs[j]))
        mean_sim = np.mean(similarities)
        print(f"    {prog}: {mean_sim:.4f} ({len(embs)} perturbations)")

print()

# =============================================================================
# STEP 7: VISUALIZE PERTURBATION LANDSCAPE
# =============================================================================

print("=" * 80)
print("STEP 7: Visualizing perturbation landscape")
print("=" * 80)
print()

print("Generating visualizations...")

# Prepare data for visualization
effect_matrix = []
effect_labels = []
effect_programs = []
effect_types = []

for pert, effect in perturbation_effects.items():
    effect_matrix.append(effect)
    effect_labels.append(pert)
    effect_programs.append(perturbation_to_program[pert])
    
    if pert in single_perts:
        effect_types.append('Single')
    elif pert in double_perts:
        effect_types.append('Double')
    else:
        effect_types.append('Other')

effect_matrix = np.array(effect_matrix)

# 7.1: PCA visualization
print("  Creating PCA visualization...")
pca = PCA(n_components=2)
effect_pca = pca.fit_transform(effect_matrix)

plt.figure(figsize=(12, 8))
for prog in set(effect_programs):
    mask = np.array(effect_programs) == prog
    plt.scatter(effect_pca[mask, 0], effect_pca[mask, 1], 
                label=prog, alpha=0.6, s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Norman Dataset Perturbations - PCA by Gene Program')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'perturbations_pca_by_program.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: perturbations_pca_by_program.png")

# 7.2: PCA by perturbation type
plt.figure(figsize=(10, 8))
for ptype in ['Single', 'Double']:
    mask = np.array(effect_types) == ptype
    plt.scatter(effect_pca[mask, 0], effect_pca[mask, 1], 
                label=ptype, alpha=0.6, s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Norman Dataset Perturbations - PCA by Type')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'perturbations_pca_by_type.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: perturbations_pca_by_type.png")

# 7.3: UMAP visualization
print("  Creating UMAP visualization...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
effect_umap = reducer.fit_transform(effect_matrix)

plt.figure(figsize=(12, 8))
for prog in set(effect_programs):
    mask = np.array(effect_programs) == prog
    plt.scatter(effect_umap[mask, 0], effect_umap[mask, 1], 
                label=prog, alpha=0.6, s=50)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Norman Dataset Perturbations - UMAP by Gene Program')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'perturbations_umap_by_program.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: perturbations_umap_by_program.png")

print()

# =============================================================================
# STEP 8: GENERATE SUMMARY REPORT
# =============================================================================

print("=" * 80)
print("STEP 8: Generating summary report")
print("=" * 80)
print()

summary = {
    'model': MODEL_NAME,
    'n_cells': len(adata),
    'n_perturbations': len(perturbation_effects),
    'n_single': len(single_perts),
    'n_double': len(double_perts),
    'silhouette_score': silhouette,
    'pca_variance_explained': pca.explained_variance_ratio_[:2].tolist(),
}

# Save summary
summary_file = OUTPUT_DIR / 'evaluation_summary.pkl'
with open(summary_file, 'wb') as f:
    pickle.dump(summary, f)

print(f"✓ Saved summary to: {summary_file}")

# Save perturbation embeddings and effects
results = {
    'perturbation_embeddings': perturbation_to_embeddings,
    'perturbation_effects': perturbation_effects,
    'perturbation_to_program': perturbation_to_program,
    'control_embedding': control_embedding,
}

results_file = OUTPUT_DIR / 'perturbation_embeddings.pkl'
with open(results_file, 'wb') as f:
    pickle.dump(results, f)

print(f"✓ Saved results to: {results_file}")
print()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  • Model: {MODEL_NAME}")
print(f"  • Cells processed: {len(adata):,}")
print(f"  • Perturbations analyzed: {len(perturbation_effects)}")
print(f"    - Single: {len(single_perts)}")
print(f"    - Double: {len(double_perts)}")
print()
print("Key Metrics:")
print(f"  • Silhouette score (gene programs): {silhouette:.4f}")
print(f"  • PCA variance explained (PC1+PC2): {sum(pca.explained_variance_ratio_[:2]):.1%}")
print()
print("Outputs saved to:", OUTPUT_DIR)
print("  • geneformer_embeddings.npy - All cell embeddings")
print("  • perturbation_embeddings.pkl - Mean embeddings per perturbation")
print("  • evaluation_summary.pkl - Summary statistics")
print("  • perturbations_pca_by_program.png - PCA visualization")
print("  • perturbations_umap_by_program.png - UMAP visualization")
print()
print("Next steps:")
print("  • Compare with other models (scGPT, scBERT, etc.)")
print("  • Test on held-out perturbations (train/test split)")
print("  • Measure correlation with known biology")
print("  • Evaluate synergy detection more comprehensively")
print()
print("=" * 80)

