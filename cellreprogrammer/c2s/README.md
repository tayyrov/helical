# Cell2Sen (C2S) Perturbation Guide

This directory contains utilities for running Cell2Sen perturbation experiments in the CellReprogrammer framework.

## Overview

Cell2Sen (C2S-Scale) is a large language model (LLM) based foundation model for single-cell biology that:
- Transforms single-cell RNA-seq profiles into textual "cell sentences"
- Uses generative AI to predict perturbed cell states
- Supports text-based perturbation descriptions (e.g., "overexpress POU5F1")
- Trained on 50+ million human and mouse cells

## Key Features

### Native Perturbation Support
Cell2Sen has built-in `get_perturbations()` method that:
- Takes text-based perturbation descriptions
- Generates new cell sentences representing perturbed states
- Uses LLM (Gemma) to predict gene expression changes

### Generative Approach
Unlike deterministic perturbation methods:
- **Geneformer**: Moves genes to front of tokenized sequence
- **scGPT**: Multiplies expression values by fold_change
- **Cell2Sen**: Generates entirely new cell sentences via LLM

## Installation

Cell2Sen is included in Helical. Ensure you have the latest version:

```bash
pip install helical
# or
pip install --upgrade git+https://github.com/helicalAI/helical.git
```

## Quick Start

### Basic Usage

```python
from helical.models.c2s import Cell2Sen, Cell2SenConfig
from cellreprogrammer.src.adapters import Cell2SenAdapter
import anndata as ad

# 1. Initialize model
config = Cell2SenConfig(
    batch_size=16,
    model_size="2B",  # or "27B" for larger model
    use_quantization=False  # Set True to reduce memory
)
model = Cell2Sen(config)

# 2. Create adapter
adapter = Cell2SenAdapter(model, config)

# 3. Load and process data
adata = ad.read_h5ad("your_data.h5ad")
dataset = adapter.process_data(adata)

# 4. Apply perturbation (text-based)
perturbed_dataset = adapter.apply_perturbation(
    dataset,
    genes_to_perturb=["POU5F1", "SOX2", "KLF4", "MYC"],
    perturbation_type="overexpress",
    fold_change=2.0  # Optional, for display
)

# 5. Extract embeddings from perturbed sentences
perturbed_embeddings = adapter.extract_perturbed_embeddings(perturbed_dataset)
```

### Running Full Experiment

Use the provided script for complete experiments with controls:

```bash
python cellreprogrammer/c2s/run_perturbation.py \
    --data data/prepared_data.h5ad \
    --genes POU5F1 SOX2 KLF4 MYC \
    --random GAPDH ACTB B2M \
    --output results/c2s_experiment \
    --start-state Fibroblast \
    --goal-state iPSC \
    --model-size 2B
```

### Command-Line Options

```bash
python cellreprogrammer/c2s/run_perturbation.py --help
```

**Required:**
- `--data`: Path to AnnData file
- `--genes`: Genes to perturb (space-separated)
- `--output`: Output directory

**Optional:**
- `--random`: Random control genes (default: GAPDH ACTB B2M MT-ATP6)
- `--start-state`: Starting cell state (default: Fibroblast)
- `--goal-state`: Goal cell state (default: iPSC)
- `--max-cells`: Maximum cells to use (default: all)
- `--perturbation-type`: "overexpress" or "knockdown" (default: overexpress)
- `--fold-change`: Fold change for display (optional)
- `--model-size`: "2B" or "27B" (default: 2B)
- `--use-quantization`: Use 4-bit quantization (reduces memory)

## How It Works

### 1. Data Processing
Cell2Sen converts expression data to "cell sentences":
- Genes are ranked by expression (highest = rank 1)
- Sentence format: "GENE1 GENE2 GENE3 ..." (space-separated)
- Example: "POU5F1 SOX2 NANOG KLF4 MYC ..."

### 2. Perturbation
Perturbations are described in natural language:
- Single gene: `"overexpress POU5F1"`
- Multiple genes: `"overexpress POU5F1, SOX2, KLF4, and MYC"`
- With fold change: `"overexpress POU5F1 by 2.0x"`

### 3. Generation
The LLM generates a new cell sentence representing the perturbed state:
- Input: Control cell sentence + perturbation description
- Output: Predicted perturbed cell sentence
- Example: `"POU5F1 SOX2 NANOG KLF4 MYC ..."` → `"POU5F1 SOX2 KLF4 MYC NANOG ..."`

### 4. Embedding Extraction
The generated sentence is converted back to embeddings:
- Process perturbed sentence through model
- Extract cell embeddings
- Compare with baseline and goal states

## Model Sizes

### Cell2Sen-2B
- **Parameters**: 2 billion
- **Memory**: ~4-8 GB GPU
- **Speed**: Faster inference
- **Use case**: Quick experiments, development

### Cell2Sen-27B
- **Parameters**: 27 billion
- **Memory**: ~50+ GB GPU (or use quantization)
- **Speed**: Slower but more accurate
- **Use case**: Production experiments, best results

### Quantization
Use 4-bit quantization to reduce memory:
```python
config = Cell2SenConfig(
    model_size="27B",
    use_quantization=True  # Reduces memory by ~4x
)
```

## Comparison with Other Models

| Feature | Geneformer | scGPT | Cell2Sen |
|---------|-----------|-------|----------|
| **Perturbation Method** | Token reordering | Expression multiplication | Text generation |
| **Input Format** | Ensembl IDs | Gene symbols | Gene symbols |
| **Output** | Modified tokens | Modified expression | Generated sentence |
| **Deterministic** | Yes | Yes | No (generative) |
| **Native Support** | Yes (InSilicoPerturber) | Yes (expression mod) | Yes (get_perturbations) |

## Advantages of Cell2Sen

1. **Generative**: Predicts entirely new cell states, not just modifications
2. **Flexible**: Natural language perturbations allow complex descriptions
3. **Large-scale**: Trained on 50M+ cells
4. **Cross-species**: Supports human and mouse
5. **LLM-powered**: Leverages language model capabilities

## Limitations

1. **Non-deterministic**: Generative nature means results may vary
2. **Computational**: LLM inference is slower than deterministic methods
3. **Memory**: Large models require significant GPU memory
4. **Text-based**: Requires gene symbols, not Ensembl IDs

## Tips for Best Results

1. **Use appropriate model size**: 2B for quick tests, 27B for production
2. **Enable quantization**: If memory is limited
3. **Batch size**: Adjust based on GPU memory (default: 16)
4. **Gene symbols**: Ensure data uses gene symbols, not Ensembl IDs
5. **Valid perturbations**: Some cells may not generate valid perturbed sentences (handled automatically)

## Troubleshooting

### Out of Memory
```python
# Use quantization
config = Cell2SenConfig(use_quantization=True)

# Or use smaller model
config = Cell2SenConfig(model_size="2B")

# Or reduce batch size (default is 1, which is safest)
config = Cell2SenConfig(batch_size=1)

# Or reduce genes per cell
# Use --max-genes-per-cell 2000 or lower
```

### No Valid Perturbed Sentences
- Check that genes are in gene symbol format
- Verify genes exist in your data
- Some cells may fail generation (handled by adapter)

### Slow Performance
- Use 2B model instead of 27B
- Enable quantization
- Keep batch_size=1 (default, safest)
- Use GPU if available
- Reduce max_genes_per_cell (fewer genes = faster)

### Negative Shifts
If you see negative shifts (cells moving AWAY from goal):

1. **Check goal state**: Verify iPSC embeddings are correct
   ```python
   # Inspect goal embeddings
   goal_embeddings = adapter.extract_embeddings(goal_dataset)
   print(f"Goal embeddings shape: {goal_embeddings.shape}")
   print(f"Goal embeddings mean: {goal_embeddings.mean(axis=0)[:10]}")
   ```

2. **Validate perturbation**: Check if generated sentences make sense
   ```python
   # Inspect generated sentences
   perturbed_sentences = perturbed_dataset['perturbed_cell_sentence']
   print(f"First perturbed sentence: {perturbed_sentences[0][:200]}...")
   ```

3. **Try different approach**: 
   - Use different perturbation description
   - Try smaller gene sets
   - Compare with Geneformer/scGPT results

4. **Understand generative nature**: Cell2Sen predicts what *might* happen, not what *should* happen. Results may differ from deterministic models.

### Warnings

**Duplicate Variable Names**:
- Automatically handled by calling `var_names_make_unique()`
- Safe to ignore if you see the warning briefly

**CUDAGraph Dynamic Shapes**:
- Performance optimization warning, not an error
- Automatically suppressed in the script
- Safe to ignore

**TensorFloat32**:
- Performance optimization warning, not an error  
- Automatically suppressed in the script
- Safe to ignore

## Understanding the Perturbation Pipeline

### What Happens Step-by-Step

1. **Data Preparation**
   - Loads AnnData with single-cell expression data
   - Filters to starting cell state (e.g., Fibroblast)
   - Converts Ensembl IDs → gene symbols (Cell2Sen requires symbols)
   - Filters to top N highly variable genes (default: 5000, configurable)

2. **Cell Sentence Creation**
   - Cell2Sen converts expression data to "cell sentences"
   - Genes are ranked by expression (highest = rank 1)
   - Format: "GENE1 GENE2 GENE3 ..." (space-separated gene symbols)
   - Example: "POU5F1 SOX2 NANOG KLF4 MYC ..."

3. **Baseline Embedding Extraction**
   - Processes cell sentences through Cell2Sen model
   - Extracts cell embeddings (one per cell)
   - Computes goal state centroid (mean of iPSC embeddings)

4. **Perturbation Generation**
   - Creates text description: "overexpress POU5F1, SOX2, KLF4, and MYC"
   - Uses Cell2Sen's `get_perturbations()` method
   - LLM generates NEW cell sentences representing perturbed state
   - Example: Original "GENE1 GENE2 GENE3..." → Perturbed "POU5F1 SOX2 GENE1 GENE3..."

5. **Perturbed Embedding Extraction**
   - Processes generated perturbed sentences through model
   - Extracts embeddings from perturbed sentences
   - Compares with baseline and goal state

6. **Shift Calculation**
   - Computes cosine distance: baseline → goal, perturbed → goal
   - Shift = baseline_distance - perturbed_distance
   - **Positive shift**: Moved TOWARD goal (good!)
   - **Negative shift**: Moved AWAY from goal (may indicate issue or unexpected behavior)

### Understanding Negative Shifts

**Why might shifts be negative?**

1. **Generative Model Behavior**: Cell2Sen is generative - it predicts what *might* happen, not what *should* happen. The LLM may generate cell states that are biologically plausible but not necessarily closer to the goal.

2. **Model Limitations**: The model was trained on diverse data but may not perfectly capture reprogramming dynamics. Generative models can produce unexpected outputs.

3. **Goal State Representation**: The goal centroid (mean of iPSC embeddings) may not be the optimal reference point. Consider:
   - Using a more specific iPSC subpopulation
   - Using a different goal state representation
   - Validating goal embeddings make biological sense

4. **Perturbation Description**: The text-based perturbation ("overexpress X") may not translate perfectly to the model's understanding. The model interprets this through its training, which may differ from expected behavior.

**What to do if shifts are negative:**

- Check if both target and random are negative (both moving away)
- Compare the magnitude: if target is less negative than random, that's still improvement
- Try different perturbation descriptions
- Validate that goal state embeddings are correct
- Consider that generative models may need different interpretation than deterministic models

### Example Output

```
================================================================================
Cell2Sen (C2S) Perturbation Experiment
================================================================================

Model: Cell2Sen-2B
Perturbation type: overexpress

✓ GPU detected: NVIDIA A100
✓ Model initialized
  Using batch_size=1 (processing 1 cell(s) at a time)

Loading data...
✓ Loaded: 32138 cells × 32738 genes
✓ Filtered to Fibroblast: 7079 cells
⚠ Limiting to 50 cells

Extracting goal state embeddings (iPSC)...
⚠ Filtering to top 1000 genes per cell (to avoid OOM)
  Original: 23897 genes
  Filtered: 1000 genes
✓ Goal state embeddings: (50, 2304)

Processing baseline data...
✓ Baseline embeddings: (50, 2304)

Generating perturbed cell sentences for: POU5F1, SOX2, KLF4, MYC
  (This uses Cell2Sen's generative LLM to predict perturbed states)
✓ Valid perturbed embeddings: 50 / 50
✓ Mean shift: -0.583854 ± 0.030438

Generating perturbed cell sentences for random control: GAPDH, ACTB, B2M
✓ Random mean shift: -0.579106 ± 0.031102

================================================================================
RESULTS SUMMARY
================================================================================

When POU5F1, SOX2, KLF4, MYC were overexpress:
  • Mean shift toward iPSC: -0.583854 ± 0.030438
  • Random controls shift: -0.579106 ± 0.031102

⚠ Both target and random perturbations moved cells AWAY from iPSC
  However, target genes (POU5F1, SOX2, KLF4, MYC) moved less away than random controls
  Improvement: -0.004748 (target is 0.004748 closer than random)
  Note: Negative shifts may indicate the perturbation approach needs adjustment

Results saved to: results/
```

### Interpreting Results

**Positive Shifts (Good)**:
- `+0.05` or higher: Strong positive effect
- `+0.01` to `+0.05`: Moderate positive effect
- `+0.001` to `+0.01`: Weak but positive effect

**Negative Shifts (Concerning)**:
- Both target and random negative: May indicate model/approach issue
- Target less negative than random: Still improvement, but weak
- Target more negative than random: Unexpected, investigate

**Key Metrics**:
- **Improvement**: `target_mean - random_mean` (higher is better)
- **Fold improvement**: `target_mean / random_mean` (only when both same sign)
- **Standard deviation**: Lower = more consistent effects

## Citation

If you use Cell2Sen, please cite:

```bibtex
@article{rizvi2025cell2sen,
  title={Scaling Large Language Models for Next-Generation Single-Cell Analysis},
  author={Rizvi, Syed Asad and Levine, Daniel and Patel, Aakash and Zhang, Shiyang and Wang, Eric and Perry, Curtis Jamison and Constante, Nicole and He, Sizhuang and Zhang, David and Tang, Cerise and Lyu, Zhuoyang and Darji, Rayyan and Li, Chang and Sun, Emily and Jeong, David and Zhao, Lawrence and Kwan, Jennifer and Braun, David and Hafler, Brian and Chung, Hattie and Dhodapkar, Rahul M. and Perozzi, Bryan and Ishizuka, Jeffrey and Azizi, Shekoofeh and van Dijk, David},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.04.14.648850}
}
```

## Further Reading

- [Cell2Sen Model Card](../../docs/model_cards/c2s.md)
- [Cell2Sen Tutorial Notebook](../../docs/notebooks/Cell2Sen-Tutorial.ipynb)
- [CellReprogrammer Main README](../README.md)

