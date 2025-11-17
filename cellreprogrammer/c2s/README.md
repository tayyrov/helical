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

# Or reduce batch size
config = Cell2SenConfig(batch_size=8)
```

### No Valid Perturbed Sentences
- Check that genes are in gene symbol format
- Verify genes exist in your data
- Some cells may fail generation (handled by adapter)

### Slow Performance
- Use 2B model instead of 27B
- Enable quantization
- Reduce batch size
- Use GPU if available

## Example Output

```
================================================================================
Cell2Sen (C2S) Perturbation Experiment
================================================================================

Model: Cell2Sen-2B
Perturbation type: overexpress

✓ GPU detected: NVIDIA A100
✓ Model initialized

Loading data...
✓ Loaded: 1000 cells × 2000 genes
✓ Filtered to Fibroblast: 500 cells
✓ Using all 500 cells

Extracting goal state embeddings (iPSC)...
✓ Goal state embeddings: (300, 2048)

Processing baseline data...
✓ Baseline embeddings: (500, 2048)

Generating perturbed cell sentences for: POU5F1, SOX2, KLF4, MYC
  (This uses Cell2Sen's generative LLM to predict perturbed states)
Extracting embeddings from perturbed sentences...
✓ Valid perturbed embeddings: 485 / 500
✓ Mean shift: +0.023456 ± 0.012345

Generating perturbed cell sentences for random control: GAPDH, ACTB, B2M
Extracting embeddings from random control perturbed sentences...
✓ Random mean shift: +0.001234 ± 0.008765

================================================================================
RESULTS SUMMARY
================================================================================

When POU5F1, SOX2, KLF4, MYC were overexpressed 2.0x:
  • Mean shift toward iPSC: +0.023456 ± 0.012345
  • Random controls shift: +0.001234 ± 0.008765

✓ Target genes showed 19.00x better shift toward iPSC
  compared to random controls (GAPDH, ACTB, B2M)

Results saved to: results/c2s_experiment
```

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

