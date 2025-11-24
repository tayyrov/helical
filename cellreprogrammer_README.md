# Cellreprogrammer

Cell reprogramming experiments using foundation models from [helical](https://github.com/helicalAI/helical).

## Overview

This repository contains experiments for evaluating single-cell foundation models on:
- **OSKM reprogramming**: Testing Yamanaka factors (OCT4, SOX2, KLF4, MYC)
- **Norman dataset**: CRISPRa perturbation prediction in K562 cells
- **Perturbation benchmarking**: Integration with PertEval for standardized evaluation

## Installation

### Prerequisites
- Python ≥3.10
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cellreprogrammer
cd cellreprogrammer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install helical and dependencies
pip install helical
pip install -r requirements.txt

# Install cellreprogrammer in editable mode (optional)
pip install -e .
```

## Quick Start

### OSKM Reprogramming Experiment

```bash
cd experiments/oskm
python prepare_reprogramming_data.py
python test_oskm_combinations.py
```

### Norman Dataset Evaluation

```bash
cd experiments/norman

# Step 1: Prepare data
python prepare_norman_data.py

# Step 2: Evaluate Geneformer
python evaluate_norman_geneformer.py

# Results will be saved to ../../results/norman_evaluation/
```

## Project Structure

```
cellreprogrammer/
├── experiments/
│   ├── oskm/              # OSKM factor experiments
│   └── norman/            # Norman dataset experiments
├── benchmarks/            # PertEval integration
├── data/                  # Data directory (not in git)
└── results/               # Results directory (not in git)
```

## Data

Data should be placed in the `data/` directory:
- `data/raw/`: Raw input files
- `data/prepared/`: Processed h5ad files
- `data/tokenized/`: Tokenized datasets for models

> **Note**: Large data files are not tracked in git. Configure data paths via environment variable if needed:
> ```bash
> export CELLREPROGRAMMER_DATA="/path/to/data"
> ```

## Benchmarking with PertEval

For standardized perturbation evaluation:

```bash
# Setup (one-time)
cd benchmarks
./setup_perteval.sh

# Extract embeddings for PertEval
python perteval_adapters/extract_embeddings.py --model geneformer

# Run PertEval benchmark
cd ~/path/to/PertEval
python src/train.py experiment=mlp_norman_train.yaml model=geneformer
```

See [benchmarks/README.md](benchmarks/README.md) for details.

## Available Models

This project uses models from [helical](https://github.com/helicalAI/helical):
- **Geneformer** (v1, v2, v3)
- **scGPT**
- **UCE**
- **scBERT**
- **scFoundation**

## Citation

If you use this code, please cite:
- The helical package
- The original model papers (Geneformer, scGPT, etc.)
- The Norman et al. 2019 dataset (if used)

## License

MIT License - see LICENSE file for details



