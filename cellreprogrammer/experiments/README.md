# Unified Experiments

This folder contains generic/unified experiment scripts that work across multiple models.

## Scripts

- **`unified_perturbation_experiment.py`**: Unified perturbation experiment runner that routes to model-specific implementations. Supports both Geneformer and scGPT (and can be extended to other models).

## Usage

```bash
# Run with Geneformer
python unified_perturbation_experiment.py --model geneformer --genes OCT4 SOX2 KLF4 MYC

# Run with scGPT
python unified_perturbation_experiment.py --model scgpt --genes OCT4 SOX2 KLF4 MYC --fold-change 2.0
```

## Model-Specific Scripts

For model-specific experiments, see:
- **Geneformer**: `../geneformer/experiments/`
- **scGPT**: `../scgpt/run_perturbation.py` (can be run standalone)

## Data Preparation

For data preparation scripts, see:
- **Generic data conversion**: `../data_preparation/`

