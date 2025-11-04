# Geneformer Experiments

This folder contains Geneformer-specific experiment scripts for cell reprogramming.

## Scripts

- **`prepare_reprogramming_data.py`**: Prepares and tokenizes data for Geneformer models (V1 and V2/V3). Creates tokenized datasets for both model versions.

- **`reproduce_reprogramming.py`**: Reproduces the OSKM reprogramming experiment from Theodoris et al. Nature 2023 using a single Geneformer model.

- **`compare_all_models.py`**: Compares all available Geneformer models (V1, V2, V3) for the OSKM reprogramming task and identifies the best performing model.

- **`test_oskm_combinations.py`**: Tests all 15 combinations of OSKM factors (OCT4, SOX2, KLF4, MYC) to identify optimal factor combinations.

## Usage

```bash
# Prepare data for Geneformer
python prepare_reprogramming_data.py

# Run single model experiment
python reproduce_reprogramming.py

# Compare all models
python compare_all_models.py

# Test all OSKM combinations
python test_oskm_combinations.py
```

## Dependencies

These scripts require the original Geneformer package for advanced perturbation utilities:
- `InSilicoPerturber`
- `EmbExtractor`
- `InSilicoPerturberStats`

Install with: `pip install geneformer` or use the local repo at `/home/ubuntu/data-at-virginia/Geneformer/`
