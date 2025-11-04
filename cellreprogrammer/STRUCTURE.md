# CellReprogrammer Structure

This document explains the organization of the CellReprogrammer framework.

## Directory Structure

```
cellreprogrammer/
├── data_preparation/          # Model-agnostic data preparation
│   ├── convert_geo_data.py   # Convert GEO datasets to AnnData
│   └── README.md
│
├── experiments/               # Generic/unified experiment scripts
│   ├── unified_perturbation_experiment.py  # Routes to model-specific scripts
│   └── README.md
│
├── geneformer/               # Geneformer-specific code
│   ├── run_perturbation.py   # Main perturbation function (importable)
│   ├── experiments/          # Geneformer-specific experiment scripts
│   │   ├── prepare_reprogramming_data.py
│   │   ├── reproduce_reprogramming.py
│   │   ├── compare_all_models.py
│   │   ├── test_oskm_combinations.py
│   │   └── README.md
│   └── __init__.py
│
├── scgpt/                    # scGPT-specific code
│   ├── run_perturbation.py   # Main perturbation function (importable)
│   └── __init__.py
│
├── examples/                 # Example/utility scripts
│   └── example_overexpression.py
│
└── src/                      # Core framework (shared utilities)
    ├── adapters/             # Model adapters (PerturbationAdapter base)
    └── utils/                # Shared utilities (comparison, etc.)
```

## Organization Principles

### 1. **Model-Specific Code in Model Folders**
   - Each model has its own folder (`geneformer/`, `scgpt/`)
   - Model-specific scripts are in `model_name/experiments/`
   - Model-specific functions (like `run_perturbation.py`) are directly in `model_name/`

### 2. **Generic Code in Shared Folders**
   - Data preparation scripts → `data_preparation/`
   - Unified experiment runners → `experiments/`
   - Core framework → `src/`

### 3. **Clear Naming**
   - Scripts have descriptive names (no numbered prefixes)
   - Model-specific scripts clearly indicate which model they're for
   - Folders indicate purpose (experiments, data_preparation, etc.)

## File Naming Changes

| Old Name | New Location | Reason |
|----------|-------------|--------|
| `01_convert_geo_data.py` | `data_preparation/convert_geo_data.py` | Generic data prep, not model-specific |
| `02_prepare_reprogramming_data.py` | `geneformer/experiments/prepare_reprogramming_data.py` | Geneformer-specific tokenization |
| `03_reproduce_reprogramming.py` | `geneformer/experiments/reproduce_reprogramming.py` | Geneformer-specific experiment |
| `04_compare_all_models.py` | `geneformer/experiments/compare_all_models.py` | Compares Geneformer models |
| `05_test_all_oskm_combinations.py` | `geneformer/experiments/test_oskm_combinations.py` | Geneformer-specific |
| `06_unified_perturbation_experiment.py` | `experiments/unified_perturbation_experiment.py` | Generic router |

## Usage

### Data Preparation (Model-Agnostic)
```bash
cd cellreprogrammer/data_preparation
python convert_geo_data.py
```

### Geneformer Experiments
```bash
cd cellreprogrammer/geneformer/experiments
python prepare_reprogramming_data.py
python reproduce_reprogramming.py
python compare_all_models.py
python test_oskm_combinations.py
```

### Unified Experiments (Cross-Model)
```bash
cd cellreprogrammer/experiments
python unified_perturbation_experiment.py --model geneformer --genes OCT4 SOX2 KLF4 MYC
python unified_perturbation_experiment.py --model scgpt --genes OCT4 SOX2 KLF4 MYC
```

### Model-Specific Functions (Importable)
```python
from cellreprogrammer.geneformer import run_perturbation_experiment
from cellreprogrammer.scgpt import run_perturbation_experiment

# Use directly
run_perturbation_experiment(...)
```

## Benefits of This Structure

1. **Clear Separation**: Model-specific code is isolated from generic code
2. **Easy Navigation**: Folder names indicate purpose and model
3. **Scalable**: Adding new models is straightforward (create new folder)
4. **No Duplication**: Shared utilities in `src/`, model-specific code in model folders
5. **Better Naming**: Descriptive names instead of numbers

## Adding a New Model

To add support for a new model (e.g., TranscriptFormer):

1. Create `transcriptformer/` folder
2. Add `transcriptformer/run_perturbation.py` with `run_perturbation_experiment()` function
3. Create `transcriptformer/experiments/` if needed for model-specific experiments
4. Update `experiments/unified_perturbation_experiment.py`:
   ```python
   from transcriptformer.run_perturbation import run_perturbation_experiment as run_transcriptformer_perturbation
   
   MODEL_ROUTER = {
       ...
       "transcriptformer": run_transcriptformer_perturbation,
   }
   ```

## Migration Notes

- All scripts have been moved but paths inside them remain the same (they use `CELLREPROGRAMMER_DIR` which resolves correctly)
- Import statements may need updating if scripts import from each other
- The unified script imports from the new locations automatically
