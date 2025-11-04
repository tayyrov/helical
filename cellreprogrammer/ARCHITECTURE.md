# CellReprogrammer Architecture

## Overview

CellReprogrammer is designed as a modular framework for running in-silico perturbation experiments across multiple foundation models. The architecture allows easy addition of new models while maintaining consistent interfaces.

## File Organization

```
cellreprogrammer/
├── experiments/              # Experiment scripts
│   ├── 01_convert_geo_data.py
│   ├── 02_prepare_reprogramming_data.py
│   ├── 03_reproduce_reprogramming.py      # Geneformer-specific
│   ├── 04_compare_all_models.py           # Geneformer comparison
│   ├── 05_test_all_oskm_combinations.py   # Geneformer combinations
│   └── 06_unified_perturbation_experiment.py  # Multi-model unified
│
├── src/
│   ├── adapters/            # Model-specific adapters
│   │   ├── base_adapter.py       # Abstract base class
│   │   ├── geneformer_adapter.py # Geneformer adapter (uses original utilities)
│   │   └── scgpt_adapter.py      # scGPT adapter (generic perturbation)
│   │
│   ├── perturbations/       # Perturbation implementations (legacy)
│   └── models/              # Model utilities
│
├── configs/                 # Configuration files
├── data/                    # Data (gitignored)
│   ├── raw/
│   ├── prepared/
│   └── tokenized/
├── results/                 # Results (gitignored)
│   ├── oskm_experiment/
│   ├── model_comparison/
│   └── unified_perturbation/
│
└── learning_concepts/       # Educational docs (gitignored)

```

## Architecture Components

### 1. Adapter Pattern

The core architecture uses the **Adapter Pattern** to provide a unified interface across different models:

```
┌─────────────────────────────────────────────────────────┐
│           Unified Experiment Runner                      │
│    (06_unified_perturbation_experiment.py)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              PerturbationAdapter (Abstract)             │
│  - process_data()                                       │
│  - extract_embeddings()                                 │
│  - apply_perturbation()                                 │
│  - compute_shift()                                      │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│ GeneformerAdapter│    │   scGPTAdapter   │
│                  │    │                  │
│ Uses original    │    │ Generic          │
│ InSilicoPerturber│    │ perturbation     │
└──────────────────┘    └──────────────────┘
```

### 2. Model-Specific Adapters

Each adapter implements the `PerturbationAdapter` interface:

- **GeneformerAdapter**: Wraps original `InSilicoPerturber` utilities (token sequence manipulation)
- **scGPTAdapter**: Generic perturbation by modifying AnnData expression values
- **TranscriptFormerAdapter**: (To be implemented) Similar to scGPT

### 3. Perturbation Strategies

Different models use different perturbation strategies:

#### Geneformer
- **Strategy**: Modify tokenized sequences
- **Method**: Move gene tokens to front of sequence (simulates high expression)
- **Tools**: Original `InSilicoPerturber` from Geneformer package

#### scGPT / TranscriptFormer
- **Strategy**: Modify expression values in AnnData
- **Method**: Multiply expression by fold-change (e.g., 2x for overexpression)
- **Tools**: Generic implementation in adapter

## Adding a New Model

To add a new model (e.g., TranscriptFormer, UCE):

### Step 1: Create Adapter

Create `cellreprogrammer/src/adapters/transcriptformer_adapter.py`:

```python
from .base_adapter import PerturbationAdapter

class TranscriptFormerAdapter(PerturbationAdapter):
    def process_data(self, adata, **kwargs):
        # Implement model-specific data processing
        return self.model.process_data(adata, **kwargs)
    
    def extract_embeddings(self, dataset, **kwargs):
        # Implement embedding extraction
        return self.model.get_embeddings(dataset, **kwargs)
    
    def apply_perturbation(self, dataset, genes_to_perturb, ...):
        # Implement perturbation (modify AnnData or dataset)
        # Then re-process and return
        pass
    
    def get_gene_mapping(self):
        # Return gene identifier mapping
        pass
```

### Step 2: Register Model

Add to `06_unified_perturbation_experiment.py`:

```python
MODEL_REGISTRY = {
    # ... existing models ...
    "transcriptformer": {
        "model_class": TranscriptFormer,
        "config_class": TranscriptFormerConfig,
        "adapter_class": TranscriptFormerAdapter,
        "default_config": {"batch_size": 50},
    },
}
```

### Step 3: Update Imports

Update `cellreprogrammer/src/adapters/__init__.py`:

```python
from .transcriptformer_adapter import TranscriptFormerAdapter
```

### Step 4: Test

Run experiment:

```bash
python cellreprogrammer/experiments/06_unified_perturbation_experiment.py \
    --model transcriptformer \
    --genes OCT4 SOX2 KLF4 MYC \
    --random GAPDH ACTB
```

## Experiment Scripts

### Geneformer-Specific Scripts

- **`03_reproduce_reprogramming.py`**: Single model, OSKM vs random
- **`04_compare_all_models.py`**: Compare all Geneformer models
- **`05_test_all_oskm_combinations.py`**: Test all OSKM combinations

### Unified Scripts

- **`06_unified_perturbation_experiment.py`**: Multi-model unified experiment runner

## Data Flow

### Generic Models (scGPT, TranscriptFormer)

```
AnnData → process_data() → Dataset
                           ↓
                    baseline embeddings
                           ↓
Modify AnnData expression → process_data() → Dataset
                           ↓
                    perturbed embeddings
                           ↓
Compare to goal state → compute_shift()
```

### Geneformer (Special Case)

```
Tokenized Dataset → InSilicoPerturber → Modified sequences
                                        ↓
                            Model forward pass
                                        ↓
                            Perturbed embeddings
                                        ↓
                            Original analysis pipeline
```

## Key Design Decisions

### 1. Why Adapters?

- **Unified Interface**: Same experiment code works for all models
- **Model-Specific Logic**: Each adapter handles model quirks
- **Easy Extension**: Add new models without changing existing code

### 2. Why Two Perturbation Strategies?

- **Geneformer**: Has sophisticated original utilities (token manipulation)
- **Others**: Simpler generic approach (expression modification)

### 3. Why Separate Scripts?

- **Geneformer**: Original scripts work well, keep for compatibility
- **Unified**: New script for multi-model comparison

## Future Improvements

1. **Add more models**: TranscriptFormer, UCE
2. **Unified comparison**: Script to compare all models side-by-side
3. **Perturbation types**: Knockdown, knockout, in addition to overexpression
4. **Batch processing**: Run experiments for multiple gene sets
5. **Visualization**: Unified plotting across models

## Questions?

See `learning_concepts/` folder for educational materials explaining:
- Model weights vs embeddings
- Geneformer file structure
- In-silico perturbation mechanics
