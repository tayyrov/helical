# Supported Models in CellReprogrammer

This document provides an overview of all models supported in CellReprogrammer, their perturbation methods, and use cases.

## Model Comparison

| Model | Perturbation Method | Input Format | Output | Deterministic | Native Support |
|-------|-------------------|--------------|--------|---------------|----------------|
| **Geneformer** | Token reordering | Ensembl IDs | Modified tokens | Yes | ✅ InSilicoPerturber |
| **scGPT** | Expression multiplication | Gene symbols | Modified expression | Yes | ✅ Expression mod |
| **Cell2Sen** | Text generation | Gene symbols | Generated sentence | No (generative) | ✅ get_perturbations |

## Geneformer

### Overview
Transformer-based model for single-cell RNA-seq data. Uses rank-value encoding where genes are ranked by expression within each cell.

### Perturbation Method
- **Native utility**: `InSilicoPerturber`
- **Mechanism**: Moves target genes to the front of the tokenized sequence
- **Input**: Ensembl IDs
- **Output**: Modified tokenized sequences

### Use Cases
- Cell state transitions
- Disease modeling
- Therapeutic target identification
- Transcription factor analysis

### Example
```python
from cellreprogrammer.src.adapters import GeneformerAdapter

adapter = GeneformerAdapter(model, config)
# Uses native InSilicoPerturber via run_perturbation_experiment()
```

### Documentation
- See `geneformer/` directory
- [Geneformer Model Card](../../docs/model_cards/geneformer.md)

## scGPT

### Overview
GPT-style transformer model pre-trained on 33+ million human cells. Uses expression binning and tokenization.

### Perturbation Method
- **Native support**: Expression modification
- **Mechanism**: Multiplies expression values by `fold_change`
- **Input**: Gene symbols
- **Output**: Modified expression matrix (re-processed)

### Use Cases
- Cell type annotation
- Perturbation response prediction
- Batch integration
- Multi-omic integration

### Example
```python
from cellreprogrammer.src.adapters import scGPTAdapter

adapter = scGPTAdapter(model, config)
perturbed_dataset = adapter.apply_perturbation(
    dataset,
    genes_to_perturb=["POU5F1", "SOX2"],
    perturbation_type="overexpress",
    fold_change=2.0
)
```

### Documentation
- See `scgpt/` directory
- [scGPT Model Card](../../docs/model_cards/scgpt.md)

## Cell2Sen (C2S)

### Overview
Large language model (LLM) based foundation model that transforms single-cell profiles into textual "cell sentences". Uses Gemma architecture.

### Perturbation Method
- **Native support**: `get_perturbations()` method
- **Mechanism**: Text-based perturbation descriptions → LLM generates new cell sentences
- **Input**: Gene symbols (as text descriptions)
- **Output**: Generated perturbed cell sentences

### Key Features
- **Generative**: Predicts entirely new cell states
- **Text-based**: Natural language perturbations
- **Large-scale**: Trained on 50M+ cells
- **Flexible**: Can describe complex perturbations

### Use Cases
- Generative perturbation prediction
- Text-based experimental design
- Cross-species analysis (human, mouse)
- Hypothesis generation

### Example
```python
from cellreprogrammer.src.adapters import Cell2SenAdapter

adapter = Cell2SenAdapter(model, config)
perturbed_dataset = adapter.apply_perturbation(
    dataset,
    genes_to_perturb=["POU5F1", "SOX2", "KLF4", "MYC"],
    perturbation_type="overexpress",
    fold_change=2.0  # Optional, for display
)

# Extract embeddings from generated sentences
perturbed_embeddings = adapter.extract_perturbed_embeddings(perturbed_dataset)
```

### Model Sizes
- **2B**: Faster, lower memory (~4-8 GB GPU)
- **27B**: More accurate, higher memory (~50+ GB GPU, or use quantization)

### Documentation
- See `c2s/README.md` for detailed guide
- [Cell2Sen Model Card](../../docs/model_cards/c2s.md)

## Choosing the Right Model

### Use Geneformer if:
- You have Ensembl IDs
- You need deterministic results
- You're working with cell state transitions
- You need transcription factor analysis

### Use scGPT if:
- You have gene symbols
- You want expression-based perturbations
- You need batch integration
- You're doing cell type annotation

### Use Cell2Sen if:
- You want generative predictions
- You prefer text-based perturbations
- You need cross-species support
- You're exploring novel perturbations

## Model Integration

All models are integrated via the adapter pattern:

```python
from cellreprogrammer.src.models.model_factory import ModelFactory
from cellreprogrammer.src.adapters import (
    GeneformerAdapter,
    scGPTAdapter,
    Cell2SenAdapter
)

factory = ModelFactory()

# Load any model
model = factory.load_model("geneformer")  # or "scgpt", "c2s"

# Create appropriate adapter
if model_name == "geneformer":
    adapter = GeneformerAdapter(model, config)
elif model_name == "scgpt":
    adapter = scGPTAdapter(model, config)
elif model_name == "c2s":
    adapter = Cell2SenAdapter(model, config)
```

## Future Models

Models under consideration:
- **UCE**: Universal Cell Embedding (would need manual perturbation)
- **TranscriptFormer**: Generative model (explicitly not for perturbation)
- **Helix-mRNA**: mRNA-level model (different use case)

## Contributing New Models

To add a new model:

1. Create adapter in `src/adapters/`
2. Add to `ModelFactory.MODEL_REGISTRY`
3. Create `run_perturbation.py` script
4. Add documentation
5. Update this file

See `ARCHITECTURE.md` for detailed integration guide.

