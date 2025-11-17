# CellReprogrammer

**Genetic Perturbation and Cell Reprogramming Framework**

A specialized toolkit built on top of [Helical](https://github.com/helicalAI/helical) for conducting genetic perturbation experiments and cell reprogramming studies using state-of-the-art foundation models.

## Overview

CellReprogrammer provides a unified framework for:
- 🧬 **Genetic Perturbations**: Overexpression, knockdown, and other perturbations
- 🔬 **Multi-Model Support**: Works with Geneformer, scGPT, and other models via Helical
- 📊 **Standardized APIs**: Easy-to-use interfaces for common experiments
- 🚀 **Extensible Design**: Add new perturbation types and models easily

## Why CellReprogrammer?

While you can use Helical directly for perturbation experiments, CellReprogrammer adds:
- **High-level abstractions** for common perturbation workflows
- **Reusable components** for different experiment types
- **Configuration management** for reproducible experiments
- **Comparison utilities** for analyzing perturbation effects

## Installation

Since CellReprogrammer is built on Helical, ensure Helical is installed first:

```bash
# Install Helical
pip install helical

# Or for the latest version
pip install --upgrade git+https://github.com/helicalAI/helical.git
```

CellReprogrammer is currently a development module within the Helical repository. To use it:

```bash
cd cellreprogrammer
# Add to Python path or install in development mode
pip install -e .  # If you add a setup.py
```

## Quick Start

### Basic Overexpression Experiment (scGPT)

```python
from cellreprogrammer.src.models.model_factory import ModelFactory
from cellreprogrammer.src.adapters import scGPTAdapter
import anndata as ad

# 1. Load a model using the factory
factory = ModelFactory()
model = factory.load_model("scgpt", config_overrides={"batch_size": 10})

# 2. Create adapter
adapter = scGPTAdapter(model, model.config)

# 3. Load your data
ann_data = ad.read_h5ad("your_data.h5ad")

# 4. Process data
dataset = adapter.process_data(ann_data)

# 5. Apply perturbation (overexpress genes)
perturbed_dataset = adapter.apply_perturbation(
    dataset,
    genes_to_perturb=["POU5F1", "SOX2", "KLF4", "MYC"],
    perturbation_type="overexpress",
    fold_change=2.0
)

# 6. Extract embeddings
baseline_embeddings = adapter.extract_embeddings(dataset)
perturbed_embeddings = adapter.extract_embeddings(perturbed_dataset)
print(f"Embeddings shape: {perturbed_embeddings.shape}")
```

### Cell2Sen Generative Perturbation

```python
from cellreprogrammer.src.models.model_factory import ModelFactory
from cellreprogrammer.src.adapters import Cell2SenAdapter
import anndata as ad

# 1. Load Cell2Sen model
factory = ModelFactory()
model = factory.load_model("c2s", config_overrides={"model_size": "2B"})

# 2. Create adapter
adapter = Cell2SenAdapter(model, model.config)

# 3. Load and process data
ann_data = ad.read_h5ad("your_data.h5ad")
dataset = adapter.process_data(ann_data)

# 4. Apply text-based perturbation (generates new cell sentences)
perturbed_dataset = adapter.apply_perturbation(
    dataset,
    genes_to_perturb=["POU5F1", "SOX2", "KLF4", "MYC"],
    perturbation_type="overexpress",
    fold_change=2.0  # Optional, for display in text
)

# 5. Extract embeddings from generated perturbed sentences
perturbed_embeddings = adapter.extract_perturbed_embeddings(perturbed_dataset)
print(f"Perturbed embeddings shape: {perturbed_embeddings.shape}")
```

### Running Full Perturbation Experiments

Use the provided scripts for complete experiments with controls:

```bash
# scGPT perturbation
python cellreprogrammer/scgpt/run_perturbation.py \
    --data data/prepared_data.h5ad \
    --genes POU5F1 SOX2 KLF4 MYC \
    --random GAPDH ACTB B2M \
    --output results/scgpt_experiment \
    --start-state Fibroblast \
    --goal-state iPSC \
    --fold-change 2.0

# Cell2Sen perturbation
python cellreprogrammer/c2s/run_perturbation.py \
    --data data/prepared_data.h5ad \
    --genes POU5F1 SOX2 KLF4 MYC \
    --random GAPDH ACTB B2M \
    --output results/c2s_experiment \
    --start-state Fibroblast \
    --goal-state iPSC \
    --model-size 2B
```

## Project Structure

```
cellreprogrammer/
├── src/
│   ├── models/
│   │   ├── model_factory.py      # Unified model loading interface
│   │   └── __init__.py
│   ├── perturbations/
│   │   ├── base_perturbation.py   # Abstract base class
│   │   ├── overexpression.py      # Overexpression implementation
│   │   ├── knockdown.py           # (TODO) Knockdown implementation
│   │   └── __init__.py
│   └── utils/                     # Utility functions
├── experiments/
│   ├── example_overexpression.py  # Example scripts
│   └── configs/                   # Configuration files
├── notebooks/                     # Jupyter notebooks
├── data/                          # Experiment data (created at runtime)
│   ├── raw/                       # Raw data files
│   ├── prepared/                  # Processed data
│   └── tokenized/                 # Tokenized datasets
├── results/                       # Experiment outputs
└── README.md
```

**Important:** All data and results are stored under the `cellreprogrammer/` directory to keep the helical package clean and uncontaminated.

## Available Models

CellReprogrammer supports multiple models with native perturbation capabilities:

| Model | Perturbation Method | Description | Use Case |
|-------|-------------------|-------------|----------|
| **Geneformer** | Token-based (InSilicoPerturber) | Transformer for scRNA-seq | Cell state transitions, perturbation prediction |
| **scGPT** | Expression modification (fold_change) | GPT-style model for cells | Cell type annotation, perturbation effects |
| **Cell2Sen (C2S)** | Generative text-based | LLM-based generative model | Text-based perturbation, generative predictions |

### Model-Specific Features

- **Geneformer**: Uses native `InSilicoPerturber` utilities that move genes to front of tokenized sequence
- **scGPT**: Modifies expression values directly (multiply by fold_change), then re-processes
- **Cell2Sen**: Uses text-based perturbations (e.g., "overexpress POU5F1") to generate new cell sentences via LLM

Add more models easily in `model_factory.py`!

## Supported Perturbations

### Overexpression (`OverexpressionPerturbation`)
Multiply gene expression values by a factor.

```python
oe = OverexpressionPerturbation(
    model=model,
    perturbation_genes=["GENE1", "GENE2"],
    perturbation_strength=2.0  # 2x overexpression
)
```

### Future Perturbations (Coming Soon)
- **Knockdown**: Reduce gene expression
- **Knockout**: Complete gene removal
- **Promoter Swap**: Modify regulatory elements
- **Multi-gene Programs**: Complex co-expression patterns

## Creating Custom Perturbations

Create your own perturbation by inheriting from `BasePerturbation`:

```python
from cellreprogrammer.src.perturbations.base_perturbation import BasePerturbation

class MyCustomPerturbation(BasePerturbation):
    def apply(self, ann_data):
        # Your perturbation logic here
        modified_data = self.modify_expression(ann_data)
        return self.model.process_data(modified_data)
    
    def get_perturbation_type(self):
        return "custom"
```

## Configuration

Use YAML or Python configs for reproducible experiments:

```yaml
# configs/experiment_config.yaml
model:
  name: "geneformer"
  config:
    model_name: "gf-12L-38M-i4096"
    batch_size: 10
    device: "cuda"

perturbation:
  type: "overexpression"
  genes: ["BRCA1", "TP53", "MYC"]
  strength: 2.5

data:
  path: "data/sample.h5ad"
  n_cells: 1000
```

## Advanced Usage

### Multi-Model Comparison

Compare how different models respond to perturbations:

```python
models_to_test = ["geneformer", "scgpt", "uce"]

for model_name in models_to_test:
    model = factory.load_model(model_name)
    oe = OverexpressionPerturbation(model=model, perturbation_genes=["BRCA1"])
    
    # Your comparison logic here
```

### Batch Experiments

Run experiments across multiple gene sets:

```python
gene_lists = [
    ["BRCA1", "TP53"],
    ["MYC", "EGFR"],
    ["P53", "RB1"]
]

for genes in gene_lists:
    oe = OverexpressionPerturbation(model=model, perturbation_genes=genes)
    # Run experiment
```

## Integration with Existing Workflows

CellReprogrammer integrates seamlessly with:
- **Scanpy/AnnData**: Native AnnData support
- **HuggingFace Datasets**: Load from HuggingFace
- **Existing Helical workflows**: Drop-in replacement for custom code
- **Your Geneformer experiments**: Migrate easily from standalone Geneformer

## Model-Specific Documentation

Each model has detailed documentation:

- **Geneformer**: See `geneformer/` directory
- **scGPT**: See `scgpt/` directory  
- **Cell2Sen**: See `c2s/README.md` for detailed guide

## Future Roadmap

- [x] Cell2Sen integration with native perturbation support
- [ ] Additional perturbation types (knockdown, knockout, etc.)
- [ ] Integration with more models as they're added to Helical
- [ ] Visualization utilities for perturbation effects
- [ ] Statistical analysis tools
- [ ] GPU optimization for large datasets
- [ ] Distributed computing support

## Contributing

We welcome contributions! Whether it's:
- New perturbation types
- Model integrations
- Bug fixes
- Documentation improvements

Please feel free to open an issue or submit a PR!

## Separation Strategy

Currently, CellReprogrammer lives within the Helical repository. If you want to separate it into its own repository while keeping Helical as a dependency:

### Option 1: Git Submodule (Recommended for Active Development)
```bash
# In your CellReprogrammer repo
git submodule add https://github.com/helicalAI/helical.git helical
```

### Option 2: Fork and Maintain
Create your own fork of Helical and maintain it separately.

### Option 3: PyPI Dependency
Once Helical stabilizes, use it as a standard dependency:
```bash
pip install helical
```

## Citation

If you use CellReprogrammer in your research, please cite both CellReprogrammer and Helical:

```bibtex
@software{helical_2024,
  author = {Helical Team},
  title = {helical: A Framework for Bio Foundation Models},
  year = {2024}
}

@software{cellreprogrammer_2024,
  author = {Your Name},
  title = {CellReprogrammer: Genetic Perturbation Framework},
  year = {2024}
}
```

## License

Follows the same license as Helical (see LICENSE file in main directory).

## Support

- **Issues**: [GitHub Issues](https://github.com/helicalAI/helical/issues)
- **Slack**: [Helical Community](https://dk1sxv04.eu1.hubspotlinksfree.com/Ctc/L2+113/dk1sxv04/VWtlqj8M7nFNVf1vhw52bPfMW8wLjj95ptQw7N1k24YY3m2ndW8wLKSR6lZ3ldW7fZmPx5PxJ2lW8mYJtq5xWH5BVsxw821cWpdKW8CYXdj753XHSW8b5vG-7PTQ2LW1zs6x622rZxDW6930hX7RPKh3N5-trBXyRHkwVfJ3Zs3wRQV_N5NbYL3-lm47W1HvYX63pJp9cW6QXY-x6QsWMTW8G5jZh7T4vphN4Qtr7dMCxlJW8rM1-Y42pS-PW5sfJbh4FyRMhW5mHPkD4yCl56W36YW1_4GpPrGW7-sRYG1gXy8hMXqK6Sp5p69W8YTpvd3tC80SW2PTYtr6hP0dxW863B5F4KNCYkVFSWl390bSlQW78rxWn7JbS3LW14ZJ735n7SpFVSVlQr7lm7vwVlWslf6g9JRQf8mBL3b04)
- **Docs**: [Helical Documentation](https://helical.readthedocs.io/)

## Acknowledgments

Built on top of [Helical](https://github.com/helicalAI/helical) and the excellent work of the bio foundation model community.


