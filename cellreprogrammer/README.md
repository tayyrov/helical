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

### Basic Overexpression Experiment

```python
from cellreprogrammer.src.models.model_factory import ModelFactory
from cellreprogrammer.src.perturbations.overexpression import OverexpressionPerturbation
import anndata as ad

# 1. Load a model using the factory
factory = ModelFactory()
model = factory.load_model(
    "geneformer",
    config_overrides={"model_name": "gf-12L-38M-i4096", "batch_size": 10}
)

# 2. Load your data
ann_data = ad.read_h5ad("your_data.h5ad")

# 3. Create perturbation
oe = OverexpressionPerturbation(
    model=model,
    perturbation_genes=["BRCA1", "TP53", "MYC"],
    perturbation_strength=2.5  # 2.5x overexpression
)

# 4. Apply perturbation
perturbed_dataset = oe.apply(ann_data)

# 5. Compute embeddings
embeddings = oe.compute_embeddings(perturbed_dataset)
print(f"Embeddings shape: {embeddings.shape}")
```

### Comparing Control vs Perturbed

```python
# Process control data
control_dataset = model.process_data(ann_data)

# Apply perturbation
perturbed_dataset = oe.apply(ann_data)

# Compare conditions
results = oe.compare_conditions(control_dataset, perturbed_dataset)
print(f"Distance between conditions: {results['distance']:.4f}")
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

CellReprogrammer supports all models available through Helical's unified API:

| Model | Description | Use Case |
|-------|-------------|----------|
| **Geneformer** | Transformer for scRNA-seq | Cell state transitions, perturbation prediction |
| **scGPT** | GPT-style model for cells | Cell type annotation, perturbation effects |
| **UCE** | Universal Cell Embedding | General embeddings, comparisons |
| **Helix-mRNA** | mRNA foundation model | mRNA-level perturbations |
| **Mamba2-mRNA** | Mamba-architecture mRNA model | Long-sequence mRNA analysis |

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

## Future Roadmap

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


