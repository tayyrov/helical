# Quick Start Guide

Get started with CellReprogrammer in 5 minutes!

## Prerequisites

Ensure you have Helical installed:

```bash
pip install helical
# or for latest from git
pip install --upgrade git+https://github.com/helicalAI/helical.git
```

## Installation

Since CellReprogrammer lives within the Helical repo, just ensure you have the latest code:

```bash
cd ~/GitHub/helical
git pull origin main
```

Add to your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("~/GitHub/helical/cellreprogrammer").expanduser()))
```

Or install in development mode (if you add a setup.py later):

```bash
cd ~/GitHub/helical
pip install -e .
```

## Your First Experiment (5 minutes)

### 1. Create a Simple Script

Create `my_first_experiment.py`:

```python
import sys
from pathlib import Path

# Add CellReprogrammer to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cellreprogrammer"))

from cellreprogrammer.src.models.model_factory import ModelFactory
from cellreprogrammer.src.perturbations.overexpression import OverexpressionPerturbation
import anndata as ad

# Initialize factory
factory = ModelFactory()

# Load a model
model = factory.load_model(
    "geneformer",
    config_overrides={
        "model_name": "gf-12L-38M-i4096",
        "batch_size": 10,
        "device": "cuda"  # or "cpu"
    }
)

# Load your data
ann_data = ad.read_h5ad("path/to/your/data.h5ad")

# Create overexpression perturbation
oe = OverexpressionPerturbation(
    model=model,
    perturbation_genes=["BRCA1", "TP53", "MYC"],
    perturbation_strength=2.5  # 2.5x overexpression
)

# Apply perturbation and get embeddings
perturbed_dataset = oe.apply(ann_data)
embeddings = oe.compute_embeddings(perturbed_dataset)

print(f"✓ Success! Computed embeddings with shape: {embeddings.shape}")
```

### 2. Run It

```bash
cd ~/GitHub/helical
python my_first_experiment.py
```

That's it! 🎉

## Next Steps

### Explore More Features

1. **Control vs Perturbed Comparison**:
```python
# Process control data
control_dataset = model.process_data(ann_data)

# Apply perturbation
perturbed_dataset = oe.apply(ann_data)

# Compare
results = oe.compare_conditions(control_dataset, perturbed_dataset)
print(f"Distance: {results['distance']:.4f}")
```

2. **Try Different Models**:
```python
# Easy to switch models!
scgpt_model = factory.load_model("scgpt", {...})
oe2 = OverexpressionPerturbation(model=scgpt_model, ...)
```

3. **Use Config Files**:
```python
from cellreprogrammer.configs import load_config

config = load_config("cellreprogrammer/configs/example_config.yaml")
# Use config to setup everything automatically
```

### Check Out Examples

```bash
# Look at the example script
cat cellreprogrammer/experiments/example_overexpression.py
```

### Learn More

- **Full documentation**: `README.md`
- **Migration guide**: `MIGRATION_GUIDE.md` (from Geneformer repo)
- **Separation guide**: `SEPARATION_GUIDE.md` (future planning)

## Common Tasks

### Task 1: Overexpression Experiment
```python
oe = OverexpressionPerturbation(model, genes=["BRCA1"], strength=2.0)
results = oe.apply(data)
```

### Task 2: Batch Experiments
```python
for genes in gene_lists:
    for strength in strengths:
        oe = OverexpressionPerturbation(model, genes, strength)
        embeddings = oe.compute_embeddings(oe.apply(data))
        # Store results
```

### Task 3: Model Comparison
```python
for model_name in ["geneformer", "scgpt", "uce"]:
    model = factory.load_model(model_name, {...})
    oe = OverexpressionPerturbation(model, ...)
    # Compare results across models
```

## Troubleshooting

### Import Error
**Problem**: `ModuleNotFoundError: No module named 'cellreprogrammer'`

**Solution**: Add to Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("path/to/helical/cellreprogrammer")))
```

### CUDA Error
**Problem**: CUDA out of memory

**Solution**: Reduce batch size:
```python
model = factory.load_model("geneformer", {"batch_size": 4})  # Smaller batch
```

### Gene Not Found
**Problem**: Gene not found in data

**Solution**: Check gene names:
```python
print(ann_data.var.index[:10])  # Check available genes
```

## Get Help

- 📖 Read the full `README.md`
- 🐛 Check examples in `experiments/`
- 💬 Open an issue on GitHub
- 📧 Ask on Slack

Happy experimenting! 🧬


