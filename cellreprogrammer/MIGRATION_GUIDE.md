# Migration Guide: From Geneformer Repo to CellReprogrammer

This guide helps you migrate your existing genetic perturbation experiments from the original Geneformer repository to the new CellReprogrammer framework.

## Why Migrate?

CellReprogrammer provides:
- ✅ **Unified API** for multiple models (Geneformer, scGPT, UCE, etc.)
- ✅ **Reusable components** for common perturbation workflows
- ✅ **Better organization** with config management
- ✅ **Easy model switching** - test different models with same code

## Migration Overview

We'll migrate your code in a few simple steps:

### Before (Old Geneformer Way)

```python
# Your old code in Geneformer repo
from geneformer import Geneformer
import anndata as ad

# Manually load model
model = Geneformer(
    model_name="gf-12L-38M-i4096",
    batch_size=10
)

# Manually load data
ann_data = ad.read_h5ad("data.h5ad")

# Manually apply overexpression
perturbed_ann_data = apply_overexpression(ann_data, ["BRCA1", "TP53"], 2.0)

# Manually process and get embeddings
dataset = model.process_data(perturbed_ann_data)
embeddings = model.get_embeddings(dataset)
```

### After (CellReprogrammer Way)

```python
# New code with CellReprogrammer
from cellreprogrammer.src.models.model_factory import ModelFactory
from cellreprogrammer.src.perturbations.overexpression import OverexpressionPerturbation
import anndata as ad

# Load model via factory
factory = ModelFactory()
model = factory.load_model("geneformer", {
    "model_name": "gf-12L-38M-i4096",
    "batch_size": 10
})

# Load data
ann_data = ad.read_h5ad("data.h5ad")

# Use perturbation class
oe = OverexpressionPerturbation(
    model=model,
    perturbation_genes=["BRCA1", "TP53"],
    perturbation_strength=2.0
)

# Apply and compute in one flow
perturbed_dataset = oe.apply(ann_data)
embeddings = oe.compute_embeddings(perturbed_dataset)
```

## Step-by-Step Migration

### Step 1: Identify Your Current Workflow

Look through your old code and identify:
1. **Model loading**: How do you load models?
2. **Perturbation logic**: How do you apply perturbations?
3. **Data processing**: How do you prepare data?
4. **Embedding computation**: How do you get embeddings?
5. **Comparison logic**: How do you compare conditions?

### Step 2: Map to CellReprogrammer Components

| Old Component | New Component |
|--------------|---------------|
| Manual model loading | `ModelFactory.load_model()` |
| Custom perturbation functions | `BasePerturbation` subclasses |
| Manual data processing | `model.process_data()` (unchanged) |
| Manual embedding computation | `perturbation.compute_embeddings()` |
| Custom comparison code | `perturbation.compare_conditions()` |

### Step 3: Convert Your Code

Let's convert common patterns:

#### Pattern 1: Simple Overexpression

**Before**:
```python
def apply_overexpression(ann_data, genes, strength):
    # Your custom logic
    perturbed = ann_data.copy()
    for gene in genes:
        if gene in perturbed.var.index:
            idx = perturbed.var.index.get_loc(gene)
            perturbed.X[:, idx] *= strength
    return perturbed

# Usage
perturbed_ann_data = apply_overexpression(ann_data, ["BRCA1"], 2.0)
dataset = model.process_data(perturbed_ann_data)
embeddings = model.get_embeddings(dataset)
```

**After**:
```python
from cellreprogrammer.src.perturbations.overexpression import OverexpressionPerturbation

oe = OverexpressionPerturbation(
    model=model,
    perturbation_genes=["BRCA1"],
    perturbation_strength=2.0
)
perturbed_dataset = oe.apply(ann_data)
embeddings = oe.compute_embeddings(perturbed_dataset)
```

#### Pattern 2: Control vs Perturbed Comparison

**Before**:
```python
# Control
control_dataset = model.process_data(control_ann_data)
control_embeddings = model.get_embeddings(control_dataset)

# Perturbed
perturbed_dataset = model.process_data(perturbed_ann_data)
perturbed_embeddings = model.get_embeddings(perturbed_dataset)

# Compare
control_mean = np.mean(control_embeddings, axis=0)
perturbed_mean = np.mean(perturbed_embeddings, axis=0)
distance = np.linalg.norm(perturbed_mean - control_mean)
```

**After**:
```python
# Process data
control_dataset = model.process_data(control_ann_data)
perturbed_dataset = oe.apply(control_ann_data)  # Automatically processes

# Compare
results = oe.compare_conditions(control_dataset, perturbed_dataset)
distance = results['distance']
```

#### Pattern 3: Multiple Models

**Before**:
```python
# Testing with different models required code duplication
geneformer_model = Geneformer(model_name="gf-12L-38M-i4096")
scgpt_model = scGPT(batch_size=10)

# Same perturbation logic repeated for each
# ...
```

**After**:
```python
factory = ModelFactory()
models = ["geneformer", "scgpt"]

for model_name in models:
    model = factory.load_model(model_name, {...})
    oe = OverexpressionPerturbation(model=model, ...)
    # Same workflow works for all models!
```

#### Pattern 4: Custom Perturbations

**Before**:
```python
# Your custom perturbation function
def my_custom_perturbation(ann_data, genes, params):
    # Complex logic here
    ...
    return modified_data

# Different workflow every time
```

**After**:
```python
# Create reusable perturbation class
class MyCustomPerturbation(BasePerturbation):
    def apply(self, ann_data):
        # Your logic here
        return processed_data

# Same interface for all perturbations
perturbation = MyCustomPerturbation(model, ...)
```

### Step 4: Organize with Configs

**Before**: Hardcoded parameters in scripts

```python
model_name = "gf-12L-38M-i4096"
batch_size = 10
genes = ["BRCA1", "TP53"]
strength = 2.0
```

**After**: Config file

```yaml
# configs/my_experiment.yaml
model:
  name: "geneformer"
  config:
    model_name: "gf-12L-38M-i4096"
    batch_size: 10

perturbation:
  type: "overexpression"
  genes: ["BRCA1", "TP53"]
  strength: 2.0
```

Load in your script:
```python
from cellreprogrammer.configs import load_config

config = load_config("configs/my_experiment.yaml")
model = factory.load_model(**config['model'])
```

## Common Migration Scenarios

### Scenario 1: Batch Experiments

**Before**:
```python
gene_lists = [["BRCA1"], ["TP53"], ["MYC"]]
strengths = [1.5, 2.0, 2.5]

results = {}
for genes in gene_lists:
    for strength in strengths:
        # Repeated setup code
        model = Geneformer(...)
        perturbed = apply_overexpression(ann_data, genes, strength)
        dataset = model.process_data(perturbed)
        embeddings = model.get_embeddings(dataset)
        results[f"{genes}_{strength}"] = embeddings
```

**After**:
```python
for genes in gene_lists:
    for strength in strengths:
        oe = OverexpressionPerturbation(model, genes, strength)
        perturbed_dataset = oe.apply(ann_data)
        embeddings = oe.compute_embeddings(perturbed_dataset)
        results[f"{genes}_{strength}"] = embeddings
```

### Scenario 2: Model Comparison

**Before**:
```python
# Duplicated code for each model
geneformer_results = run_geneformer(ann_data, ...)
scgpt_results = run_scgpt(ann_data, ...)
uce_results = run_uce(ann_data, ...)

# Different functions for each model
```

**After**:
```python
models_to_test = ["geneformer", "scgpt", "uce"]
results = {}

for model_name in models_to_test:
    model = factory.load_model(model_name, {...})
    oe = OverexpressionPerturbation(model, ...)
    # Same workflow for all!
    results[model_name] = oe.compute_embeddings(oe.apply(ann_data))
```

### Scenario 3: Complex Workflows

**Before**:
```python
# Linear, hard to reuse
def run_experiment(ann_data, genes, strength, model_config):
    # Setup model
    model = setup_model(model_config)
    
    # Apply perturbation
    perturbed = apply_perturbation(ann_data, genes, strength)
    
    # Process
    dataset = process_data(perturbed, model)
    
    # Compute
    embeddings = compute_embeddings(dataset, model)
    
    # Post-process
    results = post_process(embeddings)
    
    return results
```

**After**:
```python
# Modular, reusable
def run_experiment(model, perturbation, ann_data):
    perturbed_dataset = perturbation.apply(ann_data)
    embeddings = perturbation.compute_embeddings(perturbed_dataset)
    # Each component is independent and testable
```

## Handling Edge Cases

### Edge Case 1: Gene Mapping

**Before**: Manual gene symbol to Ensembl ID mapping
```python
# Your custom mapping logic
```

**After**: Built-in support
```python
oe = OverexpressionPerturbation(
    model=model,
    perturbation_genes=["BRCA1"],  # Can use symbols
    perturbation_strength=2.0,
    use_ensembl=False  # Automatically maps
)
```

### Edge Case 2: Sparse Matrices

**Before**: Handle sparse manually
```python
if issparse(ann_data.X):
    ann_data.X = ann_data.X.todense()
# ... apply perturbation
```

**After**: Handled automatically in `OverexpressionPerturbation`

### Edge Case 3: Model-Specific Configs

**Before**: Different configs for each model type
```python
if model_type == "geneformer":
    config = {"model_name": "..."}
elif model_type == "scgpt":
    config = {"batch_size": ...}
```

**After**: Factory handles model-specific configs
```python
model = factory.load_model(model_name, config_overrides={...})
# Factory knows default configs for each model
```

## Testing Your Migration

1. **Start with a simple experiment**: Convert your simplest perturbation
2. **Compare outputs**: Verify embeddings are the same (or very similar)
3. **Check performance**: Ensure similar runtime
4. **Gradually migrate**: Convert one experiment at a time

## Getting Help

If you run into issues during migration:

1. Check the README for usage examples
2. Look at `examples/example_overexpression.py`
3. Review the base classes in `src/perturbations/base_perturbation.py`
4. Open an issue in the helical repo

## Benefits After Migration

After migration, you'll have:

✅ **Less code duplication**: Reusable components  
✅ **Easier model switching**: Test different models with minimal changes  
✅ **Better organization**: Clear separation of concerns  
✅ **Easier testing**: Modular components  
✅ **Configuration management**: YAML configs for reproducibility  
✅ **Future-proof**: Easy to add new models or perturbations  

## Rollback Plan

If you need to go back temporarily:

1. Your old code is still there
2. CellReprogrammer doesn't delete anything
3. You can mix old and new code in the same script
4. Gradually migrate as you're comfortable

## Next Steps

1. ✅ Read this guide
2. ✅ Try the example script: `experiments/example_overexpression.py`
3. ✅ Convert one simple experiment
4. ✅ Verify results match
5. ✅ Gradually migrate more experiments

Good luck with your migration! 🚀


