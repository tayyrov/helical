# CellReprogrammer Server Setup Guide

## Quick Start

On your server at `/home/ubuntu/data-at-virginia/helical`, run these commands:

### 1. Install Geneformer

```bash
cd /home/ubuntu/data-at-virginia/helical
source .venv/bin/activate
pip install -e ../Geneformer/
```

This installs the original Geneformer package so scripts can import InSilicoPerturber, etc.

### 2. Verify Setup

```bash
cd /home/ubuntu/data-at-virginia/helical/cellreprogrammer/experiments
python 03_reproduce_reprogramming.py
```

You should see: "Using original Geneformer package"  
And the model will automatically download from HuggingFace on first run.

### 3. Run Experiments

```bash
# Step 1: Convert GEO data
python 01_convert_geo_data.py

# Step 2: Prepare and tokenize
python 02_prepare_reprogramming_data.py

# Step 3: Run reprogramming experiment
python 03_reproduce_reprogramming.py
```

## Note on Model Download

The script uses the HuggingFace model ID `gf-12L-38M-i4096` which will automatically 
download from `ctheodoris/Geneformer` on first run and cache in `~/.cache/huggingface/`.

This is a ~1.5GB download but only happens once.

## What's Working

✅ **Helical Integration**: Core helical APIs work  
✅ **Tokenizer Bug Fixed**: Files download automatically  
✅ **Data Organization**: All data under cellreprogrammer/data/  
✅ **Scripts 1 & 2**: Convert and prepare data working  

⚠️ **Script 3**: Needs original Geneformer for InSilicoPerturber utilities

## Why Not Just Use Helical?

Helical has the Geneformer model and tokenizer, but NOT the perturbation utilities
(InSilicoPerturber, EmbExtractor, InSilicoPerturberStats). These are from the original
Geneformer paper and haven't been ported to helical yet.

Your `cellreprogrammer/src/geneformer_utils/` contains copied files but they're not
yet adapted to helical's architecture (different function signatures, etc.)

## Future Work

Consider migrating to use your own `OverexpressionPerturbation` class from
`cellreprogrammer/src/perturbations/` instead of relying on original Geneformer.

