# CellReprogrammer Experiments

This directory contains complete workflows for genetic perturbation experiments.

## Available Workflows

### GSE118258 Reprogramming Pipeline

Complete pipeline for fibroblast → iPSC reprogramming experiments using the 
GSE118258 dataset (Xing et al. 2020).

**Scripts (run in order):**

1. **`01_convert_geo_data.py`** - Convert raw GSE118258 data to h5ad format
2. **`02_prepare_reprogramming_data.py`** - Prepare and tokenize data for Geneformer
3. **`03_reproduce_reprogramming.py`** - Run OSKM reprogramming experiment

## Quick Start

### Prerequisites

1. **Download GSE118258 data:**
   ```bash
   # Download to data/raw/
   cd /home/ubuntu/data-at-virginia/helical/data/raw
   wget <GSE118258_UMI.csv.gz>
   wget <GSE118258_Annotation.txt.gz>
   ```

2. **Install dependencies:**
   ```bash
   # helical (required)
   pip install helical
   
   # geneformer (for advanced perturbation utilities)
   pip install geneformer
   ```

3. **Update paths in scripts:**
   - All scripts use BASE_DIR = `/home/ubuntu/data-at-virginia/helical`
   - Data goes under `cellreprogrammer/data/` (keeps helical clean)
   - Adjust if your setup differs

### Run Pipeline

```bash
cd /home/ubuntu/data-at-virginia/helical/cellreprogrammer/experiments

# Step 1: Convert raw data
python 01_convert_geo_data.py

# Step 2: Prepare and tokenize
python 02_prepare_reprogramming_data.py

# Step 3: Run reprogramming experiment
python 03_reproduce_reprogramming.py
```

## Script Details

### 01_convert_geo_data.py

**Purpose:** Convert raw GEO data to AnnData format

**Input:**
- `data/raw/GSE118258_UMI.csv.gz`
- `data/raw/GSE118258_Annotation.txt.gz`

**Output:**
- `data/prepared/GSE118258_converted.h5ad`

**Features:**
- Handles gzipped UMI matrices
- Converts genes × cells to cells × genes
- Maps timepoints to cell types
- Creates sparse matrices for efficiency

### 02_prepare_reprogramming_data.py

**Purpose:** Prepare and tokenize data for Geneformer

**Input:**
- `data/prepared/GSE118258_converted.h5ad`

**Output:**
- `data/prepared/fibroblast_ipsc_prepared.h5ad`
- `data/tokenized/fibroblast_ipsc.dataset`

**Features:**
- Uses helical's TranscriptomeTokenizer
- Quality control filtering
- Metadata preservation
- Model V2 compatible

### 03_reproduce_reprogramming.py

**Purpose:** Run OSKM reprogramming experiment

**Input:**
- `data/tokenized/fibroblast_ipsc.dataset`
- Geneformer model files

**Output:**
- `results/oskm_experiment/oskm_individual_stats.csv`
- `results/oskm_experiment/random_control_stats.csv`

**Features:**
- Tests OSKM (OCT4, SOX2, KLF4, MYC) overexpression
- Random control comparison
- State shift analysis
- Uses geneformer package for advanced features

**Note:** Requires both helical AND geneformer packages

## Directory Structure

```
cellreprogrammer/experiments/
├── 01_convert_geo_data.py           # Data conversion
├── 02_prepare_reprogramming_data.py  # Tokenization
├── 03_reproduce_reprogramming.py    # Main experiment
├── example_overexpression.py         # Simple example
└── README.md                         # This file
```

Expected data structure:
```
/home/ubuntu/data-at-virginia/helical/
├── cellreprogrammer/                   # CellReprogrammer workspace
│   ├── data/                           # Experiment data (isolated!)
│   │   ├── raw/                        # Raw GEO files
│   │   ├── prepared/                   # Converted h5ad files
│   │   └── tokenized/                  # Tokenized datasets
│   └── results/                        # Experiment results
│       └── oskm_experiment/            # OSKM experiment output
├── models/                             # Geneformer models
└── helical/                            # Helical package (untouched!)
```

## Configuration

All scripts use a base directory:
```python
BASE_DIR = Path("/home/ubuntu/data-at-virginia/helical")
```

Update this in each script if your setup differs.

## Troubleshooting

### Missing geneformer package

If you see:
```
ERROR: Missing geneformer package
```

**Solution:** Install with `pip install geneformer`

**Note:** Some scripts require both helical AND geneformer. This is intentional
because helical doesn't yet wrap all Geneformer functionality.

### CUDA memory errors

If you get out of memory errors:

1. Reduce `MAX_NCELLS` in `03_reproduce_reprogramming.py`
2. Reduce `FORWARD_BATCH_SIZE`
3. Use CPU: set `device='cpu'` in model config

### Path not found errors

Ensure:
1. Raw data files exist in `cellreprogrammer/data/raw/`
2. Previous scripts completed successfully
3. Base directory path is correct

### Tokenization issues

If tokenization fails:
1. Check that data has `ensembl_id` in `var`
2. Check that data has `n_counts` in `obs`
3. Verify `cell_type` column exists

## Next Steps

After running experiments:

1. Analyze results in `results/oskm_experiment/`
2. Compare OSKM vs random controls
3. Visualize state shifts
4. Try different gene combinations

## Additional Examples

See `example_overexpression.py` for a simpler example using just helical's API.

## Alternative: Pure Helical Approach

If you prefer not to use the original geneformer package, see:
- `src/perturbations/overexpression.py` - Overexpression implementation
- `src/models/model_factory.py` - Unified model loading

These provide similar functionality through helical's APIs.

## Reference

- Xing, Q. R. et al. (2020) Science Advances
- Theodoris et al. Nature 2023 (Geneformer paper)
- [Geneformer repo](https://github.com/HuggingFace/Geneformer)
- [Helical docs](https://helical.readthedocs.io/)

