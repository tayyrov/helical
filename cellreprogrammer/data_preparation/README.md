# Data Preparation

This folder contains scripts for preparing data for perturbation experiments.

## Scripts

- **`convert_geo_data.py`**: Converts GEO datasets (e.g., GSE118258) to AnnData format suitable for perturbation experiments. This is a model-agnostic data conversion script.

## Usage

```bash
python convert_geo_data.py
```

## Output

The script outputs processed `.h5ad` files in the `cellreprogrammer/data/` directory.
