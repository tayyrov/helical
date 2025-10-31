# Data Organization

## Directory Structure

All experiment data and results are organized under the `cellreprogrammer/` directory to keep the helical package clean.

```
/home/ubuntu/data-at-virginia/helical/
├── helical/                      # Helical package (untouched!)
├── cellreprogrammer/             # Your CellReprogrammer workspace
│   ├── src/                      # Framework code
│   ├── experiments/              # Experiment scripts
│   ├── configs/                  # Configurations
│   ├── data/                     # Experiment data (created at runtime)
│   │   ├── raw/                  # Raw input files
│   │   ├── prepared/             # Converted/processed data
│   │   └── tokenized/            # Tokenized datasets
│   ├── results/                  # Experiment outputs
│   │   └── oskm_experiment/      # OSKM reprogramming results
│   └── notebooks/                # Jupyter notebooks
└── models/                       # Downloaded models (shared)
```

## Why This Organization?

### ✅ Clean Separation
- **Helical package**: Pure, uncontaminated codebase
- **CellReprogrammer data**: Isolated under your workspace
- **Easy cleanup**: Just delete `cellreprogrammer/data/` and `cellreprogrammer/results/`

### ✅ No Contamination
- Data files won't clutter the helical repository
- Results won't mix with helical's code
- Git ignores these directories (see `.gitignore`)

### ✅ Easy Backups
- Backup just `cellreprogrammer/` folder
- Or backup `data/` and `results/` separately
- Models can be redownloaded if needed

### ✅ Portability
- Move entire workspace: `cp -r cellreprogrammer/ new_location/`
- Copy to new server easily
- Share experiments without sharing helical codebase

## Data Flow

### Typical Workflow

```
1. Raw Data
   ↓
   data/raw/GSE118258_*.gz

2. Convert
   ↓
   python experiments/01_convert_geo_data.py
   ↓
   data/prepared/GSE118258_converted.h5ad

3. Prepare & Tokenize
   ↓
   python experiments/02_prepare_reprogramming_data.py
   ↓
   data/tokenized/fibroblast_ipsc.dataset

4. Run Experiment
   ↓
   python experiments/03_reproduce_reprogramming.py
   ↓
   results/oskm_experiment/*.csv
```

## Git Ignore

The `.gitignore` file automatically excludes:
- `data/` - All experiment data
- `results/` - All experiment outputs
- `__pycache__/` - Python cache files
- Virtual environments
- IDE files

This keeps the repository clean while allowing you to work freely.

## Best Practices

### ✅ DO:
- Keep all experiment data under `cellreprogrammer/data/`
- Save results to `cellreprogrammer/results/`
- Use clear subdirectories (e.g., `results/oskm_experiment/`)
- Document your data sources

### ❌ DON'T:
- Put data files in the helical package directory
- Commit large data files to git
- Mix experimental data with helical's code
- Hardcode absolute paths

## Multiple Experiments

Organize multiple experiments like this:

```
cellreprogrammer/
├── data/
│   ├── raw/
│   │   ├── experiment_1_*.gz
│   │   └── experiment_2_*.gz
│   ├── prepared/
│   │   ├── experiment_1_*.h5ad
│   │   └── experiment_2_*.h5ad
│   └── tokenized/
│       ├── experiment_1.dataset
│       └── experiment_2.dataset
├── results/
│   ├── experiment_1/
│   │   └── *.csv
│   └── experiment_2/
│       └── *.csv
```

Or use completely separate directories:

```
experiments/
├── experiment_1/
│   ├── data/
│   ├── results/
│   └── scripts/
└── experiment_2/
    ├── data/
    ├── results/
    └── scripts/
```

## Storage Considerations

### Expected Sizes

- **Raw data**: Can be GBs (e.g., GSE118258 ~10GB uncompressed)
- **Prepared data**: Usually smaller (h5ad with compression)
- **Tokenized data**: Similar to prepared
- **Results**: Usually MBs (CSV files, plots)

### Disk Space

Make sure you have enough disk space:
```bash
du -sh cellreprogrammer/data/
du -sh cellreprogrammer/results/
```

### Cleanup

To remove old experiments:
```bash
# Remove specific experiment
rm -rf cellreprogrammer/results/old_experiment/

# Remove all data (careful!)
rm -rf cellreprogrammer/data/
rm -rf cellreprogrammer/results/
```

## Remote Server Setup

On your remote server at `/home/ubuntu/data-at-virginia/helical`:

```bash
# Verify structure
ls -la

# Should see:
# helical/          # The package
# cellreprogrammer/ # Your workspace
# models/           # Downloaded models (optional)

# Data will be created under:
mkdir -p cellreprogrammer/data/{raw,prepared,tokenized}
mkdir -p cellreprogrammer/results
```

## Summary

- **All data goes under `cellreprogrammer/data/`**
- **All results go under `cellreprogrammer/results/`**
- **Helical package stays clean**
- **Easy to backup, share, and clean up**

This organization makes it easy to:
- ✅ Keep experiments isolated
- ✅ Share code without data
- ✅ Clean up when done
- ✅ Move to new environments
- ✅ Collaborate without conflicts

