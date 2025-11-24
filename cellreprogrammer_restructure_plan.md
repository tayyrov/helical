# Cellreprogrammer Restructuring Plan

## New Repository Structure

```
cellreprogrammer/
├── .gitignore
├── README.md
├── pyproject.toml  (or setup.py)
├── requirements.txt
│
├── src/cellreprogrammer/
│   ├── __init__.py
│   ├── models/              # Wrappers for helical models
│   │   ├── __init__.py
│   │   ├── geneformer_wrapper.py
│   │   └── base_wrapper.py
│   │
│   ├── perturbations/       # Perturbation logic
│   │   ├── __init__.py
│   │   ├── overexpression.py
│   │   └── base_perturbation.py
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── data_utils.py
│
├── experiments/
│   ├── oskm/
│   │   ├── prepare_data.py
│   │   ├── test_oskm_combinations.py
│   │   └── README.md
│   │
│   └── norman/
│       ├── prepare_norman_data.py
│       ├── test_norman_dataset.py
│       ├── test_norman_perturbations.py
│       ├── evaluate_norman_geneformer.py
│       └── README.md
│
├── benchmarks/
│   ├── README.md
│   ├── setup_perteval.sh
│   └── perteval_adapters/
│       ├── __init__.py
│       ├── extract_embeddings.py
│       └── config_templates/
│
├── data/                   # Data directory (add to .gitignore)
│   ├── raw/
│   ├── prepared/
│   └── tokenized/
│
├── results/                # Results directory (add to .gitignore)
│   ├── oskm/
│   └── norman/
│
└── tests/
    └── test_basic.py
```

## Files to Create/Update

### 1. requirements.txt
```
helical>=0.0.1
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scanpy>=1.9.0
anndata>=0.9.0
datasets>=2.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
umap-learn>=0.5.0
tqdm>=4.65.0
```

### 2. pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "cellreprogrammer"
version = "0.1.0"
description = "Cell reprogramming experiments using helical foundation models"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "helical>=0.0.1",
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scanpy>=1.9.0",
]
requires-python = ">=3.10"

[project.optional-dependencies]
dev = ["pytest>=7.0", "black>=23.0", "flake8>=6.0"]
```

### 3. .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/
dist/
build/

# Data (don't commit large files)
data/raw/*
data/prepared/*
data/tokenized/*
results/*
*.h5ad
*.dataset
*.npy
*.pkl
*.pickle

# Keep directory structure
!data/raw/.gitkeep
!data/prepared/.gitkeep
!data/tokenized/.gitkeep
!results/.gitkeep

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
```

## Import Changes Required

### Before (old helical/cellreprogrammer):
```python
from helical.models.geneformer import Geneformer, GeneformerConfig
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### After (new cellreprogrammer repo):
```python
# Just import from helical (installed package)
from helical.models.geneformer import Geneformer, GeneformerConfig

# For cellreprogrammer utilities
from cellreprogrammer.utils import data_utils
from cellreprogrammer.models import geneformer_wrapper
```

## Path Changes Required

### Before:
```python
BASE_DIR = script_dir.parent.parent  # helical/
CELLREPROGRAMMER_DIR = BASE_DIR / "cellreprogrammer"
```

### After:
```python
# Assume data is in repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Or use environment variable
DATA_DIR = Path(os.getenv("CELLREPROGRAMMER_DATA", "~/data/cellreprogrammer")).expanduser()
```

## Migration Steps

1. **Create new repo:**
   ```bash
   cd ~/GitHub
   mkdir cellreprogrammer
   cd cellreprogrammer
   git init
   ```

2. **Copy files:**
   ```bash
   # Copy from helical
   cp -r ~/GitHub/helical/cellreprogrammer/geneformer/experiments/* experiments/oskm/
   # Organize files into proper structure
   ```

3. **Create new files:**
   - requirements.txt
   - pyproject.toml
   - .gitignore
   - README.md

4. **Update all scripts:**
   - Remove sys.path hacks
   - Change imports to use helical package
   - Update data paths
   - Test each script

5. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install helical
   pip install -e .
   ```

6. **Test:**
   ```bash
   # Test imports
   python -c "from helical.models.geneformer import Geneformer; print('✓')"
   python -c "from cellreprogrammer.utils import data_utils; print('✓')"
   
   # Test scripts
   cd experiments/norman
   python prepare_norman_data.py
   ```

7. **Commit:**
   ```bash
   git add .
   git commit -m "Initial commit: cellreprogrammer as standalone repo"
   git remote add origin git@github.com:YOUR_USERNAME/cellreprogrammer.git
   git push -u origin main
   ```



