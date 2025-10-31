# CellReprogrammer Separation Guide

This guide explains how to separate CellReprogrammer into its own repository while maintaining access to Helical updates.

## Current Setup

Currently, CellReprogrammer lives within the Helical repository at:
```
helical/
├── cellreprogrammer/     # Your custom code here
│   ├── src/
│   ├── experiments/
│   ├── configs/
│   └── README.md
├── helical/              # Helical core models
├── examples/
└── ...
```

This is **perfect for development** because:
- ✅ Easy to test changes against latest Helical updates
- ✅ No version mismatches
- ✅ Direct access to Helical internals if needed
- ✅ Simple imports

## When to Separate?

Consider separating when:
1. CellReprogrammer has its own release cycle
2. You want to share CellReprogrammer without sharing Helical code
3. You have multiple projects using CellReprogrammer
4. You want a cleaner public-facing repository

## Separation Strategies

### Strategy 1: Git Submodule (Recommended for Active Development)

**Best for**: Active development, need latest Helical features

**Setup**:

```bash
# 1. Create your CellReprogrammer repository
cd ~/GitHub
git clone https://github.com/YOUR_USERNAME/CellReprogrammer.git
cd CellReprogrammer

# 2. Add Helical as a submodule
git submodule add https://github.com/helicalAI/helical.git helical
git commit -m "Add helical as submodule"

# 3. Update imports in your code
# Change: from helical.models.geneformer import Geneformer
# To: from helical.helical.models.geneformer import Geneformer
# Or add to sys.path
```

**Advantages**:
- ✅ Easy to pull latest Helical updates: `git submodule update --remote`
- ✅ Pin to specific Helical versions
- ✅ Clean separation of codebases

**Disadvantages**:
- ⚠️ Slightly more complex setup
- ⚠️ Need to update submodule references

**Usage**:
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/YOUR_USERNAME/CellReprogrammer.git

# Update helical submodule
git submodule update --remote helical
```

### Strategy 2: PyPI Dependency (Best for Production)

**Best for**: Stable releases, multiple users

**Setup**:

1. **Helical is on PyPI** (already is!)
   ```bash
   pip install helical
   ```

2. **Create your own package** with `setup.py` or `pyproject.toml`:

   ```python
   # setup.py or pyproject.toml
   from setuptools import setup, find_packages
   
   setup(
       name="cellreprogrammer",
       version="0.1.0",
       packages=find_packages(where="src"),
       package_dir={"": "src"},
       install_requires=[
           "helical>=1.4.0",  # Depend on helical
           "pandas>=2.0.0",
           # ... other dependencies
       ],
   )
   ```

3. **Publish to PyPI**:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

**Advantages**:
- ✅ Cleanest separation
- ✅ Version control via PyPI
- ✅ Easy installation for users: `pip install cellreprogrammer`

**Disadvantages**:
- ⚠️ Need to publish releases to PyPI
- ⚠️ Wait for Helical releases to PyPI

### Strategy 3: Fork and Maintain (Most Control)

**Best for**: Need to modify Helical, custom forks

**Setup**:

```bash
# 1. Fork helical on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/helical.git
cd helical

# 3. Add upstream remote
git remote add upstream https://github.com/helicalAI/helical.git

# 4. Your CellReprogrammer code goes in cellreprogrammer/

# 5. To pull updates from upstream
git fetch upstream
git checkout main
git merge upstream/main
```

**Advantages**:
- ✅ Full control over codebase
- ✅ Can modify Helical if needed

**Disadvantages**:
- ⚠️ Need to manually merge upstream changes
- ⚠️ More maintenance overhead

### Strategy 4: Development Mode (Current Best)

**Best for**: Current situation - active development within helical

Keep CellReprogrammer inside Helical and use it in development mode:

```bash
cd ~/GitHub/helical
pip install -e .
```

Then use in your scripts:
```python
import sys
sys.path.insert(0, '/path/to/helical/cellreprogrammer')

from cellreprogrammer.src.models import ModelFactory
```

**Advantages**:
- ✅ Zero overhead
- ✅ Always latest code
- ✅ Easy to test

**Disadvantages**:
- ⚠️ Tightly coupled
- ⚠️ Harder to share separately

## Recommended Migration Path

For your use case, I recommend this progression:

### Phase 1: Now - Keep Inside Helical ✅
- Develop CellReprogrammer inside helical
- Use for your research
- Iterate quickly

### Phase 2: Git Submodule (When Sharing)
- When ready to share or need separation
- Use Strategy 1 (git submodule)
- Allows easy Helical updates

### Phase 3: Production (If Needed)
- If CellReprogrammer becomes a standalone project
- Migrate to Strategy 2 (PyPI dependency)
- Most professional approach

## Import Path Considerations

When you separate, you'll need to update imports. Here are the options:

### Option A: Keep Relative Imports (Current)
```python
# Inside cellreprogrammer/src/models/model_factory.py
from helical.models.geneformer import Geneformer

# Becomes (after separation with submodule):
from helical.helical.models.geneformer import Geneformer
# Or add to path
```

### Option B: Always Use Helical from PyPI
```python
# If helical is installed via pip
from helical.models.geneformer import Geneformer
# Works anywhere after pip install helical
```

### Option C: Add Import Helpers
Create a compatibility layer:

```python
# cellreprogrammer/src/utils/imports.py
import sys
from pathlib import Path

def setup_helical_imports():
    """Setup paths for helical imports."""
    # Try to import from installed package
    try:
        import helical
        return True
    except ImportError:
        pass
    
    # Try to import from git submodule
    base_path = Path(__file__).parent.parent.parent.parent
    helical_path = base_path / "helical"
    if helical_path.exists():
        sys.path.insert(0, str(helical_path))
        return True
    
    # Try to import from sibling directory
    sibling_helical = base_path.parent / "helical"
    if sibling_helical.exists():
        sys.path.insert(0, str(sibling_helical))
        return True
    
    raise ImportError("Could not find helical installation")
```

## Example: Quick Submodule Migration

If you want to try Strategy 1 right now:

```bash
# 1. Create a new repo
cd ~/GitHub
mkdir CellReprogrammer
cd CellReprogrammer
git init

# 2. Copy your code (don't copy helical directory)
mkdir src
# Copy cellreprogrammer/src/* to src/
# Copy cellreprogrammer/experiments/ to experiments/
# Copy cellreprogrammer/configs/ to configs/

# 3. Add helical as submodule
git submodule add https://github.com/helicalAI/helical.git

# 4. Create adapter for imports
# Create src/helical_adapter.py
import sys
from pathlib import Path

helical_path = Path(__file__).parent.parent / "helical" / "helical"
if helical_path.exists():
    sys.path.insert(0, str(helical_path.parent))

# 5. In your imports, add:
# from helical_adapter import helical
# from helical.models.geneformer import Geneformer
```

## Summary

| Strategy | Complexity | Maintenance | Best For |
|----------|-----------|-------------|----------|
| **Keep Inside** | ⭐ Easy | Low | Development |
| **Git Submodule** | ⭐⭐ Medium | Medium | Sharing/Active Dev |
| **PyPI Dependency** | ⭐⭐⭐ Hard | Low | Production |
| **Fork** | ⭐⭐ Medium | High | Custom Mods |

**My Recommendation**: Stay with the current setup for now (Phase 1). When you're ready to share or separate, go straight to git submodule (Phase 2). Only move to PyPI (Phase 3) if CellReprogrammer becomes a major standalone project.

## Questions?

If you need help with any migration step, feel free to:
- Open an issue in the helical repo
- Reach out on Slack
- Modify this guide as you learn!

The beauty of this approach is that **you're not locked in** - you can always change strategies later as your needs evolve.


