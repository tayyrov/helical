#!/bin/bash
# Complete migration script for cellreprogrammer

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Complete Cellreprogrammer Migration${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check location
if [ ! -d "helical/cellreprogrammer" ]; then
    echo "Error: Please run from ~/GitHub"
    exit 1
fi

# Create new cellreprogrammer if it doesn't exist
if [ ! -d "cellreprogrammer" ]; then
    mkdir cellreprogrammer
    cd cellreprogrammer
    git init
    git branch -M main
else
    cd cellreprogrammer
fi

echo -e "${YELLOW}Step 1: Copying all files from helical/cellreprogrammer...${NC}"

# Copy entire directory structure, excluding git/pycache
rsync -av --exclude='.git' \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='.DS_Store' \
          ../helical/cellreprogrammer/ .

echo -e "${GREEN}✓ Files copied${NC}"
echo ""

echo -e "${YELLOW}Step 2: Reorganizing structure...${NC}"

# Keep experiments organized
if [ ! -d "experiments/geneformer" ]; then
    mv geneformer/experiments experiments/geneformer 2>/dev/null || true
fi

if [ ! -d "experiments/scgpt" ] && [ -f "scgpt/run_perturbation.py" ]; then
    mkdir -p experiments/scgpt
    mv scgpt/run_perturbation.py experiments/scgpt/ 2>/dev/null || true
fi

if [ ! -d "experiments/c2s" ] && [ -f "c2s/run_perturbation.py" ]; then
    mkdir -p experiments/c2s
    mv c2s/run_perturbation.py experiments/c2s/ 2>/dev/null || true
fi

# Clean up now-empty model dirs if they exist
rmdir geneformer 2>/dev/null || true
rmdir scgpt 2>/dev/null || true  
rmdir c2s 2>/dev/null || true

echo -e "${GREEN}✓ Structure reorganized${NC}"
echo ""

echo -e "${YELLOW}Step 3: Creating additional directories...${NC}"

# Create data and results directories
mkdir -p data/{raw,prepared,tokenized}
mkdir -p results/{oskm,norman,evaluation}
mkdir -p benchmarks/perteval_adapters/config_templates

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/prepared/.gitkeep
touch data/tokenized/.gitkeep
touch results/.gitkeep

echo -e "${GREEN}✓ Directories created${NC}"
echo ""

echo -e "${YELLOW}Step 4: Copying setup files...${NC}"

# Copy the setup files we created
cp ../helical/cellreprogrammer_requirements.txt requirements.txt 2>/dev/null || echo "requirements.txt not found, skipping"
cp ../helical/cellreprogrammer_pyproject.toml pyproject.toml 2>/dev/null || echo "pyproject.toml not found, skipping"
cp ../helical/cellreprogrammer_gitignore .gitignore 2>/dev/null || echo ".gitignore not found, skipping"

# Update README if we have a new one
if [ -f "../helical/cellreprogrammer_README.md" ]; then
    echo -e "${YELLOW}  Found new README, replacing...${NC}"
    cp ../helical/cellreprogrammer_README.md README.md
fi

echo -e "${GREEN}✓ Setup files copied${NC}"
echo ""

echo "=" * 80
echo -e "${GREEN}Migration complete!${NC}"
echo ""
echo "Directory contents:"
ls -la
echo ""
echo "Next steps:"
echo "  1. Review the migrated files"
echo "  2. Run: python ../helical/fix_imports.py"
echo "  3. Set up virtual environment:"
echo "     python3.10 -m venv .venv"
echo "     source .venv/bin/activate"
echo "     pip install helical crc32c"
echo "     pip install -r requirements.txt"
echo "  4. Test imports"
echo "  5. Commit to git"
echo ""



