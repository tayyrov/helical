#!/bin/bash
# Migration script to move cellreprogrammer out of helical

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cellreprogrammer Repository Migration${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -d "helical" ]; then
    echo -e "${RED}Error: Please run this script from ~/GitHub${NC}"
    exit 1
fi

# Create new cellreprogrammer directory
echo -e "${YELLOW}Step 1: Creating new cellreprogrammer directory...${NC}"
mkdir -p cellreprogrammer
cd cellreprogrammer

# Initialize git
echo -e "${YELLOW}Step 2: Initializing git repository...${NC}"
git init
git branch -M main

# Create directory structure
echo -e "${YELLOW}Step 3: Creating directory structure...${NC}"
mkdir -p src/cellreprogrammer/{models,perturbations,utils}
mkdir -p experiments/{oskm,norman}
mkdir -p benchmarks/perteval_adapters/config_templates
mkdir -p data/{raw,prepared,tokenized}
mkdir -p results/{oskm,norman}
mkdir -p tests

# Create .gitkeep files for empty directories
touch data/raw/.gitkeep
touch data/prepared/.gitkeep
touch data/tokenized/.gitkeep
touch results/.gitkeep

# Copy experiment scripts
echo -e "${YELLOW}Step 4: Copying experiment scripts...${NC}"
cp ../helical/cellreprogrammer/geneformer/experiments/*.py experiments/oskm/ 2>/dev/null || true

# Copy Norman scripts if they exist
if [ -d "../helical/cellreprogrammer/geneformer/experiments/" ]; then
    find ../helical/cellreprogrammer/geneformer/experiments/ -name "*norman*" -exec cp {} experiments/norman/ \;
fi

echo -e "${GREEN}✓ Directory structure created${NC}"
echo -e "${GREEN}✓ Experiment scripts copied${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. cd cellreprogrammer"
echo "  2. Review the copied files in experiments/"
echo "  3. Run: python ~/GitHub/helical/fix_imports.py"
echo "  4. Create virtual environment and install"
echo ""
echo -e "${GREEN}Migration preparation complete!${NC}"

