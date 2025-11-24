#!/usr/bin/env python3
"""
Fix imports in migrated cellreprogrammer scripts
Replaces sys.path hacks and relative imports with proper helical imports
"""

import os
import re
from pathlib import Path

def fix_script(filepath: Path) -> bool:
    """Fix imports in a single Python script"""
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        changes = []
        
        # Remove sys.path.insert hacks
        if 'sys.path.insert' in content:
            content = re.sub(
                r'sys\.path\.insert\(0,\s*str\(.*?\)\)\n?',
                '',
                content
            )
            changes.append("Removed sys.path.insert")
        
        # Fix path calculations for BASE_DIR
        if 'BASE_DIR = script_dir.parent.parent' in content:
            content = content.replace(
                'BASE_DIR = script_dir.parent.parent',
                '# BASE_DIR not needed - using environment variable or relative paths'
            )
            changes.append("Removed BASE_DIR calculation")
        
        # Fix CELLREPROGRAMMER_DIR to use PROJECT_ROOT
        if 'CELLREPROGRAMMER_DIR' in content:
            # Add PROJECT_ROOT calculation if not present
            if 'PROJECT_ROOT' not in content:
                script_dir_match = re.search(r'script_dir = Path\(__file__\)\.resolve\(\)\.parent', content)
                if script_dir_match:
                    # Add after script_dir definition
                    content = content.replace(
                        'script_dir = Path(__file__).resolve().parent',
                        'script_dir = Path(__file__).resolve().parent\n'
                        'PROJECT_ROOT = script_dir.parent.parent  # cellreprogrammer repo root'
                    )
            
            # Replace CELLREPROGRAMMER_DIR references
            content = content.replace(
                'CELLREPROGRAMMER_DIR / "data"',
                'PROJECT_ROOT / "data"'
            )
            content = content.replace(
                'CELLREPROGRAMMER_DIR / "results"',
                'PROJECT_ROOT / "results"'
            )
            content = re.sub(
                r'CELLREPROGRAMMER_DIR = .*?\n',
                '',
                content
            )
            changes.append("Fixed data/results paths")
        
        # Fix helical imports to be simpler
        # Already correct: from helical.models.geneformer import ...
        # Just ensure sys import is present if needed
        
        # Add comment about data paths
        if 'DATA_DIR' in content and '# Data paths' not in content:
            content = re.sub(
                r'(DATA_DIR = .*?)\n',
                r'# Data paths (relative to project root)\n\1\n',
                content,
                count=1
            )
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True, changes
        
        return False, []
    
    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        return False, []

def main():
    print("=" * 80)
    print("Fixing imports in cellreprogrammer scripts")
    print("=" * 80)
    print()
    
    # Find all Python files in experiments/
    repo_root = Path.cwd()
    if not (repo_root / 'experiments').exists():
        print("Error: experiments/ directory not found")
        print("Please run this from the cellreprogrammer repository root")
        return
    
    python_files = list((repo_root / 'experiments').rglob('*.py'))
    
    if not python_files:
        print("No Python files found in experiments/")
        return
    
    print(f"Found {len(python_files)} Python files")
    print()
    
    modified = 0
    for filepath in python_files:
        rel_path = filepath.relative_to(repo_root)
        print(f"Processing: {rel_path}")
        
        changed, changes = fix_script(filepath)
        if changed:
            modified += 1
            for change in changes:
                print(f"  ✓ {change}")
        else:
            print(f"  - No changes needed")
        print()
    
    print("=" * 80)
    print(f"Complete! Modified {modified}/{len(python_files)} files")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review the changes: git diff")
    print("  2. Create requirements.txt and pyproject.toml")
    print("  3. Set up virtual environment:")
    print("     python -m venv .venv")
    print("     source .venv/bin/activate")
    print("     pip install helical")
    print("  4. Test the scripts!")
    print()

if __name__ == '__main__':
    main()



