#!/usr/bin/env python
"""Quick validation script to test CellReprogrammer setup.

This script verifies that all imports work correctly and basic functionality is available.
"""

import sys
from pathlib import Path

# Add parent directory to path
base_path = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(base_path))
sys.path.insert(0, str(base_path / "cellreprogrammer"))

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 80)
    print("CellReprogrammer Setup Validation")
    print("=" * 80)
    
    issues = []
    
    # Test 1: Core imports
    print("\n1. Testing core imports...")
    try:
        from cellreprogrammer.src.models import ModelFactory, get_available_models
        print("   ✓ Model imports successful")
    except Exception as e:
        issues.append(f"Model imports failed: {e}")
        print(f"   ✗ Model imports failed: {e}")
    
    # Test 2: Perturbation imports
    print("\n2. Testing perturbation imports...")
    try:
        from cellreprogrammer.src.perturbations import BasePerturbation, OverexpressionPerturbation
        print("   ✓ Perturbation imports successful")
    except Exception as e:
        issues.append(f"Perturbation imports failed: {e}")
        print(f"   ✗ Perturbation imports failed: {e}")
    
    # Test 3: Config imports
    print("\n3. Testing config imports...")
    try:
        from cellreprogrammer.configs import load_config, validate_config, get_config_template
        print("   ✓ Config imports successful")
    except Exception as e:
        issues.append(f"Config imports failed: {e}")
        print(f"   ✗ Config imports failed: {e}")
    
    # Test 4: Helical imports
    print("\n4. Testing helical imports...")
    try:
        from helical.models.geneformer import Geneformer, GeneformerConfig
        from helical.models.base_models import HelicalRNAModel
        print("   ✓ Helical imports successful")
    except Exception as e:
        issues.append(f"Helical imports failed: {e}")
        print(f"   ✗ Helical imports failed: {e}")
    
    # Test 5: Model Factory
    print("\n5. Testing ModelFactory...")
    try:
        factory = ModelFactory()
        available_models = factory.get_available_models()
        print(f"   ✓ ModelFactory initialized")
        print(f"   Available models: {available_models}")
    except Exception as e:
        issues.append(f"ModelFactory test failed: {e}")
        print(f"   ✗ ModelFactory test failed: {e}")
    
    # Test 6: Config loading
    print("\n6. Testing config system...")
    try:
        template = get_config_template()
        print(f"   ✓ Config template available")
        print(f"   Config keys: {list(template.keys())}")
    except Exception as e:
        issues.append(f"Config test failed: {e}")
        print(f"   ✗ Config test failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    if issues:
        print("⚠️  VALIDATION INCOMPLETE")
        print(f"   {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease fix the issues above before proceeding.")
        return False
    else:
        print("✓ VALIDATION COMPLETE")
        print("   All imports and basic functionality working correctly!")
        print("\nYou're ready to start using CellReprogrammer! 🎉")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

