import sys
from pathlib import Path
import os

# Project root setup
HELICAL_ROOT = Path("/home/atayyr/GitHub/helical").resolve()
CELL_ROOT = Path("/home/atayyr/GitHub/cellreprogrammer").resolve()

sys.path.insert(0, str(HELICAL_ROOT))
sys.path.insert(0, str(CELL_ROOT))

def test_stack_targeted():
    print("Performing targeted verification of Stack components...")
    try:
        print("\n1. Importing Helical Stack model...")
        from helical.models.stack import Stack, StackConfig
        print("✓ Success")
        
        print("\n2. Importing CellReprogrammer StackAdapter...")
        # We might need to mock base_adapter if it has issues, but let's try
        from src.adapters.stack_adapter import StackAdapter
        print("✓ Success")
        
        print("\n3. Verifying StackConfig defaults...")
        config = StackConfig(checkpoint_path="test.ckpt", genelist_path="test.pkl")
        assert config.config["checkpoint_path"] == "test.ckpt"
        print("✓ Success")
        
        return True
    except Exception as e:
        print(f"Error during targeted verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stack_targeted()
    if success:
        print("\nTARGETED VERIFICATION SUCCESSFUL")
        sys.exit(0)
    else:
        print("\nTARGETED VERIFICATION FAILED")
        sys.exit(1)
