#!/usr/bin/env python3
"""Test runner for mixedbread-ai-haystack"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run pytest with provided arguments"""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Parse custom test types
    args = sys.argv[1:]
    pytest_args = []
    
    if args and args[0] in ["unit", "integration", "all"]:
        test_type = args[0]
        remaining_args = args[1:]
        
        if test_type == "unit":
            pytest_args = ["-m", "not integration"]
        elif test_type == "integration":
            pytest_args = ["-m", "integration"]
        elif test_type == "all":
            pytest_args = []
        
        pytest_args.extend(remaining_args)
    else:
        # Pass through all arguments as-is
        pytest_args = args
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"] + pytest_args
    
    # Run pytest
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()