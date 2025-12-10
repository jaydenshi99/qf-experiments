"""
American Options Visualisation with Early Exercise Analysis

This creates a visualisation showing American options pricing with
early exercise decisions highlighted.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main function from the modular structure
from src.american.main import main

if __name__ == "__main__":
    main()
