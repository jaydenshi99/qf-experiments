"""
American vs European Options Visualizer

Entry point for the Streamlit application.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.american.main import main

if __name__ == "__main__":
    main()

