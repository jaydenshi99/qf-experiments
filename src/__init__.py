"""
Binomial Options Pricing Model Package

A comprehensive implementation of the binomial options pricing model
with interactive visualization capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .american.binomial_model import BinomialModel
from .american.node import BinomialNode
from .american.tree import BinomialTree

# Import other modules when they are created
# from .visualization import BinomialTreeVisualizer
# from .utils import calculate_greeks, validate_parameters

__all__ = [
    "BinomialModel",
    "BinomialNode",
    "BinomialTree",
    # "BinomialTreeVisualizer", 
    # "calculate_greeks",
    # "validate_parameters"
]
