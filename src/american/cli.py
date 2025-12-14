"""
Command Line Interface for Binomial Options Pricing Model

This provides a simple CLI for testing the model without the web interface.
Run with: python src/cli.py
"""

import sys
import os
import argparse

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.american.binomial_model import BinomialModel

def main():
    """Command line interface for the binomial model."""
    
    parser = argparse.ArgumentParser(
        description="Binomial Options Pricing Model CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--S0', type=float, default=100.0,
                       help='Initial stock price')
    parser.add_argument('--K', type=float, default=105.0,
                       help='Strike price')
    parser.add_argument('--T', type=float, default=0.25,
                       help='Time to maturity (years)')
    parser.add_argument('--r', type=float, default=0.05,
                       help='Risk-free rate')
    parser.add_argument('--sigma', type=float, default=0.2,
                       help='Volatility')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of time steps')
    parser.add_argument('--type', choices=['call', 'put'], default='call',
                       help='Option type')
    parser.add_argument('--style', choices=['european', 'american'], default='european',
                       help='Option style')
    
    args = parser.parse_args()
    
    try:
        # Create the model
        model = BinomialModel(
            S0=args.S0,
            K=args.K,
            T=args.T,
            r=args.r,
            sigma=args.sigma,
            n_steps=args.steps,
            option_type=args.type,
            option_style=args.style
        )
        
        # Display results
        print("=" * 60)
        print("BINOMIAL OPTIONS PRICING MODEL")
        print("=" * 60)
        print(model)
        print("=" * 60)
        
        # Calculate option price
        option_price = model.get_option_price()
        print(f"Option Price: ${option_price:.4f}")
        
        # Show some tree statistics
        print(f"Tree Nodes: {len(model.tree.nodes)}")
        print(f"Terminal Nodes: {len(model.tree.get_terminal_nodes())}")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
