"""
Binomial Options Pricing Model Implementation

This module contains the core BinomialModel class for pricing options using
the binomial lattice method.
"""

import numpy as np
import math
from typing import Literal, Optional


class BinomialModel:
    """
    A class to implement the binomial options pricing model.
    
    The binomial model is a discrete-time model for pricing options that assumes
    the underlying asset price can move up or down by specific factors over
    each time step.
    
    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price of the option
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility of the underlying asset (annual)
    n_steps : int
        Number of time steps in the binomial tree
    option_type : str, optional
        Type of option: 'call' or 'put' (default: 'call')
    
    Attributes
    ----------
    dt : float
        Time step size (T / n_steps)
    u : float
        Up factor for stock price movement
    d : float
        Down factor for stock price movement
    p : float
        Risk-neutral probability of up movement
    q : float
        Risk-neutral probability of down movement (1 - p)
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int,
        option_type: Literal['call', 'put'] = 'call',
        option_style: Literal['european', 'american'] = 'european'
    ):
        """
        Initialize the BinomialModel with the given parameters.
        
        Parameters
        ----------
        S0 : float
            Initial stock price
        K : float
            Strike price of the option
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annual)
        sigma : float
            Volatility of the underlying asset (annual)
        n_steps : int
            Number of time steps in the binomial tree
        option_type : str, optional
            Type of option: 'call' or 'put' (default: 'call')
        option_style : str, optional
            Style of option: 'european' or 'american' (default: 'european')
        
        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        # Validate inputs
        self._validate_inputs(S0, K, T, r, sigma, n_steps, option_type, option_style)
        
        # Store basic parameters
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.n_steps = int(n_steps)
        self.option_type = option_type.lower()
        self.option_style = option_style.lower()
        
        # Calculate derived parameters
        self.dt = self.T / self.n_steps
        
        # Calculate up and down factors using Cox-Ross-Rubinstein parameters
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = math.exp(-self.sigma * math.sqrt(self.dt))
        
        # Calculate risk-neutral probabilities
        self.p = (math.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.q = 1 - self.p
        
        # Validate that probabilities are valid
        if self.p < 0 or self.p > 1:
            raise ValueError(
                f"Invalid risk-neutral probability p={self.p:.4f}. "
                f"This can happen when dt is too large or parameters are inconsistent. "
                f"Try increasing n_steps or adjusting other parameters."
            )
        
        # Initialize tree storage (will be populated later)
        self.tree = None
        
    def _validate_inputs(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int,
        option_type: str,
        option_style: str
    ) -> None:
        """
        Validate input parameters.
        
        Parameters
        ----------
        S0, K, T, r, sigma, n_steps, option_type
            Parameters to validate
            
        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        if S0 <= 0:
            raise ValueError("Initial stock price S0 must be positive")
        
        if K <= 0:
            raise ValueError("Strike price K must be positive")
        
        if T <= 0:
            raise ValueError("Time to maturity T must be positive")
        
        if r < 0:
            raise ValueError("Risk-free rate r should be non-negative")
        
        if sigma <= 0:
            raise ValueError("Volatility sigma must be positive")
        
        if n_steps <= 0:
            raise ValueError("Number of steps n_steps must be positive")
        
        if not isinstance(n_steps, (int, np.integer)):
            raise ValueError("Number of steps n_steps must be an integer")
        
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        if option_style.lower() not in ['european', 'american']:
            raise ValueError("Option style must be 'european' or 'american'")
    
    def get_model_info(self) -> dict:
        """
        Get a summary of the model parameters.
        
        Returns
        -------
        dict
            Dictionary containing all model parameters and derived values
        """
        return {
            'S0': self.S0,
            'K': self.K,
            'T': self.T,
            'r': self.r,
            'sigma': self.sigma,
            'n_steps': self.n_steps,
            'option_type': self.option_type,
            'option_style': self.option_style,
            'dt': self.dt,
            'u': self.u,
            'd': self.d,
            'p': self.p,
            'q': self.q
        }
    
    def __repr__(self) -> str:
        """String representation of the BinomialModel."""
        return (
            f"BinomialModel(S0={self.S0}, K={self.K}, T={self.T}, "
            f"r={self.r}, sigma={self.sigma}, n_steps={self.n_steps}, "
            f"option_type='{self.option_type}')"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Binomial Options Pricing Model\n"
            f"Stock Price: ${self.S0:.2f}\n"
            f"Strike Price: ${self.K:.2f}\n"
            f"Time to Maturity: {self.T:.2f} years\n"
            f"Risk-free Rate: {self.r:.1%}\n"
            f"Volatility: {self.sigma:.1%}\n"
            f"Steps: {self.n_steps}\n"
            f"Option Type: {self.option_type.title()}\n"
            f"Up Factor: {self.u:.4f}\n"
            f"Down Factor: {self.d:.4f}\n"
            f"Risk-neutral Probability: {self.p:.4f}"
        )
    
    def build_stock_price_tree(self):
        """Build the stock price tree."""
        try:
            from .tree import BinomialTree
        except ImportError:
            from tree import BinomialTree
        
        if self.tree is None:
            self.tree = BinomialTree(self)
        
        self.tree.build_stock_price_tree()
    
    def build_option_price_tree(self):
        """Build the option price tree using backward induction."""
        if self.tree is None:
            self.build_stock_price_tree()
        
        self.tree.build_option_price_tree()
    
    def get_option_price(self):
        """Get the current option price."""
        if self.tree is None:
            self.build_stock_price_tree()
        
        if self.tree.root.option_price is None:
            self.build_option_price_tree()
        
        return self.tree.get_option_price()
