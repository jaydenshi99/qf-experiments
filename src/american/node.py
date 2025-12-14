"""
Simple Node class for Binomial Tree Structure

Minimal implementation focused on core functionality.
Visualization properties can be added later.
"""

from typing import Optional


class BinomialNode:
    """
    A simple node in the binomial tree.
    
    Attributes
    ----------
    time_step : int
        Time step (0 to n_steps)
    node_index : int
        Node index within the time step (0 to time_step)
    stock_price : float
        Stock price at this node
    option_price : Optional[float]
        Option price at this node (calculated later)
    parent : Optional[BinomialNode]
        Parent node (None for root)
    up_child : Optional[BinomialNode]
        Up movement child node
    down_child : Optional[BinomialNode]
        Down movement child node
    """
    
    def __init__(
        self, 
        time_step: int, 
        node_index: int, 
        stock_price: float,
        option_price: Optional[float] = None
    ):
        self.time_step = time_step
        self.node_index = node_index
        self.stock_price = stock_price
        self.option_price = option_price
        
        # Tree relationships
        self.parent: Optional[BinomialNode] = None
        self.up_child: Optional[BinomialNode] = None
        self.down_child: Optional[BinomialNode] = None
    
    def __repr__(self) -> str:
        return (
            f"BinomialNode(t={self.time_step}, i={self.node_index}, "
            f"S={self.stock_price:.2f}, O={self.option_price})"
        )
    
    def __str__(self) -> str:
        return f"Node(t={self.time_step}, i={self.node_index}): S=${self.stock_price:.2f}"
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (final time step)."""
        return self.up_child is None and self.down_child is None
    
    def is_root(self) -> bool:
        """Check if this is the root node (time step 0)."""
        return self.parent is None
    
    def get_payoff(self, strike_price: float, option_type: str) -> float:
        """
        Calculate the payoff at this node.
        
        Parameters
        ----------
        strike_price : float
            Strike price of the option
        option_type : str
            'call' or 'put'
            
        Returns
        -------
        float
            Payoff value
        """
        if option_type.lower() == 'call':
            return max(self.stock_price - strike_price, 0)
        elif option_type.lower() == 'put':
            return max(strike_price - self.stock_price, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
    
    def get_exercise_value(self, strike_price: float, option_type: str) -> float:
        """
        Calculate the exercise value for American options.
        
        Parameters
        ----------
        strike_price : float
            Strike price of the option
        option_type : str
            'call' or 'put'
            
        Returns
        -------
        float
            Exercise value
        """
        return self.get_payoff(strike_price, option_type)
