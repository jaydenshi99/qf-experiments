"""
Parsing functions for bet conditions and payoff functions.

This module handles parsing of:
- Simple bet conditions (e.g., "H_5 >= 3")
- Payoff function expressions (e.g., "max(H_5 - 3, 0) * 0.5 * I")
"""

import re
import ast


def parse_condition(condition):
    """
    Parse a condition string into its components.
    
    Args:
        condition: String like "H_5 >= 3" or "H_10 == 7"
    
    Returns:
        tuple: (t, operator, k) where:
            - t: number of coin flips
            - operator: comparison operator ('==', '>=', '<=', '>', '<', '!=')
            - k: threshold value
    
    Example:
        >>> parse_condition("H_10 >= 6")
        (10, '>=', 6)
    """
    # Match pattern: H_<number> <operator> <number>
    pattern = r'H_(\d+)\s*(==|>=|<=|>|<|!=)\s*(\d+)'
    match = re.match(pattern, condition.strip())
    
    if not match:
        raise ValueError(f"Invalid condition format: {condition}")
    
    t = int(match.group(1))
    operator = match.group(2)
    k = int(match.group(3))
    
    return t, operator, k


def parse_payoff_function(payoff_expr):
    """
    Parse a payoff function expression into an AST tree for efficient reuse.
    
    Args:
        payoff_expr: String expression like "max(H_5 - 3, 0) * 0.5 * I - 0.1 * I"
    
    Returns:
        AST node: parsed expression tree
    
    Example:
        >>> tree = parse_payoff_function("max(H_5 - 3, 0) * 0.5 * I")
        >>> type(tree)
        <class '_ast.BinOp'>
    """
    try:
        # Strip whitespace to avoid "unexpected indent" errors
        payoff_expr = payoff_expr.strip()
        tree = ast.parse(payoff_expr, mode='eval')
        return tree.body
    except Exception as e:
        raise ValueError(f"Error parsing payoff function '{payoff_expr}': {str(e)}")


def get_max_time_from_payoff(payoff_expr):
    """
    Extract the maximum time t from a payoff expression.
    
    Args:
        payoff_expr: String expression like "max(H_5 - 3, 0) * 0.5"
    
    Returns:
        int: maximum time t needed, or 0 if none found
    
    Example:
        >>> get_max_time_from_payoff("max(H_5 - 3, 0) * 0.5 + H_10")
        10
    """
    matches = re.findall(r'H_(\d+)', payoff_expr)
    if matches:
        return max(int(t) for t in matches)
    return 0

