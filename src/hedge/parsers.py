"""
Parsing functions for bet conditions and payoff functions.

This module handles parsing of:
- Simple bet conditions (e.g., "H_5 >= 3")
- Payoff function expressions (e.g., "max(H_5 - 3, 0) * 0.5 * I")
"""

import re
import ast


def parse_condition(condition):
    """Parse condition string to (t, operator, k)."""
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
    """Parse payoff expression to AST tree."""
    try:
        # Strip whitespace to avoid "unexpected indent" errors
        payoff_expr = payoff_expr.strip()
        tree = ast.parse(payoff_expr, mode='eval')
        return tree.body
    except Exception as e:
        raise ValueError(f"Error parsing payoff function '{payoff_expr}': {str(e)}")


def get_max_time_from_payoff(payoff_expr):
    """Extract max time from payoff expression."""
    matches = re.findall(r'H_(\d+)', payoff_expr)
    if matches:
        return max(int(t) for t in matches)
    return 0

