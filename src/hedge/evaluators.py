"""
Evaluation functions for bet conditions and payoff functions.

This module handles evaluation of:
- Simple bet conditions against simulated outcomes
- Payoff function expressions with H_t and I variables
"""

import ast
import operator as op
import re
from .parsers import parse_condition, parse_payoff_function


# Safe operators for AST evaluation
_SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

# Safe functions for AST evaluation
_SAFE_FUNCTIONS = {
    'max': max,
    'min': min,
    'abs': abs,
}


def _safe_eval(node, h_array, investment=None):
    """Evaluate AST node with H_t and I variables."""
    if isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.Constant):  # Python >= 3.8
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left, h_array, investment)
        right = _safe_eval(node.right, h_array, investment)
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand, h_array, investment)
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op_func(operand)
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [_safe_eval(arg, h_array, investment) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)
    elif isinstance(node, ast.Name):
        # Handle variables: H_t and I (investment)
        var_name = node.id
        if var_name == 'I':
            if investment is None:
                raise ValueError("Variable 'I' (investment) used but not provided")
            return float(investment)
        match = re.match(r'H_(\d+)', var_name)
        if match:
            t = int(match.group(1))
            # H_t is at index t-1 (H_1 at index 0, H_2 at index 1, etc.)
            array_index = t - 1
            if array_index < 0:
                raise ValueError(f"Invalid time index: H_{t} (must be >= 1)")
            if array_index >= len(h_array):
                raise ValueError(f"H_{t} not available in simulation (array length: {len(h_array)})")
            return float(h_array[array_index])
        else:
            raise ValueError(f"Unknown variable: {var_name}")
    else:
        raise ValueError(f"Unsupported AST node type: {type(node)}")


def evaluate_payoff_ast(ast_node, h_array, investment=None):
    """Evaluate pre-parsed AST."""
    return _safe_eval(ast_node, h_array, investment)


def evaluate_payoff_function(payoff_expr, h_array, investment=None):
    """Parse and evaluate payoff expression."""
    ast_node = parse_payoff_function(payoff_expr)
    return evaluate_payoff_ast(ast_node, h_array, investment)


def evaluate_condition(condition, H):
    """Evaluate condition against cumulative heads."""
    t, operator, k = parse_condition(condition)
    
    # Check if we have enough flips
    if t > len(H):
        raise ValueError(f"Not enough flips: need {t}, have {len(H)}")
    
    # Get number of heads at time t (0-indexed, so t-1)
    heads_at_t = H[t - 1]
    
    # Evaluate the condition
    if operator == '==':
        return heads_at_t == k
    elif operator == '>=':
        return heads_at_t >= k
    elif operator == '>':
        return heads_at_t > k
    elif operator == '<=':
        return heads_at_t <= k
    elif operator == '<':
        return heads_at_t < k
    elif operator == '!=':
        return heads_at_t != k
    else:
        return False

