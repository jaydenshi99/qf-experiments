"""
Portfolio Hedging Calculations

Functions for calculating expected profit and risk metrics for bet portfolios.
"""

import numpy as np
import re
from scipy.stats import binom

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


def calculate_win_probability(t, operator, k, p):
    """
    Calculate probability that a bet wins using binomial distribution.
    
    Args:
        t: number of coin flips
        operator: comparison operator
        k: threshold value
        p: probability of heads
    
    Returns:
        float: probability that the condition is satisfied
    """
    if operator == '==':
        return binom.pmf(k, t, p)
    elif operator == '>=':
        return 1 - binom.cdf(k - 1, t, p)
    elif operator == '>':
        return 1 - binom.cdf(k, t, p)
    elif operator == '<=':
        return binom.cdf(k, t, p)
    elif operator == '<':
        return binom.cdf(k - 1, t, p)
    elif operator == '!=':
        return 1 - binom.pmf(k, t, p)
    else:
        raise ValueError(f"Unsupported operator: {operator}")

def evaluate_condition(condition, H):
    """
    Evaluate a condition given a sequence of cumulative heads.
    
    Args:
        condition: String like "H_5 >= 3"
        H: numpy array where H[t] = number of heads up to time t (0-indexed)
    
    Returns:
        bool: whether the condition is satisfied
    """
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

def calculate_expected_profit_analytical(bets, allocations, p):
    """
    Calculate expected profit analytically using binomial distribution.
    
    Args:
        bets: list of dicts with 'condition' and 'odds' keys
        allocations: list of dollar amounts for each bet
        p: probability of heads
    
    Returns:
        dict with:
            - total_expected_profit: expected profit across all bets
            - total_invested: total amount invested
            - expected_return_pct: expected return as percentage
            - bet_details: list of dicts with per-bet statistics
    
    Example:
        >>> bets = [{'condition': 'H_2 == 1', 'odds': 2.0}]
        >>> allocations = [100]
        >>> calculate_expected_profit_analytical(bets, allocations, 0.5)
        {'total_expected_profit': 0.0, 'total_invested': 100, ...}
    """
    total_expected = 0
    total_invested = sum(allocations)
    bet_details = []
    
    for bet, allocation in zip(bets, allocations):
        try:
            # Parse the condition
            t, operator, k = parse_condition(bet['condition'])
            
            # Calculate win probability
            prob_win = calculate_win_probability(t, operator, k, p)
            
            # Calculate expected profit for this bet
            # Win: get back allocation * odds
            # Lose: lose the allocation
            win_amount = allocation * bet['odds']
            lose_amount = -allocation
            expected_profit = prob_win * win_amount + (1 - prob_win) * lose_amount
            
            total_expected += expected_profit
            
            bet_details.append({
                'condition': bet['condition'],
                'odds': bet['odds'],
                'allocation': allocation,
                'prob_win': prob_win,
                'expected_profit': expected_profit,
                'expected_return_pct': (expected_profit / allocation * 100) if allocation > 0 else 0
            })
            
        except Exception as e:
            bet_details.append({
                'condition': bet['condition'],
                'odds': bet['odds'],
                'allocation': allocation,
                'error': str(e)
            })
    
    return {
        'total_expected_profit': total_expected,
        'total_invested': total_invested,
        'expected_return_pct': (total_expected / total_invested * 100) if total_invested > 0 else 0,
        'bet_details': bet_details
    }


def calculate_expected_profit_simulation(bets, allocations, p, n_simulations=10000, seed=None):
    """
    Calculate expected profit using Monte Carlo simulation.
    
    Args:
        bets: list of dicts with 'condition' and 'odds' keys
        allocations: list of dollar amounts for each bet
        p: probability of heads
        n_simulations: number of simulations to run
        seed: random seed for reproducibility
    
    Returns:
        numpy array of profits
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Find maximum flips needed across all bets
    max_t = 0
    for bet in bets:
        try:
            t, _, _ = parse_condition(bet['condition'])
            max_t = max(max_t, t)
        except:
            pass
    
    if max_t == 0:
        # No valid bets
        return {
            'mean_profit': 0,
            'std_profit': 0,
            'total_invested': sum(allocations),
            'probability_of_profit': 0,
            'profit_distribution': np.array([]),
            'percentile_5': 0,
            'percentile_95': 0
        }
    
    # Run simulations
    profits = []
    
    for _ in range(n_simulations):
        # Simulate coin flips (True = heads, False = tails)
        flips = np.random.random(max_t) < p
        
        # Calculate cumulative heads at each time
        H = np.cumsum(flips)
        
        # Calculate profit for this simulation
        profit = 0
        for bet, allocation in zip(bets, allocations):
            try:
                if evaluate_condition(bet['condition'], H):
                    # Bet wins: receive allocation * odds
                    profit += allocation * bet['odds']
                else:
                    # Bet loses: lose the allocation
                    profit -= allocation
            except:
                # If evaluation fails, treat as loss
                profit -= allocation
        
        profits.append(profit)
    
    profits = np.array(profits)
    
    return profits