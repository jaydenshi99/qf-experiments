import numpy as np

from .parsers import parse_condition, parse_payoff_function, get_max_time_from_payoff
from .evaluators import evaluate_payoff_ast, evaluate_condition

def get_ht_from_binary(binary_string):
    """
    Converts a binary string to a list of heads
    """
    flips = [int(bit) for bit in binary_string]
    return np.cumsum(flips)
    

def get_profit_distribution_analytical(bets, allocations, p):
    """
    Calculates the pmf of the profit distribution for a set of given bets and allocations
    
    Args:
        bets: list of dicts with either:
            - {'condition': 'H_2 == 1', 'odds': 2.0}
            - {'payoff': 'max(H_5 - 3, 0) * 0.5 * I - 0.1 * I'}
        allocations: list of dollar amounts for each bet
        p: probability of heads
    """

    # parse bets
    parsed_bets = []
    for bet in bets:
        if 'payoff' in bet:
            parsed_bets.append({'payoff': parse_payoff_function(bet['payoff'])})
        elif 'condition' in bet:
            parsed_bets.append(bet)

    # find max time
    n = 0
    for bet in bets:
        if 'payoff' in bet:
            t = get_max_time_from_payoff(bet['payoff'])
            n = max(n, t)
        else:
            t, _, _ = parse_condition(bet['condition'])
            n = max(n, t)

    profit_distribution = {}

    for h in range(2**n):
        # get H_t values
        binary_string = format(h, f'0{n}b')
        h_t = get_ht_from_binary(binary_string)
        
        num_heads = h_t[-1]  # Total heads in this sequence
        num_tails = n - num_heads
        sequence_prob = (p ** num_heads) * ((1 - p) ** num_tails)

        # Calculate total profit across all bets
        total_profit = 0
        for bet, allocation in zip(parsed_bets, allocations):
            if 'payoff' in bet:
                bet_profit = evaluate_payoff_ast(bet['payoff'], h_t, investment=allocation)
            elif 'condition' in bet:
                success = evaluate_condition(bet['condition'], h_t)
                if success:
                    bet_profit = allocation * bet['odds']
                else:
                    bet_profit = -allocation
            total_profit += bet_profit
        
        # Add to distribution
        profit_distribution[total_profit] = profit_distribution.get(total_profit, 0) + sequence_prob

    return profit_distribution

def get_profit_distribution_monte_carlo(bets, allocations, p, n_simulations=10000):
    """
    Calculates the profit distribution using Monte Carlo simulation
    """
    # parse bets
    parsed_bets = []
    for bet in bets:
        if 'payoff' in bet:
            parsed_bets.append({'payoff': parse_payoff_function(bet['payoff'])})
        elif 'condition' in bet:
            parsed_bets.append(bet)

    # find max time
    n = 0
    for bet in bets:
        if 'payoff' in bet:
            t = get_max_time_from_payoff(bet['payoff'])
            n = max(n, t)
        else:
            t, _, _ = parse_condition(bet['condition'])
            n = max(n, t)

    profit_distribution = {}
    for _ in range(n_simulations):
        flips = np.random.random(n) < p
        h_t = np.cumsum(flips.astype(int))
        
        total_profit = 0
        for bet, allocation in zip(parsed_bets, allocations):
            if 'payoff' in bet:
                bet_profit = evaluate_payoff_ast(bet['payoff'], h_t, investment=allocation)
            elif 'condition' in bet:
                success = evaluate_condition(bet['condition'], h_t)
                if success:
                    bet_profit = allocation * bet['odds']
                else:
                    bet_profit = -allocation
            total_profit += bet_profit

        # count frequency        
        profit_distribution[total_profit] = profit_distribution.get(total_profit, 0) + 1
    
    # normalise to probabilities
    total_samples = sum(profit_distribution.values())
    profit_distribution = {profit: count / total_samples 
                          for profit, count in profit_distribution.items()}
    
    return profit_distribution

def get_expected_profit(profit_distribution):
    """
    Calculates the expected profit from a profit distribution
    """
    return sum(profit * probability for profit, probability in profit_distribution.items())

def get_volatility(profit_distribution, total_invested):
    """
    Calculates the volatility (standard deviation of returns) from a profit distribution.
    """
    if total_invested == 0:
        return 0.0
    
    # Calculate expected return and expected return squared
    expected_return = 0.0
    expected_return_squared = 0.0
    
    for profit, prob in profit_distribution.items():
        return_val = profit / total_invested
        expected_return += prob * return_val
        expected_return_squared += prob * (return_val ** 2)
    
    # Variance = E[return^2] - E[return]^2
    variance = expected_return_squared - (expected_return ** 2)
    
    # Volatility = std dev of returns (as percentage)
    volatility = np.sqrt(max(0, variance)) * 100
    
    return volatility