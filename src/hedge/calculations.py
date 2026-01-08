import numpy as np

from .parsers import parse_condition, parse_payoff_function, get_max_time_from_payoff
from .evaluators import evaluate_payoff_ast, evaluate_condition

def get_ht_from_binary(binary_string):
    """Convert binary string to cumulative heads."""
    flips = [int(bit) for bit in binary_string]
    return np.cumsum(flips)


def get_max_time_from_bets(bets):
    """Find max time from bets."""
    n = 0
    for bet in bets:
        if 'payoff' in bet:
            t = get_max_time_from_payoff(bet['payoff'])
            n = max(n, t)
        else:
            t, _, _ = parse_condition(bet['condition'])
            n = max(n, t)
    return n


def _get_max_time_from_bets(bets):
    """Backward compatibility alias."""
    return get_max_time_from_bets(bets)


def get_profit_distribution_analytical(bets, allocations, p):
    """Calculate exact profit distribution by enumerating all outcomes."""

    # parse bets
    parsed_bets = []
    for bet in bets:
        if 'payoff' in bet:
            parsed_bets.append({'payoff': parse_payoff_function(bet['payoff'])})
        elif 'condition' in bet:
            parsed_bets.append(bet)

    # find max time
    n = _get_max_time_from_bets(bets)

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
                    # Decimal odds: total payout = allocation * odds
                    # Profit = total payout - stake = allocation * (odds - 1)
                    bet_profit = allocation * (bet['odds'] - 1)
                else:
                    bet_profit = -allocation
            total_profit += bet_profit
        
        # Add to distribution
        profit_distribution[total_profit] = profit_distribution.get(total_profit, 0) + sequence_prob

    return profit_distribution

def get_profit_distribution_monte_carlo(bets, allocations, p, n_simulations=3000):
    """Calculate profit distribution via Monte Carlo."""
    # parse bets
    parsed_bets = []
    for bet in bets:
        if 'payoff' in bet:
            parsed_bets.append({'payoff': parse_payoff_function(bet['payoff'])})
        elif 'condition' in bet:
            parsed_bets.append(bet)

    # find max time
    n = _get_max_time_from_bets(bets)

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
                    # Decimal odds: total payout = allocation * odds
                    # Profit = total payout - stake = allocation * (odds - 1)
                    bet_profit = allocation * (bet['odds'] - 1)
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
    """Calculate expected profit."""
    return sum(profit * probability for profit, probability in profit_distribution.items())

def get_volatility(profit_distribution, total_invested):
    """Calculate volatility (std dev of returns)."""
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


def get_profit_distribution(bets, allocations, p, n_simulations=5000, analytical_threshold=18):
    """Auto-select analytical or Monte Carlo method."""
    n = _get_max_time_from_bets(bets)
    
    if n <= analytical_threshold:
        return get_profit_distribution_analytical(bets, allocations, p)
    else:
        return get_profit_distribution_monte_carlo(bets, allocations, p, n_simulations)


def get_bet_statistics_per_dollar(bets, p, n_simulations=5000, analytical_threshold=18):
    """Calculate per-dollar statistics for bets."""
    n_bets = len(bets)
    expected_profit_per_dollar = []
    variance_per_dollar = []
    
    # Calculate E_i and Var_i for each bet individually (with $1)
    for i in range(n_bets):
        allocations = [0.0] * n_bets
        allocations[i] = 1.0
        
        dist = get_profit_distribution(bets, allocations, p, n_simulations, analytical_threshold)
        expected_profit = get_expected_profit(dist)
        expected_profit_per_dollar.append(expected_profit)
        
        # Variance of return per dollar
        expected_return = expected_profit / 1.0
        variance = sum(prob * ((profit / 1.0) - expected_return) ** 2 
                      for profit, prob in dist.items())
        variance_per_dollar.append(variance)
    
    # Calculate covariance matrix
    covariance_matrix = np.zeros((n_bets, n_bets))
    
    # Diagonal is variance
    for i in range(n_bets):
        covariance_matrix[i, i] = variance_per_dollar[i]
    
    # Off-diagonal
    # When both bets have $1 each, return per dollar = (R_i + R_j) / 2
    # Var((R_i + R_j)/2) = (1/4) * Var(R_i + R_j)
    # Var(R_i + R_j) = Var(R_i) + Var(R_j) + 2*Cov(R_i, R_j)
    # Var((R_i + R_j)/2) = (1/4) * [Var(R_i) + Var(R_j) + 2*Cov(R_i, R_j)]
    # 4*Var((R_i + R_j)/2) = Var(R_i) + Var(R_j) + 2*Cov(R_i, R_j)
    # Cov(R_i, R_j) = (4*Var((R_i+R_j)/2) - Var(R_i) - Var(R_j)) / 2
    for i in range(n_bets):
        for j in range(i + 1, n_bets):
            # Calculate both bets with $1 each to get Var((R_i + R_j)/2)
            allocations = [0.0] * n_bets
            allocations[i] = 1.0
            allocations[j] = 1.0
            
            dist_both = get_profit_distribution(bets, allocations, p, n_simulations, analytical_threshold)
            expected_profit_both = get_expected_profit(dist_both)
            expected_return_both = expected_profit_both / 2.0
            
            # Var((R_i + R_j)/2)
            variance_both = sum(prob * ((profit / 2.0) - expected_return_both) ** 2 
                              for profit, prob in dist_both.items())
            
            # Cov(R_i, R_j) = (4*Var((R_i+R_j)/2) - Var(R_i) - Var(R_j)) / 2
            covariance = (4.0 * variance_both - variance_per_dollar[i] - variance_per_dollar[j]) / 2.0
            
            covariance_matrix[i, j] = covariance
            covariance_matrix[j, i] = covariance
    
    return {
        'expected_profit_per_dollar': expected_profit_per_dollar,
        'variance_per_dollar': variance_per_dollar,
        'covariance_matrix': covariance_matrix
    }


def calculate_portfolio_stats(allocations, stats_per_dollar):
    """Calculate portfolio stats from per-dollar stats."""
    allocations = np.array(allocations)
    E = np.array(stats_per_dollar['expected_profit_per_dollar'])
    Cov = stats_per_dollar['covariance_matrix']
    
    total = np.sum(allocations)
    if total == 0:
        return 0.0, 0.0
    
    # Expected profit
    expected_profit = np.dot(allocations, E)
    
    # Variance of returns
    variance_of_returns = np.dot(allocations, np.dot(Cov, allocations)) / (total ** 2)
    volatility_percent = np.sqrt(max(0, variance_of_returns)) * 100
    
    return expected_profit, volatility_percent


def get_optimal_allocation(bets, p, target_return, n_simulations=5000, analytical_threshold=18):
    """Calculate optimal allocation for target return."""
    # Get statistics per dollar
    stats_per_dollar = get_bet_statistics_per_dollar(bets, p, n_simulations, analytical_threshold)
    
    # Extract covariance matrix and expected returns
    sigma = np.array(stats_per_dollar['covariance_matrix'])
    mu = np.array(stats_per_dollar['expected_profit_per_dollar']).reshape(-1, 1)
    
    # Create vector of ones
    ones = np.ones(len(mu)).reshape(-1, 1)
    
    # Calculate inverse with error handling
    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        sigma_str = np.array2string(
            sigma,
            precision=6,
            suppress_small=True,
            max_line_width=120
        )
        raise ValueError(
            "**Cannot calculate optimal allocation:** The covariance matrix is singular.\n\n"
            "**This typically happens when:**\n"
            " - Two or more bets are perfectly correlated (identical payoff structures)\n"
            " - One bet's returns can be perfectly replicated by a linear combination of other bets\n"
            " - Numerical precision issues with very similar bets\n\n"
            f"**Covariance matrix (σ):**\n{sigma_str}\n\n"
            "Please ensure your bets have different, independent payoff structures."
        )
    
    # Calculate A, B, C
    A = (ones.T @ sigma_inv @ ones).item()
    B = (ones.T @ sigma_inv @ mu).item()
    C = (mu.T @ sigma_inv @ mu).item()
    
    # Calculate D
    D = A * C - B**2
    
    if abs(D) < 1e-10:
        raise ValueError("Covariance matrix is singular or constraints are incompatible")
    
    # Calculate alpha and beta
    alpha = (C - B * target_return) / D
    beta = (A * target_return - B) / D
    
    # Calculate optimal allocation
    optimal_allocation = alpha * (sigma_inv @ ones) + beta * (sigma_inv @ mu)
    
    return optimal_allocation