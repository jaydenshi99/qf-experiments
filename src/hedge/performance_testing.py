"""
Performance testing section for comparing analytical vs Monte Carlo results.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.hedge.calculations import (
    get_profit_distribution_analytical,
    get_profit_distribution_monte_carlo,
    get_expected_profit,
    get_volatility
)


def render_performance_testing_section(bets, p):
    """Render the performance testing section."""
    if bets:
        st.markdown("---")
        st.markdown("### Performance Testing")
    
    if bets:
        st.markdown("""
        Allocate money to your bets and compare the analytical calculation 
        (exact using binomial distribution) vs Monte Carlo simulation.
        """)
        
        # Allocation inputs
        st.markdown("#### Allocations")
        allocations = []
        
        cols = st.columns(len(bets))
        for i, (col, bet) in enumerate(zip(cols, bets)):
            with col:
                # Get bet description for help text
                if 'payoff' in bet:
                    bet_desc = bet['payoff']
                else:
                    bet_desc = bet.get('condition', 'N/A')
                
                allocation = st.number_input(
                    f"Bet {i+1}",
                    min_value=0.0,
                    value=100.0,
                    step=10.0,
                    format="%.2f",
                    key=f"allocation_{i}",
                    help=bet_desc
                )
                allocations.append(allocation)
        
        # Simulation parameters
        col1, col2 = st.columns(2)
        with col1:
            n_simulations = st.number_input(
                "Number of Simulations",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=10000,
                help="More simulations = more accurate but slower"
            )
        with col2:
            seed = st.number_input(
                "Random Seed",
                min_value=0,
                value=42,
                help="For reproducible results"
            )
        
        if st.button("Calculate Performance", type="primary", key="calculate_performance"):
            # Calculate using both methods
            try:
                # Set random seed for reproducibility
                np.random.seed(seed)
                
                # Analytical calculation
                analytical_dist = get_profit_distribution_analytical(
                    bets, 
                    allocations, 
                    p
                )
                analytical_expected = get_expected_profit(analytical_dist)
                total_invested = sum(allocations)
                analytical_volatility = get_volatility(analytical_dist, total_invested)
                
                # Monte Carlo simulation
                mc_dist = get_profit_distribution_monte_carlo(
                    bets,
                    allocations,
                    p,
                    n_simulations=n_simulations
                )
                mc_expected = get_expected_profit(mc_dist)
                mc_volatility = get_volatility(mc_dist, total_invested)
                
                # Convert distribution to array for plotting and stats
                profits_array = np.array(list(mc_dist.keys()))
                probs_array = np.array(list(mc_dist.values()))
                # Sample from distribution for histogram
                simulated_profits = np.random.choice(profits_array, size=n_simulations, p=probs_array)
                
                # Display results side by side
                st.markdown("#### Results Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Analytical (Exact)")
                    st.metric(
                        "Expected Profit",
                        f"${analytical_expected:.2f}"
                    )
                    st.metric(
                        "Expected Return",
                        f"{(analytical_expected / total_invested * 100):.2f}%" if total_invested > 0 else "0%"
                    )
                    st.metric(
                        "Volatility",
                        f"{analytical_volatility:.2f}%"
                    )
                    st.metric(
                        "Total Invested",
                        f"${total_invested:.2f}"
                    )
                
                with col2:
                    st.markdown("##### Monte Carlo Simulation")
                    mean_profit = np.mean(simulated_profits)
                    std_profit = np.std(simulated_profits)
                    stderr_profit = std_profit / np.sqrt(n_simulations)
                    
                    st.metric(
                        "Mean Profit",
                        f"${mean_profit:.2f} ± {1.96 * stderr_profit:.2f}",
                        help="95% confidence interval"
                    )
                    st.metric(
                        "Mean Return",
                        f"{(mean_profit / total_invested * 100):.2f}%" if total_invested > 0 else "0%"
                    )
                    st.metric(
                        "Volatility",
                        f"{mc_volatility:.2f}%"
                    )
                    prob_profit = np.mean(simulated_profits > 0)
                    st.metric(
                        "Probability of Profit",
                        f"{prob_profit * 100:.1f}%"
                    )
                
                # Accuracy comparison
                st.markdown("#### Accuracy Check")
                difference = abs(analytical_expected - mean_profit)
                relative_error = (difference / abs(analytical_expected) * 100) if analytical_expected != 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Absolute Difference", f"${difference:.2f}")
                with col2:
                    st.metric("Relative Error", f"{relative_error:.3f}%")
                with col3:
                    if relative_error < 1:
                        st.success("Excellent agreement!")
                    elif relative_error < 5:
                        st.info("Good agreement")
                    else:
                        st.warning("Consider more simulations")
                
                # Distribution plot
                st.markdown("#### Profit Distribution (Simulation)")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                ax.hist(simulated_profits, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(mean_profit, color='red', linestyle='--', linewidth=2, label=f'MC Mean: ${mean_profit:.2f}')
                ax.axvline(analytical_expected, color='green', linestyle='--', linewidth=2, label=f'Analytical: ${analytical_expected:.2f}')
                ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
                
                ax.set_xlabel('Profit ($)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title('Distribution of Simulated Profits', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error calculating performance: {str(e)}")
                st.exception(e)

