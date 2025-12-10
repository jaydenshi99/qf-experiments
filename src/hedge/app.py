"""
Portfolio Hedging Application

A Streamlit application for analyzing and hedging portfolio risk.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.hedge.calculations import (
    calculate_expected_profit_analytical,
    calculate_expected_profit_simulation
)


def main():
    st.set_page_config(
        page_title="Portfolio Hedging",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("Portfolio Hedging")
    
    # Initialize session state for bets
    if 'bets' not in st.session_state:
        st.session_state.bets = []
    
    # Sidebar
    with st.sidebar:
        st.title("Parameters")
        
        st.markdown("### Market Parameters")
        p = st.slider("Probability of Heads (p)", 0.0, 1.0, 0.5, 0.01, 
                      help="Probability that the coin lands heads")
        
        st.markdown("---")
        developer_mode = st.checkbox("🔧 Developer Mode", value=False,
                                     help="Show performance testing and debugging tools")
        
    st.markdown("""
    ### Toy Market
    
    To investigate optimal hedging, we propose a toy market based off a series of coinflips. 
    Each flip, the coin has a probability $p$ of landing heads, and probability $(1-p)$ of landing tails. 
    We are presented a series of bets on the value of $H_t$, the number of heads in $t$ flips. 
    This streamlit application aims to use this toy market to explore optimal hedging strategies.

    To get started, enter some bets below.
    """)

    st.markdown("### Bets")
    
    # Add new bet section
    with st.expander("➕ Add New Bet", expanded=len(st.session_state.bets) == 0):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            bet_condition = st.text_input(
                "Condition",
                value="H_2 == 1",
                key="new_bet_condition",
                help="Payout condition (e.g., 'H_2 == 1', 'H_5 == 3', 'H_5 < 10')"
            )
        
        with col2:
            bet_odds = st.number_input(
                "Odds",
                min_value=0.01,
                value=2.00,
                step=0.05,
                format="%.2f",
                key="new_bet_odds",
                help="x : 1 odds"
            )
        
        if st.button("Add Bet", type="primary"):
            st.session_state.bets.append({
                'condition': bet_condition,
                'odds': bet_odds
            })
            st.rerun()
    
    # Display current bets
    if st.session_state.bets:
        st.markdown("#### Current Bets")
        
        # Header row
        col1, col2, col3, col4 = st.columns([1, 4, 2, 1])
        with col1:
            st.caption("**#**")
        with col2:
            st.caption("**Condition**")
        with col3:
            st.caption("**Odds**")
        with col4:
            st.caption("**Action**")
        
        # Display each bet
        for i, bet in enumerate(st.session_state.bets):
            col1, col2, col3, col4 = st.columns([1, 4, 2, 1])
            
            with col1:
                st.write(f"{i + 1}")
            
            with col2:
                st.code(bet['condition'])
            
            with col3:
                st.write(f"{bet['odds']:.2f}x")
            
            with col4:
                if st.button("🗑️", key=f"delete_{i}", help="Delete bet"):
                    st.session_state.bets.pop(i)
                    st.rerun()
        
        # Clear all button
        if st.button("Clear All Bets", type="secondary"):
            st.session_state.bets = []
            st.rerun()
    else:
        st.info("No bets entered yet. Click 'Add New Bet' above to get started.")
    
    # Performance Testing Section (only in developer mode)
    if developer_mode and st.session_state.bets:
        st.markdown("---")
        
        with st.expander("🧪 Performance Testing", expanded=False):
            st.markdown("""
            Allocate money to your bets and compare the analytical calculation 
            (exact using binomial distribution) vs Monte Carlo simulation.
            """)
            
            # Allocation inputs
            st.markdown("#### Allocations")
            allocations = []
            
            cols = st.columns(len(st.session_state.bets))
            for i, (col, bet) in enumerate(zip(cols, st.session_state.bets)):
                with col:
                    allocation = st.number_input(
                        f"Bet {i+1}",
                        min_value=0.0,
                        value=100.0,
                        step=10.0,
                        format="%.2f",
                        key=f"allocation_{i}",
                        help=f"${bet['condition']}"
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
            
            if st.button("Calculate Performance", type="primary"):
                # Calculate using both methods
                try:
                    # Analytical calculation
                    analytical_result = calculate_expected_profit_analytical(
                        st.session_state.bets, 
                        allocations, 
                        p
                    )
                    
                    # Monte Carlo simulation
                    simulated_profits = calculate_expected_profit_simulation(
                        st.session_state.bets,
                        allocations,
                        p,
                        n_simulations=n_simulations,
                        seed=seed
                    )
                    
                    # Display results side by side
                    st.markdown("#### 📊 Results Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 🎯 Analytical (Exact)")
                        st.metric(
                            "Expected Profit",
                            f"${analytical_result['total_expected_profit']:.2f}"
                        )
                        st.metric(
                            "Expected Return",
                            f"{analytical_result['expected_return_pct']:.2f}%"
                        )
                        st.metric(
                            "Total Invested",
                            f"${analytical_result['total_invested']:.2f}"
                        )
                    
                    with col2:
                        st.markdown("##### 🎲 Monte Carlo Simulation")
                        mean_profit = np.mean(simulated_profits)
                        std_profit = np.std(simulated_profits)
                        variance_profit = np.var(simulated_profits)
                        stderr_profit = std_profit / np.sqrt(n_simulations)
                        total_invested = sum(allocations)
                        
                        st.metric(
                            "Mean Profit",
                            f"${mean_profit:.2f} ± {1.96 * stderr_profit:.2f}",
                            help="95% confidence interval"
                        )
                        st.metric(
                            "Mean Return",
                            f"{(mean_profit / total_invested * 100):.2f}%"
                        )
                        prob_profit = np.mean(simulated_profits > 0)
                        st.metric(
                            "Probability of Profit",
                            f"{prob_profit * 100:.1f}%"
                        )
                    
                    # Accuracy comparison
                    st.markdown("#### 🎯 Accuracy Check")
                    difference = abs(analytical_result['total_expected_profit'] - mean_profit)
                    relative_error = (difference / abs(analytical_result['total_expected_profit']) * 100) if analytical_result['total_expected_profit'] != 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Absolute Difference", f"${difference:.2f}")
                    with col2:
                        st.metric("Relative Error", f"{relative_error:.3f}%")
                    with col3:
                        if relative_error < 1:
                            st.success("✅ Excellent agreement!")
                        elif relative_error < 5:
                            st.info("✓ Good agreement")
                        else:
                            st.warning("⚠ Consider more simulations")
                    
                    # Per-bet breakdown
                    st.markdown("#### 📋 Per-Bet Breakdown (Analytical)")
                    for i, detail in enumerate(analytical_result['bet_details']):
                        if 'error' not in detail:
                            with st.expander(f"Bet {i+1}: {detail['condition']}"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Win Probability", f"{detail['prob_win']*100:.2f}%")
                                with col2:
                                    st.metric("Allocation", f"${detail['allocation']:.2f}")
                                with col3:
                                    st.metric("Expected Profit", f"${detail['expected_profit']:.2f}")
                                with col4:
                                    st.metric("Expected Return", f"{detail['expected_return_pct']:.2f}%")
                    
                    # Distribution plot
                    st.markdown("#### 📈 Profit Distribution (Simulation)")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    ax.hist(simulated_profits, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                    ax.axvline(mean_profit, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_profit:.2f}')
                    ax.axvline(analytical_result['total_expected_profit'], color='green', linestyle='--', linewidth=2, label=f'Analytical: ${analytical_result["total_expected_profit"]:.2f}')
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


if __name__ == "__main__":
    main()
