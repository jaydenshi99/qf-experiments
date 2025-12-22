"""
Test section for get_optimal_allocation function.
"""

import streamlit as st
import numpy as np
from src.hedge.calculations import (
    get_optimal_allocation,
    get_bet_statistics_per_dollar,
    calculate_portfolio_stats
)


def render_optimal_allocation_test(bets, p):
    """Render the optimal allocation test section."""
    if not bets:
        return
    
    st.markdown("---")
    st.markdown("### Optimal Allocation Test")
    
    if len(bets) < 2:
        st.info("Optimal allocation requires at least 2 bets.")
        return
    
    st.markdown("""
    Test the `get_optimal_allocation` function to find the optimal portfolio allocation
    for a given target expected return per dollar.
    """)
    
    # Input for target return
    col1, col2 = st.columns(2)
    
    with col1:
        target_return = st.number_input(
            "Target Expected Return per Dollar",
            min_value=None,
            max_value=None,
            value=0.1,
            step=0.01,
            format="%.4f",
            help="Target expected return per dollar invested"
        )
    
    with col2:
        total_capital = st.number_input(
            "Total Capital ($)",
            min_value=0.01,
            value=1.0,
            step=0.1,
            help="Total amount to allocate"
        )
    
    if st.button("Calculate Optimal Allocation", type="primary"):
        try:
            # Get statistics first
            with st.spinner('Calculating bet statistics...'):
                stats_per_dollar = get_bet_statistics_per_dollar(bets, p)
            
            # Get optimal allocation
            with st.spinner('Calculating optimal allocation...'):
                optimal_alloc = get_optimal_allocation(bets, p, target_return)
            
            # Display results
            st.markdown("#### Results")
            
            # Show bet statistics
            with st.expander("Bet Statistics (per dollar)"):
                st.write("**Expected Profits:**")
                for i, exp_prof in enumerate(stats_per_dollar['expected_profit_per_dollar']):
                    st.write(f"  Bet {i+1}: ${exp_prof:.6f}")
                
                st.write("\n**Covariance Matrix:**")
                st.write(stats_per_dollar['covariance_matrix'])
            
            # Show optimal allocation
            st.markdown("**Optimal Allocation:**")
            alloc_scaled = (optimal_alloc.flatten() * total_capital)
            
            cols = st.columns(len(bets))
            for i, (col, alloc) in enumerate(zip(cols, alloc_scaled)):
                with col:
                    st.metric(
                        f"Bet {i+1}",
                        f"${alloc:.2f}",
                        f"{alloc/total_capital*100:.2f}%"
                    )
            
            # Verify sum
            total_alloc = np.sum(alloc_scaled)
            st.write(f"**Total Allocated:** ${total_alloc:.2f}")
            
            # Verify expected return
            expected_return = np.dot(optimal_alloc.flatten(), stats_per_dollar['expected_profit_per_dollar'])
            st.write(f"**Expected Return per Dollar:** {expected_return:.10f}")
            st.write(f"**Target Return per Dollar:** {target_return:.10f}")
            st.write(f"**Difference:** {abs(expected_return - target_return):.10f}")
            
            # Calculate portfolio stats
            portfolio_exp_prof, portfolio_vol = calculate_portfolio_stats(alloc_scaled, stats_per_dollar)
            st.write(f"**Portfolio Expected Profit:** ${portfolio_exp_prof:.6f}")
            st.write(f"**Portfolio Volatility:** {portfolio_vol:.4f}%")
            
        except Exception as e:
            st.error(f"Error calculating optimal allocation: {str(e)}")
            st.exception(e)

