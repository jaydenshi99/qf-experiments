"""
Test section for get_optimal_allocation function.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    st.markdown("### Optimal Allocation")
    
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
    
    # ------------------------------------------------------------
    # Calculate statistics once (no caching to keep logic simple)
    # ------------------------------------------------------------
    try:
        with st.spinner('Calculating bet statistics...'):
            stats_per_dollar = get_bet_statistics_per_dollar(bets, p)
    except ValueError as e:
        st.error(str(e).replace("\n", "  \n"))
        return
    except Exception as e:
        st.error(f"Error calculating bet statistics: {str(e)}")
        st.exception(e)
        return

    # ------------------------------------------------------------
    # Optimal allocation for the chosen target return
    # ------------------------------------------------------------
    optimal_alloc = None
    try:
        with st.spinner('Calculating optimal allocation...'):
            optimal_alloc = get_optimal_allocation(bets, p, target_return)
    except ValueError as e:
        st.error(str(e).replace("\n", "  \n"))
    except Exception as e:
        st.error(f"Error calculating optimal allocation: {str(e)}")
        st.exception(e)

    # Display results (only if optimal allocation succeeded)
    if optimal_alloc is not None:
        st.markdown("#### Optimal Allocation Results")

        with st.expander("Bet Statistics (per dollar)"):
            st.write("**Expected Profits:**")
            for i, exp_prof in enumerate(stats_per_dollar['expected_profit_per_dollar']):
                st.write(f"  Bet {i+1}: ${exp_prof:.6f}")

            st.write("\n**Covariance Matrix:**")
            st.write(stats_per_dollar['covariance_matrix'])

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

        total_alloc = np.sum(alloc_scaled)
        st.write(f"**Total Allocated:** ${total_alloc:.2f}")

        expected_return = np.dot(optimal_alloc.flatten(), stats_per_dollar['expected_profit_per_dollar'])
        st.write(f"**Expected Return per Dollar:** {expected_return:.10f}")
        st.write(f"**Target Return per Dollar:** {target_return:.10f}")
        st.write(f"**Difference:** {abs(expected_return - target_return):.10f}")

        portfolio_exp_prof, portfolio_vol = calculate_portfolio_stats(alloc_scaled, stats_per_dollar)
        st.write(f"**Portfolio Expected Profit:** ${portfolio_exp_prof:.6f}")
        st.write(f"**Portfolio Volatility:** {portfolio_vol:.4f}%")

    # ------------------------------------------------------------
    # Efficient frontier (always rendered as its own section)
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Efficient Frontier")
    st.caption(
        "Minimum achievable volatility for each expected profit level, "
        "based on the current bets and probability."
    )

    mu = np.array(stats_per_dollar['expected_profit_per_dollar'])
    min_ret = float(mu.min())
    max_ret = float(mu.max())

    if abs(max_ret - min_ret) < 1e-10:
        st.info("Efficient frontier is not defined when all bets have the same expected return.")
        return

    returns_range = np.linspace(min_ret, max_ret, 50)
    frontier_returns = []
    frontier_vols = []

    try:
        for r in returns_range:
            try:
                w = get_optimal_allocation(bets, p, r)
            except ValueError:
                continue  # skip problematic targets

            allocs = (w.flatten() * total_capital)
            exp_prof, vol = calculate_portfolio_stats(allocs, stats_per_dollar)
            frontier_returns.append(exp_prof)
            frontier_vols.append(vol)
    except ValueError as e:
        st.error(str(e).replace("\n", "  \n"))
        return
    except Exception as e:
        st.error(f"Error generating efficient frontier: {str(e)}")
        st.exception(e)
        return

    if frontier_returns:
        paired = sorted(zip(frontier_returns, frontier_vols), key=lambda x: x[0])
        xs = [p[0] for p in paired]
        ys = [p[1] for p in paired]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, ys, "b-", linewidth=2)
        ax.set_xlabel("Expected Profit ($)")
        ax.set_ylabel("Volatility (%)")
        ax.set_title("Efficient Frontier")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Could not generate efficient frontier for the current bets.")
