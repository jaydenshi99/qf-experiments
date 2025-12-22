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
    Find the optimal portfolio allocation from a given target return.
    """)
    
    # Input for target return (percentage of capital)
    target_return_percent = st.number_input(
        "Target Expected Return (%)",
        min_value=None,
        max_value=None,
        value=10.0,
        step=0.5,
        format="%.2f",
        help="Target expected return as a percentage of capital (e.g. 10 for 10%)."
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

    # Convert target return from % to per-dollar for optimisation formulas
    target_return = target_return_percent / 100.0

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
            st.write("**Expected Returns:**")
            for i, exp_prof in enumerate(stats_per_dollar['expected_profit_per_dollar']):
                ret_pct = exp_prof * 100.0
                st.write(f"  Bet {i+1}: {ret_pct:.4f}%")

            st.write("\n**Covariance Matrix:**")
            st.write(stats_per_dollar['covariance_matrix'])

        st.markdown("**Optimal Allocation (percent of capital):**")
        weights = optimal_alloc.flatten()

        cols = st.columns(len(bets))
        for i, (col, w) in enumerate(zip(cols, weights)):
            with col:
                st.metric(
                    f"Bet {i+1}",
                    f"{w*100:.2f}%",
                    None
                )

        total_weight = np.sum(weights)
        st.write(f"**Total Allocation:** {total_weight*100:.2f}%")

        # Expected return and volatility in percentage terms
        expected_return_per_dollar, portfolio_vol = calculate_portfolio_stats(weights, stats_per_dollar)
        expected_return_pct = expected_return_per_dollar * 100.0
        st.write(f"**Expected Return:** {expected_return_pct:.2f}%")
        st.write(f"**Target Return:** {target_return_percent:.2f}%")
        st.write(f"**Difference:** {abs(expected_return_pct - target_return_percent):.2f}%")

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

            # Use weights directly (sum to 1) so expected profit is per-dollar return
            exp_prof, vol = calculate_portfolio_stats(w.flatten(), stats_per_dollar)
            frontier_returns.append(exp_prof * 100.0)  # convert to %
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
        ax.set_xlabel("Expected Return (%)")
        ax.set_ylabel("Volatility (%)")
        ax.set_title("Efficient Frontier")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Could not generate efficient frontier for the current bets.")
