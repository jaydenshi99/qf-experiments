"""
Test section for get_optimal_allocation function.
"""

import streamlit as st
import numpy as np
import pandas as pd
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
    
    # Input for target return
    target_return_percent = st.number_input(
        "Target Expected Return (%)",
        min_value=None,
        max_value=None,
        value=10.0,
        step=0.5,
        format="%.2f",
        help="Target expected return as a percentage of capital (e.g. 10 for 10%)."
    )
    
    # Calculate statistics once
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

    # Optimal allocation for the chosen target return
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

    st.markdown("---")
    st.markdown("### Efficient Frontier")
    st.caption(
        "Efficient Frontier - Minimum achievable volatility for each target expected profit level, "
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
    frontier_sharpes = []
    frontier_weights = []

    try:
        for r in returns_range:
            try:
                w = get_optimal_allocation(bets, p, r)
            except ValueError:
                continue  # skip problematic targets

            # Use weights directly (sum to 1) so expected profit is per-dollar return
            exp_prof, vol = calculate_portfolio_stats(w.flatten(), stats_per_dollar)
            ret_pct = exp_prof * 100.0  # convert to %
            frontier_returns.append(ret_pct)
            frontier_vols.append(vol)
            frontier_weights.append(w.flatten())

            # Sharpe ratio with risk‑free rate = 0 (return / volatility, both in %)
            sharpe = ret_pct / vol if vol > 0 else None
            frontier_sharpes.append(sharpe)
    except ValueError as e:
        st.error(str(e).replace("\n", "  \n"))
        return
    except Exception as e:
        st.error(f"Error generating efficient frontier: {str(e)}")
        st.exception(e)
        return

    if frontier_returns:
        paired = sorted(zip(frontier_returns, frontier_vols, frontier_sharpes, frontier_weights), key=lambda x: x[0])
        xs = [p[0] for p in paired]
        ys = [p[1] for p in paired]
        zs = [p[2] for p in paired]
        ws_sorted = [p[3] for p in paired]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Left y-axis: volatility
        vol_line, = ax1.plot(xs, ys, "b-", linewidth=2, label="Volatility")
        ax1.set_xlabel("Expected Return (%)")
        ax1.set_ylabel("Volatility (%)")
        ax1.grid(True, alpha=0.3)

        # Right y-axis: Sharpe ratio (only where defined)
        sharpe_points = [(x, z) for x, z in zip(xs, zs) if z is not None]
        sharpe_line = None
        if sharpe_points:
            sx, sz = zip(*sharpe_points)
            ax2 = ax1.twinx()
            sharpe_line, = ax2.plot(sx, sz, "g--", linewidth=2, label="Sharpe Ratio")
            ax2.set_ylabel("Sharpe Ratio")

        # Combined legend
        lines = [vol_line]
        labels = ["Volatility"]
        if sharpe_line is not None:
            lines.append(sharpe_line)
            labels.append("Sharpe Ratio")
        ax1.legend(lines, labels, loc="upper left")

        ax1.set_title("Efficient Frontier (Volatility & Sharpe)")
        plt.tight_layout()
        st.pyplot(fig)

        # Summary: min volatility point and max Sharpe point
        min_vol_idx = int(np.nanargmin(ys))
        min_vol_val = ys[min_vol_idx]
        min_vol_ret = xs[min_vol_idx]
        min_vol_w = ws_sorted[min_vol_idx]

        sharpe_defined = [(i, s) for i, s in enumerate(zs) if s is not None]
        if sharpe_defined:
            max_sh_idx, max_sh_val = max(sharpe_defined, key=lambda t: t[1])
            max_sh_ret = xs[max_sh_idx]
            max_sh_vol = ys[max_sh_idx]
            max_sh_w = ws_sorted[max_sh_idx]
        else:
            max_sh_val = max_sh_ret = max_sh_vol = max_sh_w = None

        st.markdown("#### Frontier Highlights")
        cols_info = st.columns(2)
        with cols_info[0]:
            st.markdown("**Minimum Volatility Point**")
            st.write(f"Return: {min_vol_ret:.2f}%")
            st.write(f"Volatility: {min_vol_val:.2f}%")
            st.write("Weights (%): " + ", ".join(f"{w*100:.2f}" for w in min_vol_w))
        with cols_info[1]:
            st.markdown("**Maximum Sharpe Point**")
            if max_sh_w is not None:
                st.write(f"Return: {max_sh_ret:.2f}%")
                st.write(f"Volatility: {max_sh_vol:.2f}%")
                st.write(f"Sharpe: {max_sh_val:.3f}")
                st.write("Weights (%): " + ", ".join(f"{w*100:.2f}" for w in max_sh_w))
            else:
                st.write("Sharpe is undefined (volatility is zero for all points).")

        # Detailed frontier table
        table_rows = []
        for ret, vol, sh, w in zip(xs, ys, zs, ws_sorted):
            table_rows.append({
                "Expected Return (%)": f"{ret:.4f}",
                "Volatility (%)": f"{vol:.4f}",
                "Sharpe": f"{sh:.4f}" if sh is not None else "N/A",
                "Weights (%)": ", ".join(f"{wi*100:.2f}" for wi in w)
            })
        df_frontier = pd.DataFrame(table_rows)

        with st.expander("Frontier Points (all target returns)"):
            st.dataframe(df_frontier, use_container_width=True)
    else:
        st.info("Could not generate efficient frontier for the current bets.")
