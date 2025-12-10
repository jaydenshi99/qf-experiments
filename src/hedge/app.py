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
        developer_mode = st.checkbox("Developer Mode", value=False,
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
    
    # Allocation Heatmap Section, only works for 2 bets
    if st.session_state.bets:
        st.markdown("---")
        st.markdown("### Allocation Heatmaps")
    
    if len(st.session_state.bets) == 2:
        st.markdown("""
        Explore the expected return and variance for different allocations of capital between your two bets.
        Calculated with total capital = $1.
        """)
        
        if st.button("Generate Heatmap", type="primary"):
            try:
                step_size = 0.02
                
                # Create allocation grid (0% to 100%)
                allocations_bet1 = np.arange(0, 1 + step_size, step_size)
                allocations_bet2 = np.arange(0, 1 + step_size, step_size)
                
                # Initialize results grids
                expected_returns = np.zeros((len(allocations_bet2), len(allocations_bet1)))
                std_devs = np.zeros((len(allocations_bet2), len(allocations_bet1)))
                
                # Number of Monte Carlo simulations for std dev
                n_sims = 2000
                
                # Calculate expected return and std dev for each allocation
                with st.spinner('Calculating heatmaps...'):
                    for i, alloc1 in enumerate(allocations_bet1):
                        for j, alloc2 in enumerate(allocations_bet2):
                            if alloc1 + alloc2 <= 1.0:  # Valid allocation
                                allocations = [alloc1, alloc2]
                                
                                # Analytical expected return
                                result = calculate_expected_profit_analytical(
                                    st.session_state.bets,
                                    allocations,
                                    p
                                )
                                expected_returns[j, i] = result['total_expected_profit'] * 100
                                
                                # Monte Carlo standard deviation
                                profits = calculate_expected_profit_simulation(
                                    st.session_state.bets,
                                    allocations,
                                    p,
                                    n_simulations=n_sims,
                                    seed=42
                                )
                                std_devs[j, i] = np.std(profits)
                            else:
                                expected_returns[j, i] = np.nan  # Invalid allocation
                                std_devs[j, i] = np.nan
                
                # Create side-by-side heatmaps
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                # LEFT HEATMAP: Expected Return
                masked_returns = np.ma.masked_invalid(expected_returns)
                max_abs = max(abs(np.nanmin(expected_returns)), abs(np.nanmax(expected_returns)))
                
                im1 = ax1.imshow(masked_returns, origin='lower', cmap='RdYlGn', 
                                extent=[0, 100, 0, 100], aspect='auto',
                                vmin=-max_abs, vmax=max_abs)
                
                cbar1 = plt.colorbar(im1, ax=ax1)
                cbar1.set_label('Expected Return (%)', rotation=270, labelpad=20, fontsize=11)
                
                constraint_x = np.linspace(0, 100, 100)
                constraint_y = 100 - constraint_x
                ax1.plot(constraint_x, constraint_y, 'b--', linewidth=2, alpha=0.7)
                
                valid_mask = ~np.isnan(expected_returns)
                if valid_mask.any():
                    max_idx = np.nanargmax(expected_returns)
                    max_i, max_j = np.unravel_index(max_idx, expected_returns.shape)
                    optimal_alloc1_pct = allocations_bet1[max_j] * 100
                    optimal_alloc2_pct = allocations_bet2[max_i] * 100
                    optimal_return = expected_returns[max_i, max_j]
                    
                    ax1.plot(optimal_alloc1_pct, optimal_alloc2_pct, 'r*', markersize=15)
                
                ax1.set_xlabel(f'Bet 1: {st.session_state.bets[0]["condition"]} (%)', fontsize=11, fontweight='bold')
                ax1.set_ylabel(f'Bet 2: {st.session_state.bets[1]["condition"]} (%)', fontsize=11, fontweight='bold')
                ax1.set_title('Expected Return (%)', fontsize=13, fontweight='bold')
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax1.set_xlim(0, 100)
                ax1.set_ylim(0, 100)
                
                # RIGHT HEATMAP: Standard Deviation
                masked_stddev = np.ma.masked_invalid(std_devs)
                
                # Use reversed colormap: green (low std dev) to red (high std dev)
                im2 = ax2.imshow(masked_stddev, origin='lower', cmap='RdYlGn_r', 
                                extent=[0, 100, 0, 100], aspect='auto',
                                vmin=np.nanmin(std_devs), vmax=np.nanmax(std_devs))
                
                cbar2 = plt.colorbar(im2, ax=ax2)
                cbar2.set_label('Standard Deviation ($)', rotation=270, labelpad=20, fontsize=11)
                
                ax2.plot(constraint_x, constraint_y, 'b--', linewidth=2, alpha=0.7)
                
                # Find and plot line through minimum risk points
                if valid_mask.any():
                    # Create mask for non-zero allocations
                    total_allocations = np.zeros_like(std_devs)
                    for i in range(len(allocations_bet1)):
                        for j in range(len(allocations_bet2)):
                            total_allocations[j, i] = allocations_bet1[i] + allocations_bet2[j]
                    
                    # For each total allocation level, find the minimum standard deviation
                    # Group by total allocation and find minimum std dev allocation
                    min_std_points = []
                    
                    # Consider total allocations from 0 to 100% in steps
                    for total_pct in np.arange(0, 101, 2):  # Every 2%
                        total_frac = total_pct / 100.0
                        # Find all allocations that sum to approximately this total
                        tolerance = 0.015  # Within 1.5%
                        matching_mask = np.abs(total_allocations - total_frac) < tolerance
                        matching_mask = matching_mask & ~np.isnan(std_devs)
                        
                        if np.any(matching_mask):
                            # Find minimum std dev among these allocations
                            min_std_idx = np.nanargmin(np.where(matching_mask, std_devs, np.inf))
                            min_std_i, min_std_j = np.unravel_index(min_std_idx, std_devs.shape)
                            
                            alloc1_pct = allocations_bet1[min_std_j] * 100
                            alloc2_pct = allocations_bet2[min_std_i] * 100
                            min_std_points.append((alloc1_pct, alloc2_pct))
                    
                    if np.any(min_std_points):
                        # Plot minimum std dev frontier points
                        if len(min_std_points) > 0:
                            ms_array = np.array(min_std_points)
                            ax2.scatter(ms_array[:, 0], ms_array[:, 1], 
                                       c='cyan', s=15, alpha=0.8, edgecolors='blue', linewidth=0.5,
                                       label='Minimum Risk Frontier', zorder=5)
                        
                        hedge_slope = None
                        
                        if len(min_std_points) > 1:
                            # Fit a line through these points
                            points_array = np.array(min_std_points)
                            x_coords = points_array[:, 0]
                            y_coords = points_array[:, 1]
                            
                            # Linear regression
                            coeffs = np.polyfit(x_coords, y_coords, 1)
                            hedge_slope = coeffs[0]
                            
                            # Plot the line
                            x_line = np.linspace(x_coords.min(), x_coords.max(), 100)
                            y_line = coeffs[0] * x_line + coeffs[1]
                            
                            # Clip to valid range: y >= 0, y <= 100, AND x + y <= 100
                            valid_line_mask = (y_line >= 0) & (y_line <= 100) & (x_line + y_line <= 100)
                            
                            if np.any(valid_line_mask):
                                ax2.plot(x_line[valid_line_mask], y_line[valid_line_mask], 
                                        'b-', linewidth=3, alpha=0.8, label='Minimum Risk Line')
                        elif len(min_std_points) == 1:
                            # Single point
                            ax2.plot(min_std_points[0][0], min_std_points[0][1], 'b*', markersize=15)
                    
                    # Find minimum std dev per dollar
                    # Std dev normalized by total capital allocated
                    std_per_dollar = std_devs / (total_allocations + 1e-10)  # Add small epsilon to avoid division by zero
                    
                    # Exclude zero allocations for the metric
                    non_zero_mask = total_allocations > 0
                    min_std_per_dollar = np.nanmin(std_per_dollar[non_zero_mask])
                
                ax2.set_xlabel(f'Bet 1: {st.session_state.bets[0]["condition"]} (%)', fontsize=11, fontweight='bold')
                ax2.set_ylabel(f'Bet 2: {st.session_state.bets[1]["condition"]} (%)', fontsize=11, fontweight='bold')
                ax2.set_title(f'Standard Deviation (Risk) - {n_sims:,} simulations', fontsize=13, fontweight='bold')
                ax2.legend(loc='upper right', fontsize=9)
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.set_xlim(0, 100)
                ax2.set_ylim(0, 100)
                
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Display optimal allocations
                if valid_mask.any():
                    st.markdown("#### Optimal Allocations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Best Expected Return:**")
                        subcol1, subcol2, subcol3 = st.columns(3)
                        with subcol1:
                            st.metric("Bet 1", f"{optimal_alloc1_pct:.1f}%")
                        with subcol2:
                            st.metric("Bet 2", f"{optimal_alloc2_pct:.1f}%")
                        with subcol3:
                            st.metric("Return", f"{optimal_return:.2f}%")
                    
                    with col2:
                        st.markdown("**Minimum Risk Hedge:**")
                        subcol1, subcol2 = st.columns(2)
                        with subcol1:
                            if hedge_slope is not None:
                                st.metric("Hedge Ratio", f"y = {hedge_slope:.4f}x")
                            else:
                                st.info("Not enough points to fit line")
                        with subcol2:
                            st.metric("Min Risk/$", f"${min_std_per_dollar:.4f}",
                                     help="Minimum standard deviation per dollar invested - risk per unit capital")
                
            except Exception as e:
                st.error(f"Error generating heatmap: {str(e)}")
                st.exception(e)
    
    elif st.session_state.bets and len(st.session_state.bets) != 2:
        st.info(f"Allocation heatmap is only available for exactly 2 bets. You currently have {len(st.session_state.bets)} bet(s).")
    
    # Performance Testing Section (only in developer mode)
    if developer_mode and st.session_state.bets:
        st.markdown("---")
        
        with st.expander("Performance Testing", expanded=False):
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
