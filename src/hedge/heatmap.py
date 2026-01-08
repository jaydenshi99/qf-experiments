"""
Heatmap visualization for allocation analysis.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.hedge.calculations import (
    get_bet_statistics_per_dollar,
    calculate_portfolio_stats,
    get_max_time_from_bets
)


def generate_heatmap(bets, p, step_size=0.02, n_sims=3000):
    """Generate allocation heatmaps."""
    # Create allocation grid (0% to 100%)
    allocations_bet1 = np.arange(0, 1 + step_size, step_size)
    allocations_bet2 = np.arange(0, 1 + step_size, step_size)
    
    # Initialize results grids
    expected_returns = np.zeros((len(allocations_bet2), len(allocations_bet1)))
    std_devs = np.zeros((len(allocations_bet2), len(allocations_bet1)))
    
    # Calculate statistics per dollar (only need to do this once!)
    with st.spinner('Calculating bet statistics...'):
        stats_per_dollar = get_bet_statistics_per_dollar(bets, p, n_simulations=n_sims)
    
    # Calculate expected return and std dev for each allocation (now just scaling!)
    with st.spinner('Generating heatmaps...'):
        for i, alloc1 in enumerate(allocations_bet1):
            for j, alloc2 in enumerate(allocations_bet2):
                if alloc1 + alloc2 <= 1.0:  # Valid allocation
                    allocations = [alloc1, alloc2]
                    
                    # Calculate expected profit and volatility using scaling
                    if alloc1 + alloc2 > 0:
                        expected_profit, volatility = calculate_portfolio_stats(allocations, stats_per_dollar)
                        expected_returns[j, i] = expected_profit
                        std_devs[j, i] = volatility
                    else:
                        expected_returns[j, i] = 0.0
                        std_devs[j, i] = 0.0
                else:
                    expected_returns[j, i] = np.nan  # Invalid allocation
                    std_devs[j, i] = np.nan
    
    # Check if Monte Carlo is being used
    max_time = get_max_time_from_bets(bets)
    analytical_threshold = 18  # Match the default in get_profit_distribution
    using_mc = max_time > analytical_threshold
    
    # Create side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # LEFT HEATMAP: Expected Profit
    masked_returns = np.ma.masked_invalid(expected_returns)
    max_abs = max(abs(np.nanmin(expected_returns)), abs(np.nanmax(expected_returns)))
    
    im1 = ax1.imshow(masked_returns, origin='lower', cmap='RdYlGn', 
                    extent=[0, 100, 0, 100], aspect='auto',
                    vmin=-max_abs, vmax=max_abs)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Expected Profit ($)', rotation=270, labelpad=20, fontsize=11)
    
    constraint_x = np.linspace(0, 100, 100)
    constraint_y = 100 - constraint_x
    ax1.plot(constraint_x, constraint_y, 'b--', linewidth=2, alpha=0.7)
    
    valid_mask = ~np.isnan(expected_returns)
    optimal_info = None
    if valid_mask.any():
        max_idx = np.nanargmax(expected_returns)
        max_i, max_j = np.unravel_index(max_idx, expected_returns.shape)
        optimal_alloc1_pct = allocations_bet1[max_j] * 100
        optimal_alloc2_pct = allocations_bet2[max_i] * 100
        optimal_profit = expected_returns[max_i, max_j]
        
        ax1.plot(optimal_alloc1_pct, optimal_alloc2_pct, 'r*', markersize=15)
        optimal_info = {
            'alloc1_pct': optimal_alloc1_pct,
            'alloc2_pct': optimal_alloc2_pct,
            'profit': optimal_profit
        }
    
    # Get bet labels for axes
    bet1_label = bets[0].get('payoff') or bets[0].get('condition', 'Bet 1')
    bet2_label = bets[1].get('payoff') or bets[1].get('condition', 'Bet 2')
    
    ax1.set_xlabel(f'Bet 1: {bet1_label} (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'Bet 2: {bet2_label} (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Expected Profit ($)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    
    # Add note if using Monte Carlo
    if using_mc:
        ax1.text(0.02, 0.98, 'Note: Monte Carlo simulation used\ndue to large number of outcomes', 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # RIGHT HEATMAP: Standard Deviation
    masked_stddev = np.ma.masked_invalid(std_devs)
    
    # Use reversed colormap: green (low std dev) to red (high std dev)
    im2 = ax2.imshow(masked_stddev, origin='lower', cmap='RdYlGn_r', 
                    extent=[0, 100, 0, 100], aspect='auto',
                    vmin=np.nanmin(std_devs), vmax=np.nanmax(std_devs))
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Volatility (Std Dev of Returns, %)', rotation=270, labelpad=20, fontsize=11)
    
    ax2.plot(constraint_x, constraint_y, 'b--', linewidth=2, alpha=0.7)
    
    # Find and plot line through minimum risk points
    min_volatility = None
    hedge_slope = None
    if valid_mask.any():
        # Create mask for non-zero allocations
        total_allocations = np.zeros_like(std_devs)
        for i in range(len(allocations_bet1)):
            for j in range(len(allocations_bet2)):
                total_allocations[j, i] = allocations_bet1[i] + allocations_bet2[j]
        
        # std_devs now contains volatility (std dev of returns) as percentages
        # For minimum risk frontier, we want minimum volatility for each capital level
        min_std_points = []
        
        # Consider total allocations from 0 to 100% in steps
        for total_pct in np.arange(0, 101, 2):  # Every 2%
            total_frac = total_pct / 100.0
            # Find all allocations that sum to approximately this total
            tolerance = 0.015  # Within 1.5%
            matching_mask = np.abs(total_allocations - total_frac) < tolerance
            matching_mask = matching_mask & ~np.isnan(std_devs) & (total_allocations > 0)
            
            if np.any(matching_mask):
                # Find minimum volatility among these allocations
                min_risk_idx = np.nanargmin(np.where(matching_mask, std_devs, np.inf))
                min_risk_i, min_risk_j = np.unravel_index(min_risk_idx, std_devs.shape)
                
                alloc1_pct = allocations_bet1[min_risk_j] * 100
                alloc2_pct = allocations_bet2[min_risk_i] * 100
                min_std_points.append((alloc1_pct, alloc2_pct))
        
        if np.any(min_std_points):
            # Plot minimum std dev frontier points
            if len(min_std_points) > 0:
                ms_array = np.array(min_std_points)
                ax2.scatter(ms_array[:, 0], ms_array[:, 1], 
                           c='cyan', s=15, alpha=0.8, edgecolors='blue', linewidth=0.5,
                           label='Minimum Risk Frontier', zorder=5)
            
            # Fit a line through these points
            if len(min_std_points) > 1:
                try:
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
                except Exception:
                    # If regression fails, just don't plot the line
                    hedge_slope = None
            elif len(min_std_points) == 1:
                # Single point
                ax2.plot(min_std_points[0][0], min_std_points[0][1], 'b*', markersize=15)
        
        # Find minimum volatility across all non-zero allocations
        non_zero_mask = total_allocations > 0
        min_volatility = np.nanmin(std_devs[non_zero_mask])
    
    # Use the same bet labels as before
    ax2.set_xlabel(f'Bet 1: {bet1_label} (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'Bet 2: {bet2_label} (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Volatility (Std Dev of Returns)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    
    # Add note if using Monte Carlo
    if using_mc:
        ax2.text(0.02, 0.98, 'Note: Monte Carlo simulation used\ndue to large number of outcomes', 
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, optimal_info, min_volatility, hedge_slope


def render_heatmap_section(bets, p):
    """Render heatmap section."""
    if bets:
        st.markdown("---")
        st.markdown("### Allocation Heatmaps")
    
    if len(bets) == 2:
        st.markdown("""
        Explore the expected return and variance for different allocations of capital between your two bets.
        Calculated with total capital = $1.
        """)
        
        # Auto-generate heatmap
        try:
            fig, optimal_info, min_volatility, hedge_slope = generate_heatmap(bets, p)
            st.pyplot(fig)
            
            # Display optimal allocations
            if optimal_info:
                st.markdown("#### Optimal Allocations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Best Expected Profit:**")
                    subcol1, subcol2, subcol3 = st.columns(3)
                    with subcol1:
                        st.metric("Bet 1", f"{optimal_info['alloc1_pct']:.1f}%")
                    with subcol2:
                        st.metric("Bet 2", f"{optimal_info['alloc2_pct']:.1f}%")
                    with subcol3:
                        st.metric("Expected Profit", f"${optimal_info['profit']:.2f}")
                
                with col2:
                    st.markdown("**Minimum Risk Hedge:**")
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        if hedge_slope is not None:
                            st.metric("Hedge Ratio", f"y = {hedge_slope:.4f}x")
                        else:
                            st.info("Not enough points to fit line")
                    with subcol2:
                        if min_volatility is not None:
                            st.metric("Min Volatility", f"{min_volatility:.2f}%",
                                     help="Minimum volatility (standard deviation of returns) across all allocations")
        
        except Exception as e:
            st.error(f"Error generating heatmap: {str(e)}")
            st.exception(e)
    
    elif bets and len(bets) != 2:
        st.info(f"Allocation heatmap is only available for exactly 2 bets. You currently have {len(bets)} bet(s).")

