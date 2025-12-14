"""
Monte Carlo Simulation for Options Playground

This module implements Monte Carlo simulation using geometric Brownian motion
to model stock price paths and analyze option pricing.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def geometric_brownian_motion(S0, mu, sigma, T, n_steps, n_paths):
    """
    Generate stock price paths using geometric Brownian motion.
    
    Parameters:
    - S0: Initial stock price
    - mu: Drift (expected return)
    - sigma: Volatility
    - T: Time to maturity
    - n_steps: Number of time steps
    - n_paths: Number of simulation paths
    
    Returns:
    - times: Array of time points
    - paths: Array of stock price paths (n_paths x n_steps+1)
    """
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    
    # Generate random normal increments
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    
    # Initialize paths array
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate paths using GBM formula: dS = mu*S*dt + sigma*S*dW
    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, i])
    
    return times, paths


def calculate_option_payoffs(paths, K, option_type='call'):
    """Calculate option payoffs at expiration for all paths."""
    final_prices = paths[:, -1]
    
    if option_type == 'call':
        payoffs = np.maximum(final_prices - K, 0)
    else:  # put
        payoffs = np.maximum(K - final_prices, 0)
    
    return payoffs


def pnl_analysis_tab(model_params):
    """Monte Carlo Simulation tab content"""
    
    # Portfolio Construction Section
    st.markdown("## Portfolio Construction")
    
    # Initialize portfolio in session state if not exists
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    # Add new position in a clean layout
    with st.expander("Add Position", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            option_type = st.selectbox("Option Type", ["Call", "Put"], key="main_option_type")
        
        with col2:
            strike = st.number_input("Strike Price", min_value=0.01, value=105.00, step=1.00, format="%.2f", key="main_strike")
        
        with col3:
            position = st.number_input("Position", min_value=-10, max_value=10, value=1, step=1, key="main_position", help="Positive = Long, Negative = Short")
        
        with col4:
            price = st.number_input("Option Price", min_value=0.01, value=2.50, step=0.01, format="%.2f", key="main_price", help="Premium paid/received per option")
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("Add to Portfolio", type="primary"):
                new_leg = {
                    'type': option_type.lower(),
                    'strike': strike,
                    'position': position,
                    'price': price,
                    'multiplier': 1 if position > 0 else -1
                }
                st.session_state.portfolio.append(new_leg)
                st.rerun()
    
    # Display current portfolio
    if st.session_state.portfolio:
        # Header row
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1.5, 1.5, 1, 1.5, 1, 1])
        with col2:
            st.caption("**Position**")
        with col3:
            st.caption("**Type**")
        with col4:
            st.caption("**Strike**")
        with col5:
            st.caption("**Price**")
        with col6:
            st.caption("**Total**")
        
        # Interactive table with controls for each row
        for i, leg in enumerate(st.session_state.portfolio):
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1.5, 1.5, 1, 1.5, 1, 1])
            
            with col1:
                if st.button("-", key=f"decrease_{i}", help="Decrease position"):
                    if st.session_state.portfolio[i]['position'] > -10:  # Limit to -10
                        st.session_state.portfolio[i]['position'] -= 1
                        st.session_state.portfolio[i]['multiplier'] = 1 if st.session_state.portfolio[i]['position'] > 0 else -1
                    st.rerun()
            
            with col2:
                st.write(f"**{st.session_state.portfolio[i]['position']:+d}**")
            
            with col3:
                st.write(f"{leg['type'].title()}")
            
            with col4:
                st.write(f"${leg['strike']:.2f}")
            
            with col5:
                st.write(f"${leg['price']:.2f}")
            
                with col6:
                    # Calculate total cost/credit for this position
                    total_cost = leg['position'] * leg['price'] * -1  # Negative because we pay for long positions
                    color = "red" if total_cost < 0 else "green" if total_cost > 0 else "gray"
                    st.markdown(f"<span style='color: {color}'>${total_cost:.2f}</span>", unsafe_allow_html=True)
            
            with col7:
                if st.button("+", key=f"increase_{i}", help="Increase position"):
                    if st.session_state.portfolio[i]['position'] < 10:  # Limit to +10
                        st.session_state.portfolio[i]['position'] += 1
                        st.session_state.portfolio[i]['multiplier'] = 1 if st.session_state.portfolio[i]['position'] > 0 else -1
                    st.rerun()
            
            with col8:
                if st.button("Delete", key=f"delete_{i}", help="Delete position"):
                    st.session_state.portfolio.pop(i)
                    st.rerun()
        
        # Calculate and display cumulative total
        cumulative_total = sum(leg['position'] * leg['price'] * -1 for leg in st.session_state.portfolio)
        
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1.5, 1.5, 1, 1.5, 1, 1])
        with col6:
            st.markdown(f"**Net Total: ${cumulative_total:.2f}**")

        # Clear all button
        if st.button("Clear All", type="secondary"):
            st.session_state.portfolio = []
            st.rerun()
        
        st.markdown("---")
    else:
        st.write("*No positions in portfolio*")
        st.markdown("---")
    
    # Monte Carlo Analysis Section
    st.markdown("## Monte Carlo Analysis")
    
    # Extract parameters
    S0 = model_params['S0']
    K = model_params['K']
    T = model_params['T']
    r = model_params['r']
    sigma = model_params['sigma']
    n_paths = model_params['n_paths']
    n_steps = model_params['n_time_steps']
    mu = model_params['mu']
    
    # Generate Monte Carlo paths
    np.random.seed(42)  # For reproducible results
    times, paths = geometric_brownian_motion(S0, mu, sigma, T, n_steps, n_paths)
    
    # Calculate statistics for display on graph
    avg_final = np.mean(paths[:, -1])
    std_final = np.std(paths[:, -1])
    
    # Visualization
    st.markdown("### Stock Price Paths")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot a subset of paths for clarity
    n_display = min(100, n_paths)
    display_indices = np.random.choice(n_paths, n_display, replace=False)
    
    # Get final prices for color mapping
    final_prices_display = paths[display_indices, -1]
    
    # Create colormap based on final prices (red to green spectrum)
    # Normalize final prices to [0, 1] for colormap
    price_min, price_max = np.min(final_prices_display), np.max(final_prices_display)
    if price_max > price_min:
        normalized_prices = (final_prices_display - price_min) / (price_max - price_min)
    else:
        normalized_prices = np.ones(len(final_prices_display)) * 0.5
    
    # Use RdYlGn colormap (red-yellow-green) for price-based coloring
    colors = plt.cm.RdYlGn(normalized_prices)
    
    for i, idx in enumerate(display_indices):
        alpha = 0.4 if n_display > 50 else 0.7
        ax.plot(times, paths[idx], color=colors[i], alpha=alpha, linewidth=0.8)
    
    # Plot mean path with same color as initial price
    mean_path = np.mean(paths, axis=0)
    ax.plot(times, mean_path, color='#2E86AB', linewidth=1.2, label='Mean Path', alpha=0.9, zorder=10)
    
    # Add initial price line
    ax.axhline(y=S0, color='#2E86AB', linestyle=':', linewidth=2, label=f'Initial Price (${S0})', alpha=0.8)
    
    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Stock Price ($)')
    ax.set_title(f'Monte Carlo Simulation - Stock Price Paths (Colored by Final Price)')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend with statistics included
    legend_elements = ax.get_legend_handles_labels()
    legend_text = []
    for handle, label in zip(legend_elements[0], legend_elements[1]):
        legend_text.append(label)
    
    # Add statistics to the legend
    legend_text.append(f'Mean Final: ${avg_final:.2f}')
    legend_text.append(f'Std Dev: ${std_final:.2f}')
    
    # Create legend with all elements
    ax.legend(legend_elements[0] + [plt.Line2D([0], [0], color='none')] * 2, 
              legend_text, loc='upper left', framealpha=0.9)
    
    # Add colorbar to show price mapping
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=price_min, vmax=price_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Final Stock Price ($)', rotation=270, labelpad=20)
    
    # Display charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        # Payoff Diagram
        if st.session_state.get('portfolio', []):
            # Create range of stock prices around current price
            price_range = np.linspace(S0 * 0.5, S0 * 1.5, 200)
            portfolio_payoff = np.zeros_like(price_range)
            
            # Calculate payoff for each position in portfolio
            for leg in st.session_state.portfolio:
                position = leg['position']
                option_type = leg['type']
                strike = leg['strike']
                premium = leg['price']
                
                if option_type == 'call':
                    # Call payoff: max(S - K, 0) for long, -max(S - K, 0) for short
                    option_payoff = np.maximum(price_range - strike, 0)
                else:  # put
                    # Put payoff: max(K - S, 0) for long, -max(K - S, 0) for short
                    option_payoff = np.maximum(strike - price_range, 0)
                
                # Account for position direction (long/short) and premium paid/received
                if position > 0:  # Long position
                    leg_payoff = position * (option_payoff - premium)
                else:  # Short position
                    leg_payoff = position * (option_payoff - premium)
                
                portfolio_payoff += leg_payoff
            
            # Create payoff diagram
            fig_payoff, ax_payoff = plt.subplots(figsize=(10, 6))
            
            # Plot payoff line
            ax_payoff.plot(price_range, portfolio_payoff, linewidth=2, color='blue', label='Portfolio P&L')
            ax_payoff.fill_between(price_range, 0, portfolio_payoff, 
                                  where=(portfolio_payoff >= 0), color='green', alpha=0.3, label='Profit')
            ax_payoff.fill_between(price_range, 0, portfolio_payoff, 
                                  where=(portfolio_payoff < 0), color='red', alpha=0.3, label='Loss')
            
            # Add breakeven points
            breakeven_indices = np.where(np.diff(np.sign(portfolio_payoff)))[0]
            for idx in breakeven_indices:
                breakeven_price = price_range[idx]
                ax_payoff.axvline(x=breakeven_price, color='orange', linestyle='--', alpha=0.7)
                ax_payoff.text(breakeven_price, ax_payoff.get_ylim()[1] * 0.1, 
                              f'BE: ${breakeven_price:.2f}', rotation=90, 
                              verticalalignment='bottom', fontsize=9)
            
            # Add initial stock price line
            ax_payoff.axvline(x=S0, color='black', linestyle=':', linewidth=2, 
                             label=f'Initial Price: ${S0:.2f}', alpha=0.8)
            
            # Add zero line
            ax_payoff.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            
            ax_payoff.set_xlabel('Stock Price at Expiration ($)')
            ax_payoff.set_ylabel('Portfolio P&L ($)')
            ax_payoff.set_title('Portfolio Payoff Diagram at Expiration')
            ax_payoff.legend()
            ax_payoff.grid(True, alpha=0.3)
            
            st.pyplot(fig_payoff, use_container_width=True)
        else:
            st.markdown("**Portfolio Payoff Diagram**")
            st.info("Add positions to your portfolio to see the payoff diagram.")
