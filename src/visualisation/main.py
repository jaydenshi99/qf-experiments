"""
Main Streamlit application for Binomial Options Pricing Model

This module orchestrates the tab-based interface for theoretical pricing
and P&L analysis visualizations.
"""

import streamlit as st
from .theoretical import theoretical_prices_tab
from .pnl import pnl_analysis_tab


def main():
    st.set_page_config(
        page_title="Options Playground",
        page_icon="🎮",
        layout="wide"
    )
    
    st.title("Options Playground")
    
    # Top bar selector (moved before sidebar to determine layout)
    selected_view = st.selectbox(
        "Select Analysis Type",
        ["📊 Portfolio Analysis", "📈 American vs European"],
        label_visibility="collapsed"
    )
    
    # Dynamic Sidebar based on selected view
    with st.sidebar:
        st.title("🎮 Options Playground")
        
        if selected_view == "📊 Portfolio Analysis":
            # Core parameters
            st.markdown("### Stock Parameters")
            S0 = st.number_input("Initial Stock Price (S₀)", min_value=0.01, value=100.00, step=1.00, format="%.2f")
            T = st.number_input("Time Horizon (T)", min_value=0.01, max_value=10.00, value=0.25, step=0.01, format="%.2f", help="Simulation time period in years")
            r = st.number_input("Risk-free Rate (r)", min_value=0.000, max_value=1.000, value=0.050, step=0.001, format="%.3f")
            sigma = st.number_input("Volatility (σ)", min_value=0.010, max_value=2.000, value=0.200, step=0.010, format="%.3f")
            
            use_risk_free = st.checkbox("Use Risk-Free Rate as Drift", value=True, 
                                       help="Use r as drift rate (risk-neutral valuation)")
            
            if not use_risk_free:
                custom_drift = st.number_input("Custom Drift Rate", 
                                             min_value=-0.5, max_value=0.5, 
                                             value=0.1, step=0.01, format="%.3f")
                mu = custom_drift
            else:
                mu = r
            
            # Set default for strike price (used in option pricing calculations)
            K = 105.00
            
            # Simulation parameters
            st.markdown("### Simulation Parameters")
            n_paths = st.slider("Number of Paths", 100, 5000, 1000, 100)
            n_time_steps = st.slider("Time Steps", 50, 500, 252, 50)
            
            # Set defaults for unused parameters
            n_steps = 4
            show_intermediate = False
            
        else:
            # Show model parameters checkbox
            show_intermediate = st.checkbox("Show Model Parameters", help="Show intermediate calculation values (dt, u, d, p)")
            
            # Core parameters
            st.markdown("### Option Parameters")
            S0 = st.number_input("Initial Stock Price (S₀)", min_value=0.01, value=100.00, step=1.00, format="%.2f")
            K = st.number_input("Strike Price (K)", min_value=0.01, value=105.00, step=1.00, format="%.2f")
            T = st.number_input("Time Horizon (T)", min_value=0.01, max_value=10.00, value=0.25, step=0.01, format="%.2f", help="Analysis time period in years")
            r = st.number_input("Risk-free Rate (r)", min_value=0.000, max_value=1.000, value=0.050, step=0.001, format="%.3f")
            sigma = st.number_input("Volatility (σ)", min_value=0.010, max_value=2.000, value=0.200, step=0.010, format="%.3f")
            
            # Tree parameters
            st.markdown("### Tree Parameters")
            n_steps = st.slider("Number of Steps", 1, 10, 4, 1)
            
            # Set defaults for unused parameters
            n_paths = 1000
            n_time_steps = 252
            mu = r
    
    # Model parameters dictionary
    model_params = {
        'S0': S0,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma,
        'n_steps': n_steps,
        'n_paths': n_paths,
        'n_time_steps': n_time_steps,
        'mu': mu,
        'show_intermediate': show_intermediate,
        'portfolio': st.session_state.get('portfolio', []) if selected_view == "📊 Portfolio Analysis" else []
    }
    
    # Display selected content
    if selected_view == "📊 Portfolio Analysis":
        pnl_analysis_tab(model_params)
    else:
        theoretical_prices_tab(model_params)
