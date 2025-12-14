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
        page_title="Options Visualisation",
        layout="wide"
    )
    
    st.title("European vs American Options")
    
    # Top bar selector (moved before sidebar to determine layout)
    # Portfolio Analysis temporarily disabled
    selected_view = "American vs European"
    # selected_view = st.selectbox(
    #     "Select Analysis Type",
    #     ["Portfolio Analysis", "American vs European"],
    #     label_visibility="collapsed"
    # )
    
    # Dynamic Sidebar based on selected view
    with st.sidebar:
        st.title("Jayden Shi")
        
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
        'portfolio': []
    }
    
    # Display selected content
    theoretical_prices_tab(model_params)
    # if selected_view == "Portfolio Analysis":
    #     pnl_analysis_tab(model_params)
    # else:
    #     theoretical_prices_tab(model_params)
