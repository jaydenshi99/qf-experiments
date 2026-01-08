"""
Main Streamlit application for Binomial Options Pricing Model

This module orchestrates the interface for theoretical pricing visualization.
"""

import streamlit as st
from .theoretical import theoretical_prices_tab


def main():
    st.set_page_config(
        page_title="Options Visualisation",
        layout="wide"
    )
    
    st.title("European vs American Options")

    # Sidebar
    with st.sidebar:
        st.title("Parameters")
        
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

    # Model parameters dictionary
    model_params = {
        'S0': S0,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma,
        'n_steps': n_steps,
        'show_intermediate': show_intermediate
    }

    # Display content
    theoretical_prices_tab(model_params)
