"""
Portfolio Hedging Application

A Streamlit application for analyzing and hedging portfolio risk.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from src.hedge.ui import render_sidebar, render_intro
from src.hedge.bet_management import render_add_bet_section, render_bets_display
from src.hedge.heatmap import render_heatmap_section
from src.hedge.performance_testing import render_performance_testing_section
from src.hedge.optimal_allocation_test import render_optimal_allocation_test


def main():
    st.set_page_config(
        page_title="Portfolio Hedging",
        layout="wide"
    )
    
    st.title("Portfolio Hedging")
    
    # Initialize session state for bets
    if 'bets' not in st.session_state:
        st.session_state.bets = []
    
    # Render sidebar and get parameters
    p = render_sidebar()
    
    # Render introduction
    render_intro()
    
    # Bet management section
    render_add_bet_section()
    render_bets_display()
    
    # Heatmap section
    render_heatmap_section(st.session_state.bets, p)
    
    # Optimal allocation test section
    render_optimal_allocation_test(st.session_state.bets, p)
    
    # Performance testing section
    render_performance_testing_section(st.session_state.bets, p)


if __name__ == "__main__":
    main()
