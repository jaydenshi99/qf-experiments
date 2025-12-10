"""
Portfolio Hedging Application

A Streamlit application for analyzing and hedging portfolio risk.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


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
                value="H_2 >= 1",
                key="new_bet_condition",
                help="Payout condition (e.g., 'H_2 >= 1', 'H_5 == 3', 'H_5 < 10')"
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


if __name__ == "__main__":
    main()
