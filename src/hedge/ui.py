"""
UI components and sidebar setup for the Portfolio Hedging application.
"""

import streamlit as st


def render_sidebar():
    """Render the sidebar with parameters and settings."""
    with st.sidebar:
        st.title("Parameters")
        
        st.markdown("### Market Parameters")
        p = st.slider("Probability of Heads (p)", 0.0, 1.0, 0.5, 0.01, 
                      help="Probability that the coin lands heads")
    
    return p


def render_intro():
    """Render the introduction section."""
    st.markdown("""
    ### Toy Market
    
    To investigate optimal hedging, we propose a toy market based off a series of coinflips. 
    Each flip, the coin has a probability $p$ of landing heads, and probability $(1-p)$ of landing tails. 
    We are presented a series of bets on the value of $H_t$, the number of heads in $t$ flips. 
    This streamlit application aims to use this toy market to explore optimal hedging strategies.

    To get started, enter some bets below.
    """)

