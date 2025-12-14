"""
Bet management functions for adding, displaying, and deleting bets.
"""

import streamlit as st


def render_add_bet_section():
    """Render the section for adding new bets."""
    st.markdown("### Bets")
    
    # Add new bet section
    with st.expander("Add New Bet", expanded=len(st.session_state.bets) == 0):
        bet_type = st.radio(
            "Bet Type",
            ["Simple Bet (Condition + Odds)", "Payoff Function"],
            key="new_bet_type",
            help="Simple Bet: condition-based with fixed odds. Payoff Function: flexible expression-based payoffs."
        )
        
        if bet_type == "Simple Bet (Condition + Odds)":
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
            
            if st.button("Add Bet", type="primary", key="add_simple_bet"):
                st.session_state.bets.append({
                    'condition': bet_condition,
                    'odds': bet_odds
                })
                st.rerun()
        
        else:  # Payoff Function
            st.markdown("""
            **Payoff Function**: Define total profit/loss as a function of H_t (number of heads at time t) and I (investment amount).
            
            **Examples:**
            - Call option: `max(H_5 - 3, 0) * 0.5 * I - 0.1 * I` (profit = $0.50 per head above strike 3 times investment, minus 10% premium)
            - Put option: `max(3 - H_5, 0) * 0.5 * I - 0.1 * I`
            - Simple return: `I * 0.2` (20% return regardless of outcome)
            - Spread: `max(min(H_5, 5) - 3, 0) * 0.5 * I`
            
            **Available functions:** `max()`, `min()`, `abs()`  
            **Variables:** 
            - `H_t` where t is the time (e.g., `H_5` for heads at time 5)
            - `I` represents the dollar amount invested (automatically set to your allocation)
            """)
            
            bet_payoff = st.text_input(
                "Payoff Function",
                value="max(H_5 - 3, 0) * 0.5 * I - 0.1 * I",
                key="new_bet_payoff",
                help="Mathematical expression for total profit/loss. Use variable 'I' for investment amount."
            )
            
            if st.button("Add Bet", type="primary", key="add_payoff_bet"):
                st.session_state.bets.append({
                    'payoff': bet_payoff
                })
                st.rerun()


def render_bets_display():
    """Render the current bets display section."""
    if st.session_state.bets:
        st.markdown("#### Current Bets")
        
        # Header row
        col1, col2, col3, col4 = st.columns([1, 4, 2, 1])
        with col1:
            st.caption("**#**")
        with col2:
            st.caption("**Bet Details**")
        with col3:
            st.caption("**Type**")
        with col4:
            st.caption("**Action**")
        
        # Display each bet
        for i, bet in enumerate(st.session_state.bets):
            col1, col2, col3, col4 = st.columns([1, 4, 2, 1])
            
            with col1:
                st.write(f"{i + 1}")
            
            with col2:
                if 'payoff' in bet:
                    st.code(bet['payoff'])
                else:
                    st.code(bet.get('condition', 'N/A'))
            
            with col3:
                if 'payoff' in bet:
                    st.caption("Payoff Function")
                else:
                    st.caption(f"Simple ({bet.get('odds', 'N/A'):.2f}x)")
            
            with col4:
                if st.button("Delete", key=f"delete_{i}", help="Delete bet"):
                    st.session_state.bets.pop(i)
                    st.rerun()
        
        # Clear all button
        if st.button("Clear All Bets", type="secondary"):
            st.session_state.bets = []
            st.rerun()
    else:
        st.info("No bets entered yet. Click 'Add New Bet' above to get started.")

