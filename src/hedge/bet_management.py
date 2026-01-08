"""
Bet management functions for adding, displaying, and deleting bets.
"""

import streamlit as st
from .presets import PRESET_TYPES


def render_add_bet_section():
    """Render add bet section."""
    st.markdown("### Bets")
    
    # Add new bet section
    with st.expander("Add New Bet", expanded=True):
        bet_type = st.radio(
            "Bet Type",
            ["Preset", " Custom"],
            key="new_bet_type",
            help="Preset: common bet types with easy parameters. Custom: flexible expression-based payoffs."
        )
        
        if bet_type == "Preset":
            # Select preset type
            preset_name = st.selectbox(
                "Preset Type",
                options=list(PRESET_TYPES.keys()),
                key="preset_type",
                help="Select a preset bet type"
            )
            
            preset_info = PRESET_TYPES[preset_name]
            st.caption(preset_info["description"])
            
            # Handle Simple Bet specially (it uses condition + odds, not payoff function)
            if preset_info.get("type") == "simple":
                params = preset_info["params"]
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    bet_condition = st.text_input(
                        params["condition"]["label"],
                        value=params["condition"]["default"],
                        key="preset_condition",
                        help=params["condition"]["help"]
                    )
                
                with col2:
                    bet_odds = st.number_input(
                        params["odds"]["label"],
                        min_value=params["odds"]["min"],
                        value=params["odds"]["default"],
                        step=params["odds"].get("step", 0.05),
                        format="%.2f",
                        key="preset_odds",
                        help=params["odds"]["help"]
                    )
                
                if st.button("Add Bet", type="primary", key="add_preset_bet"):
                    st.session_state.bets.append({
                        'condition': bet_condition,
                        'odds': bet_odds,
                        'preset_type': preset_name
                    })
                    st.rerun()
            
            else:
                # Handle other presets (generate payoff functions)
                params = preset_info["params"]
                param_values = {}
                
                # Create input fields based on preset parameters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    param_values["strike"] = st.number_input(
                        params["strike"]["label"],
                        min_value=params["strike"]["min"],
                        value=params["strike"]["default"],
                        step=1,
                        key="preset_strike",
                        help=params["strike"]["help"]
                    )
                
                with col2:
                    param_values["time_to_expiry"] = st.number_input(
                        params["time_to_expiry"]["label"],
                        min_value=params["time_to_expiry"]["min"],
                        value=params["time_to_expiry"]["default"],
                        step=1,
                        key="preset_time",
                        help=params["time_to_expiry"]["help"]
                    )
                
                with col3:
                    param_values["payout_multiplier"] = st.number_input(
                        params["payout_multiplier"]["label"],
                        min_value=params["payout_multiplier"]["min"],
                        value=params["payout_multiplier"]["default"],
                        step=params["payout_multiplier"].get("step", 0.01),
                        format="%.2f",
                        key="preset_payout",
                        help=params["payout_multiplier"]["help"]
                    )
                
                # Generate payoff function preview
                payoff_expr = preset_info["function"](**param_values)
                st.code(payoff_expr, language="python")
                st.caption("Generated payoff function")
                
                if st.button("Add Bet", type="primary", key="add_preset_bet"):
                    st.session_state.bets.append({
                        'payoff': payoff_expr,
                        'preset_type': preset_name,
                        'preset_params': param_values
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
    """Render bets display."""
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
                if 'preset_type' in bet:
                    st.caption(f"Preset: {bet['preset_type']}")
                elif 'payoff' in bet:
                    st.caption("Payoff Function")
                elif 'condition' in bet:
                    st.caption(f"Simple ({bet.get('odds', 'N/A'):.2f}x)")
                else:
                    st.caption("Unknown")
            
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

