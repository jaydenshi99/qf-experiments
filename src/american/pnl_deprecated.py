"""
DEPRECATED P&L ANALYSIS IMPLEMENTATION
=====================================

This file contains the original P&L analysis implementation that was used
before the project pivot. It includes:

- Trade parameters (Position Type, Price per Option, Number of Options, Commission)
- P&L calculations for different option positions
- P&L visualization with red-green color scheme
- Support for Long/Short Call/Put positions

This code is preserved for potential future reference or restoration.

Date deprecated: [Current Date]
Reason: Project pivot to new direction
"""

import streamlit as st
from .core import calculate_node_coordinates, plot_american_options_tree, compare_european_american


def pnl_analysis_tab(model_params, show_intermediate):
    """P&L Analysis tab content"""
    try:
        # Trade parameters in sidebar
        with st.sidebar:
            st.markdown("### Trade Parameters")
            
            # Position type
            position_type = st.selectbox(
                "Position Type",
                ["Long Call", "Short Call", "Long Put", "Short Put", "Custom"],
                help="Type of option position"
            )
            
            if position_type == "Custom":
                st.markdown("**Custom Position Setup**")
                st.markdown("Add multiple option legs for complex strategies")
                st.info("Custom combinations coming soon!")
                price_per_option = 0.01
                num_options = 1
                position_direction = 1  # Long
                option_type_for_pnl = 'call'  # Default to call for custom positions
            else:
                # Determine position direction and option type
                if "Long" in position_type:
                    position_direction = 1
                    option_type_for_pnl = "call" if "Call" in position_type else "put"
                else:  # Short
                    position_direction = -1
                    option_type_for_pnl = "call" if "Call" in position_type else "put"
                
                # Get theoretical price for the selected option type
                theoretical_model_params = model_params.copy()
                theoretical_model_params['option_type'] = option_type_for_pnl
                from .core import compare_european_american
                _, theoretical_model, _, theoretical_price = compare_european_american(theoretical_model_params)
                
                price_per_option = st.number_input(
                    "Price per Option ($)", 
                    min_value=0.01, 
                    value=round(theoretical_price, 2), 
                    step=0.01, 
                    help="Price you paid per option (positive for long, negative for short)"
                )
                num_options = st.number_input(
                    "Number of Options", 
                    min_value=1, 
                    value=1, 
                    step=1,
                    help="Number of individual options"
                )
            
            commission_per_contract = st.number_input(
                "Commission per Contract ($)", 
                min_value=0.00, 
                value=0.30, 
                step=0.01, 
                help="Commission cost per contract (both entry and exit). 1 contract = 100 options"
            )
            
            # Calculate contracts and entry price
            if position_type != "Custom":
                num_contracts = num_options / 100  # Convert options to contracts
                entry_price = price_per_option * position_direction
            else:
                num_contracts = 1 / 100
                entry_price = 0.01 * 1
        
        # Create P&L-specific models with the correct option type
        pnl_model_params = model_params.copy()
        pnl_model_params['option_type'] = option_type_for_pnl
        pnl_european_model, pnl_american_model, _, _ = compare_european_american(pnl_model_params)
        
        # Display all 4 option prices
        st.markdown("### Option Pricing")
        col1, col2, col3, col4 = st.columns(4)
        
        # Create models for both call and put options
        call_model_params = model_params.copy()
        call_model_params['option_type'] = 'call'
        put_model_params = model_params.copy()
        put_model_params['option_type'] = 'put'
        
        # Get all 4 option prices
        eu_call_model, am_call_model, eu_call_price, am_call_price = compare_european_american(call_model_params)
        eu_put_model, am_put_model, eu_put_price, am_put_price = compare_european_american(put_model_params)
        
        with col1:
            st.metric("European Call", f"${eu_call_price:.4f}")
        
        with col2:
            st.metric("European Put", f"${eu_put_price:.4f}")
        
        with col3:
            st.metric("American Call", f"${am_call_price:.4f}")
        
        with col4:
            st.metric("American Put", f"${am_put_price:.4f}")
        
        # Advanced metrics section (if enabled)
        if show_intermediate:
            st.markdown("### Model Parameters")
            colp1, colp2, colp3, colp4 = st.columns(4)
            with colp1:
                st.metric("Timestep Duration", f"{pnl_european_model.dt:.6f}")
            with colp2:
                st.metric("Up Factor", f"{pnl_european_model.u:.6f}")
            with colp3:
                st.metric("Down Factor", f"{pnl_european_model.d:.6f}")
            with colp4:
                st.metric("Risk Neutral Probability", f"{pnl_european_model.p:.6f}")
        
        # P&L Analysis metrics
        st.markdown("### P&L Analysis")
        pnl_col1, pnl_col2, pnl_col3, pnl_col4 = st.columns(4)
        
        with pnl_col1:
            entry_cost = abs(entry_price * num_contracts * 100)
            st.metric("Entry Cost", f"${entry_cost:.2f}")
        
        with pnl_col2:
            total_commission = commission_per_contract * num_contracts * 2
            st.metric("Total Commission", f"${total_commission:.2f}")
        
        with pnl_col3:
            breakeven_price = abs(entry_price) + (total_commission / (num_contracts * 100))
            st.metric("Breakeven Price", f"${breakeven_price:.4f}")
        
        with pnl_col4:
            max_loss = entry_cost + total_commission
            st.metric("Max Loss", f"${max_loss:.2f}")
        
        # Plot P&L analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("European Options P&L")
            coordinates_eu = calculate_node_coordinates(pnl_european_model)
            fig_eu, _ = plot_american_options_tree(
                pnl_european_model, coordinates_eu,
                show_pnl=True,
                entry_price=entry_price,
                num_contracts=num_contracts,
                commission_per_contract=commission_per_contract,
                position_direction=position_direction,
                option_type_for_pnl=option_type_for_pnl
            )
            st.pyplot(fig_eu, use_container_width=True)
        
        with col2:
            st.subheader("American Options P&L")
            coordinates_us = calculate_node_coordinates(pnl_american_model)
            fig_us, _ = plot_american_options_tree(
                pnl_american_model, coordinates_us,
                show_pnl=True,
                entry_price=entry_price,
                num_contracts=num_contracts,
                commission_per_contract=commission_per_contract,
                position_direction=position_direction,
                option_type_for_pnl=option_type_for_pnl
            )
            st.pyplot(fig_us, use_container_width=True)
        
    except ValueError as e:
        st.error(f"Parameter Error: {e}")
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        st.exception(e)
