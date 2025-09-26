"""
Theoretical Prices tab for Binomial Options Pricing Model

This module handles the theoretical pricing visualization tab,
showing European vs American option pricing with early exercise analysis.
"""

import streamlit as st
from .core import calculate_node_coordinates, plot_american_options_tree, compare_european_american


def theoretical_prices_tab(model_params):
    """Theoretical Prices tab content"""
    try:
        # Extract parameters from model_params
        show_intermediate = model_params['show_intermediate']
        
        # Filter parameters for binomial model (exclude Monte Carlo specific ones)
        binomial_params = {
            'S0': model_params['S0'],
            'K': model_params['K'],
            'T': model_params['T'],
            'r': model_params['r'],
            'sigma': model_params['sigma'],
            'n_steps': model_params['n_steps']
        }
        
        # Create models for both call and put options
        call_params = binomial_params.copy()
        call_params['option_type'] = 'call'
        put_params = binomial_params.copy()
        put_params['option_type'] = 'put'
        
        # Get all 4 option prices
        eu_call_model, am_call_model, eu_call_price, am_call_price = compare_european_american(call_params)
        eu_put_model, am_put_model, eu_put_price, am_put_price = compare_european_american(put_params)
        
        # Display all 4 option prices
        st.markdown("### Option Pricing")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("European Call", f"${eu_call_price:.4f}")
        
        with col2:
            st.metric("European Put", f"${eu_put_price:.4f}")
        
        with col3:
            st.metric("American Call", f"${am_call_price:.4f}")
        
        with col4:
            st.metric("American Put", f"${am_put_price:.4f}")
        
        # Advanced metrics section
        adv_placeholder = st.container()
        if show_intermediate:
            with adv_placeholder:
                st.markdown("### Model Parameters")
                colp1, colp2, colp3, colp4 = st.columns(4)
                with colp1:
                    st.metric("Timestep Duration", f"{am_call_model.dt:.6f}")
                with colp2:
                    st.metric("Up Factor", f"{am_call_model.u:.6f}")
                with colp3:
                    st.metric("Down Factor", f"{am_call_model.d:.6f}")
                with colp4:
                    st.metric("Risk Neutral Probability", f"{am_call_model.p:.6f}")
        else:
            with adv_placeholder:
                # Small spacer to reduce layout shift without leaving large gaps
                st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
        
        # Show all 4 trees in a 2x2 grid
        st.markdown("### Option Trees")
        
        # First row: European Call and American Call
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🇪🇺 European Call")
            coordinates_eu_call = calculate_node_coordinates(eu_call_model)
            fig_eu_call, _ = plot_american_options_tree(
                eu_call_model, coordinates_eu_call, 
                show_pnl=False
            )
            st.pyplot(fig_eu_call, use_container_width=True)
        
        with col2:
            st.subheader("🇺🇸 American Call")
            coordinates_am_call = calculate_node_coordinates(am_call_model)
            fig_am_call, early_exercise_nodes_call = plot_american_options_tree(
                am_call_model, coordinates_am_call,
                show_pnl=False
            )
            st.pyplot(fig_am_call, use_container_width=True)
        
        # Second row: European Put and American Put
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🇪🇺 European Put")
            coordinates_eu_put = calculate_node_coordinates(eu_put_model)
            fig_eu_put, _ = plot_american_options_tree(
                eu_put_model, coordinates_eu_put,
                show_pnl=False
            )
            st.pyplot(fig_eu_put, use_container_width=True)
        
        with col4:
            st.subheader("🇺🇸 American Put")
            coordinates_am_put = calculate_node_coordinates(am_put_model)
            fig_am_put, early_exercise_nodes_put = plot_american_options_tree(
                am_put_model, coordinates_am_put,
                show_pnl=False
            )
            st.pyplot(fig_am_put, use_container_width=True)
        
        # Early exercise analysis
        all_early_exercise_nodes = early_exercise_nodes_call + early_exercise_nodes_put
        if all_early_exercise_nodes:
            with st.expander("⚡ Early Exercise Analysis", expanded=False):
                # Show analysis for calls
                if early_exercise_nodes_call:
                    st.markdown("**American Call Early Exercise:**")
                    st.caption(f"Nodes where early exercise is optimal: {len(early_exercise_nodes_call)}")
                    
                    rows_call = []
                    for t, i in early_exercise_nodes_call:
                        node = am_call_model.tree.nodes[(t, i)]
                        exercise_value = node.get_exercise_value(am_call_model.K, am_call_model.option_type)
                        rows_call.append({
                            "Node": f"({t},{i})",
                            "S": f"${node.stock_price:.2f}",
                            "Exercise": f"${exercise_value:.2f}",
                            "Option": f"${node.option_price:.2f}"
                        })
                    st.dataframe(rows_call, width='stretch', hide_index=True)
                else:
                    st.markdown("**American Call:** No early exercise is optimal")
                
                # Show analysis for puts
                if early_exercise_nodes_put:
                    st.markdown("**American Put Early Exercise:**")
                    st.caption(f"Nodes where early exercise is optimal: {len(early_exercise_nodes_put)}")
                    
                    rows_put = []
                    for t, i in early_exercise_nodes_put:
                        node = am_put_model.tree.nodes[(t, i)]
                        exercise_value = node.get_exercise_value(am_put_model.K, am_put_model.option_type)
                        rows_put.append({
                            "Node": f"({t},{i})",
                            "S": f"${node.stock_price:.2f}",
                            "Exercise": f"${exercise_value:.2f}",
                            "Option": f"${node.option_price:.2f}"
                        })
                    st.dataframe(rows_put, width='stretch', hide_index=True)
                else:
                    st.markdown("**American Put:** No early exercise is optimal")
        else:
            with st.expander("⚡ Early Exercise Analysis", expanded=False):
                st.write("No early exercise is optimal for either calls or puts – American options behave like European options")
        
        # Model details
        with st.expander("🔍 Model Details"):
            st.json(am_call_model.get_model_info())
        
    except ValueError as e:
        st.error(f"❌ Parameter Error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
        st.exception(e)
