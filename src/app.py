"""
Simple Streamlit app for Binomial Options Pricing Model
"""

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from binomial_model import BinomialModel
    
    st.title("Binomial Options Pricing Model")
    st.write("Backend is working! Model can be imported successfully.")
    
    # Simple test
    model = BinomialModel(S0=100, K=105, T=0.25, r=0.05, sigma=0.2, n_steps=10)
    st.write("Model created:", model)
    
except Exception as e:
    st.error(f"Error: {e}")
    st.write("Let's fix the backend first!")
