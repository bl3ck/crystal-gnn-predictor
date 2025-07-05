import streamlit as st
import torch
import os

class Config:
    # Try to get API key from multiple sources
    MP_API_KEY = (
        os.getenv("MP_API_KEY") or  # Environment variable (for Railway, Render, etc.)
        st.secrets.get("MP_API_KEY") or  # Streamlit secrets (for Streamlit Cloud)
        None
    )
    MODEL_PATH = "crystal_gnn_model.pkl"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Target properties to predict
    PROPERTIES = {
        'band_gap': 'Band Gap (eV)',
        'formation_energy_per_atom': 'Formation Energy (eV/atom)',
        'density': 'Density (g/cm³)'
    }
