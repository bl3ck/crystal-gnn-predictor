import streamlit as st
import torch

class Config:
    MP_API_KEY = st.secrets.get("MP_API_KEY", "MACqP7aSfKafCcwSNzbGncZixvqlfKPl")
    MODEL_PATH = "crystal_gnn_model.pkl"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Target properties to predict
    PROPERTIES = {
        'band_gap': 'Band Gap (eV)',
        'formation_energy_per_atom': 'Formation Energy (eV/atom)',
        'density': 'Density (g/cm³)'
    }
