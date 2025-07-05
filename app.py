# Crystal Structure Property Predictor using Graph Neural Networks
# A complete implementation for materials informatics

import os
import numpy as np
import pandas as pd
import torch
import streamlit as st
import warnings

from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter

from config import Config
from data_processing import CrystalGraphCreator, MaterialsDataLoader
from model import CrystalGNN
from visualization import plot_crystal_structure_3d, plot_property_predictions, plot_explanation_3d
from utils import create_dummy_model, predict_properties, explain_prediction

warnings.filterwarnings('ignore')

# ==================== Data Fetching ====================
def fetch_material_data(mp_id):
    """Fetches comprehensive material data from the Materials Project."""
    if not mp_id:
        st.warning("Please enter a Material ID.")
        return None
    
    with st.spinner(f"Fetching data for {mp_id}..."):
        try:
            with MPRester(api_key=Config.MP_API_KEY) as mpr:
                # 1. Fetch core material data (structure, formula, density)
                material_docs = mpr.materials.search(
                    material_ids=[mp_id],
                    fields=["material_id", "formula_pretty", "structure", "density"]
                )
                if not material_docs:
                    st.error(f"Material with ID {mp_id} not found.")
                    return None

                material_doc = material_docs[0]

                # 2. Fetch thermodynamic properties
                thermo_docs = mpr.thermo.search(
                    material_ids=[mp_id],
                    fields=["formation_energy_per_atom"]
                )
                formation_energy = thermo_docs[0].formation_energy_per_atom if thermo_docs else None

                # 3. Fetch electronic structure properties (band gap)
                bs_docs = mpr.electronic_structure.search(
                    material_ids=[mp_id],
                    fields=["band_gap"]
                )
                band_gap = bs_docs[0].band_gap if bs_docs else None

                # 4. Combine all data
                material_data = {
                    'material_id': material_doc.material_id,
                    'formula': material_doc.formula_pretty,
                    'structure': material_doc.structure,
                    'density': material_doc.density,
                    'formation_energy_per_atom': formation_energy,
                    'band_gap': band_gap
                }
                st.success(f"Successfully fetched {material_data['formula']}!")
                return material_data

        except Exception as e:
            st.error(f"Could not fetch data for {mp_id}. Please check the ID and your API key. Error: {e}")
            return None

# ==================== Streamlit App ====================
# Set Streamlit theme and page configuration
st.set_page_config(
    page_title="CrystaLytics: A GNN-powered Materials Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar branding
st.sidebar.markdown(
    """
    ## CrystaLytics
    **Explore and predict material properties with cutting-edge AI.**
    """
)

def main():
    st.title("CrystaLytics")  # Removed emoji icon
    st.markdown("""
    **A GNN-powered tool for exploring and predicting material properties.**
    
    This application uses the Materials Project API to fetch real-time crystal structure data
    and demonstrates how Graph Neural Networks can be used to predict material properties.
    """)
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", ["Demo", "About", "Technical Details"])
    
    if page == "Demo":
        demo_page()
    elif page == "About":
        about_page()
    else:
        technical_page()

# Improved layout for the Demo page
def demo_page():
    st.header("Interactive Demo")  # Removed emoji icon

    # Material Selection Section
    st.subheader("Select or Find a Material")
    with st.expander("Find by Materials Project ID", expanded=True):
        # Initialize session state and fetch default material on first load
        if 'selected_material' not in st.session_state:
            st.session_state.selected_material = fetch_material_data("mp-149")
            st.session_state.predictions = None
            st.session_state.explanation = None # Initialize explanation state

        mp_id_input = st.text_input("Enter a Material ID (e.g., mp-149 for Si, mp-22862 for GaN):", key="mp_id_input")

        if st.button("Fetch Material Data", key="fetch_mp_id"):
            st.session_state.predictions = None
            st.session_state.explanation = None # Clear old explanation
            st.session_state.selected_material = fetch_material_data(mp_id_input)
            st.rerun()

    with st.expander("Choose a Pre-loaded Sample"):
        # Corresponds to "Choose a Pre-loaded Sample"
        data_loader = MaterialsDataLoader("demo_key")
        sample_materials = data_loader.get_sample_materials()
        material_options = [f"{mat['formula']} ({mat['material_id']})" for mat in sample_materials]
        
        selected_idx = st.selectbox(
            "Select a sample:", 
            range(len(material_options)), 
            format_func=lambda x: material_options[x], 
            key="sample_select"
        )
        
        # Update state only if the selection has changed to prevent unnecessary reruns
        if st.session_state.selected_material.get('material_id') != sample_materials[selected_idx]['material_id']:
            st.session_state.selected_material = sample_materials[selected_idx]
            st.session_state.predictions = None
            st.session_state.explanation = None # Clear old explanation
            st.rerun()

    # Display Material Properties and Visualization
    st.subheader("Material Analysis")
    if st.session_state.selected_material:
        selected_material = st.session_state.selected_material
        structure = selected_material['structure']
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Formula:**", selected_material['formula'])
            st.write("**Material ID:**", selected_material['material_id'])

            # --- Property Display ---
            st.subheader("Property Comparison")
            for prop, label in Config.PROPERTIES.items():
                actual_val = selected_material.get(prop)
                
                # Display Actual Value
                actual_display = f"{actual_val:.3f}" if isinstance(actual_val, (int, float)) else "N/A"
                st.write(f"**{label} (Actual):** {actual_display}")

                # Display Predicted Value if available
                if st.session_state.predictions:
                    pred_val = st.session_state.predictions.get(prop)
                    pred_display = f"{pred_val:.3f}" if isinstance(pred_val, (int, float)) else "N/A"
                    
                    delta_text = None
                    if isinstance(actual_val, (int, float)) and isinstance(pred_val, (int, float)) and actual_val != 0:
                        error = abs((pred_val - actual_val) / actual_val) * 100
                        delta_text = f"{error:.1f}% error"
                    
                    st.metric(
                        label=f"{label} (Predicted)",
                        value=pred_display,
                        delta=delta_text,
                        delta_color="inverse"
                    )

            # --- Crystallographic Analysis ---
            st.subheader("Crystallographic Details")
            try:
                sga = SpacegroupAnalyzer(structure)
                st.write(f"**Crystal System:** {sga.get_crystal_system()}")
                st.write(f"**Space Group:** {sga.get_space_group_symbol()}")
                st.write(f"**Lattice Type:** {sga.get_lattice_type()}")
            except Exception as e:
                st.warning(f"Could not perform crystallographic analysis. Error: {e}")

            # --- Download Button ---
            st.subheader("Download")
            try:
                cif_writer = CifWriter(structure)
                cif_string = str(cif_writer)
                st.download_button(
                    label="Download .cif File",
                    data=cif_string,
                    file_name=f"{selected_material['material_id']}.cif",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Failed to generate .cif file. Error: {e}")

            # --- Prediction Button ---
            st.subheader("GNN Predictions")
            if st.button("Predict Properties", type="primary"):
                with st.spinner("Running GNN model and generating explanation..."):
                    graph_creator = CrystalGraphCreator()
                    graph_data = graph_creator.structure_to_graph(structure)  # Ensure graph_data is initialized
                    model = create_dummy_model()
                    
                    # Store predictions in session state
                    st.session_state.predictions = predict_properties(model, graph_data)
                    
                    # Generate and store explanation
                    node_mask, edge_mask = explain_prediction(model, graph_data)
                    st.session_state.explanation = {
                        'node_mask': node_mask,
                        'edge_mask': edge_mask
                    }
                    st.session_state.graph_data = graph_data  # Store graph_data in session state
                    st.rerun() # Rerun to update the UI

        with col2:
            st.subheader("Crystal Structure Visualization")
            fig_3d = plot_crystal_structure_3d(structure, f"{selected_material['formula']} Crystal Structure")
            st.plotly_chart(fig_3d, use_container_width=True)

            # --- Prediction Analysis ---
            if st.session_state.predictions:
                st.header("Prediction Analysis")
                actual_props = {p: selected_material.get(p) for p in Config.PROPERTIES}
                predicted_props = st.session_state.predictions
                fig = plot_property_predictions(actual_props, predicted_props)
                st.plotly_chart(fig, use_container_width=True)

            # --- XAI Explanation ---
            if st.session_state.explanation:
                st.header("Prediction Explanation (XAI)")
                st.markdown("This visualization highlights the atoms and bonds that were most influential in predicting the band gap. Red atoms and thicker bonds have higher importance.")
                explanation = st.session_state.explanation
                fig_exp = plot_explanation_3d(
                    structure,
                    st.session_state.graph_data,  # Retrieve graph_data from session state
                    explanation['node_mask'],
                    explanation['edge_mask'],
                    title="GNN Prediction Explanation"
                )
                st.plotly_chart(fig_exp, use_container_width=True)

# Improved layout for the About page
def about_page():
    st.header("About CrystaLytics")
    st.markdown("""
    CrystaLytics is a professional tool designed for materials scientists and engineers.
    It leverages advanced Graph Neural Networks to predict material properties and provides
    explainable AI insights into the predictions.
    """)
    st.markdown(
        """
        ## Project Overview
        
        CrystaLytics demonstrates the power of Graph Neural Networks (GNNs) 
        in materials informatics. By representing crystal structures as graphs where atoms are nodes 
        and chemical bonds are edges, we can predict important material properties that guide 
        materials discovery and design.

        This application fetches live data directly from the **Materials Project**, a foundational open-access database
        of computational materials properties. It serves as both a powerful visualization tool and as a portfolio 
        piece demonstrating skills in machine learning, data science, and web application development.

        ## Scientific Impact
        
        The ability to rapidly predict material properties can significantly accelerate 
        the discovery of new materials for applications in renewable energy (solar cells, batteries),
        electronics (semiconductors), and structural engineering. By providing an intuitive interface
        to a complex model, CrystaLytics aims to make materials informatics more accessible.

        --- 

        ***Acknowledgments:** This project is powered by the [Materials Project](https://materialsproject.org/) and its 
        extraordinary open-access database. We are grateful for their commitment to advancing materials science.* 
        """
    )

# Improved layout for the Technical Details page
def technical_page():
    st.header("Technical Details")
    st.markdown("""
    **Graph Neural Network Architecture:**
    - Input features: Atomic properties
    - Hidden layers: Graph Convolutional Networks
    - Output: Predicted material properties (Band Gap, Formation Energy, Density)

    **Explainable AI:**
    - Uses GNNExplainer to highlight atom and bond importance.
    - Provides insights into model predictions.
    """)

# Main function
if __name__ == "__main__":
    main()