import torch
import numpy as np
from model import CrystalGNN
from torch_geometric.explain import Explainer, GNNExplainer # Corrected import path

def create_dummy_model():
    """Create a dummy trained model for demonstration"""
    model = CrystalGNN(input_dim=3, hidden_dim=64, output_dim=3)  # 3 outputs for 3 properties
    
    # Set to evaluation mode
    model.eval()
    
    return model

def predict_properties(model, graph_data):
    """Predict properties using the trained model"""
    model.eval()
    
    with torch.no_grad():
        # Create a batch with single graph
        batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
        graph_data.batch = batch
        
        # Make prediction
        pred = model(graph_data.x, graph_data.edge_index, graph_data.batch)
        
        # Convert to dictionary
        properties = ['band_gap', 'formation_energy_per_atom', 'density']
        predictions = {prop: float(pred[0, i]) for i, prop in enumerate(properties)}
        
        return predictions


def explain_prediction(model, data):
    """Generates node and edge importance masks using the modern Explainer API."""
    
    # 1. Configure the Explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100), # Pass the algorithm with its config
        explanation_type='model', # Explain the model's behavior
        model_config=dict(
            mode='regression',      # We are predicting continuous values
            task_level='graph',    # We are explaining a graph-level prediction
            return_type='raw'       # We want the raw model output
        ),
        node_mask_type='object', # Get a single mask for all node features
        edge_mask_type='object'  # Get a single mask for the edges
    )

    # 2. Generate the explanation for the graph
    # We need to explain a specific prediction. Let's focus on the band gap (output index 0).
    explanation = explainer(
        x=data.x, 
        edge_index=data.edge_index, 
        batch=data.batch, 
        target=model(data.x, data.edge_index, data.batch)[:, 0]
    )

    # 3. Return the masks
    return explanation.node_mask, explanation.edge_mask
