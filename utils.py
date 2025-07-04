import torch
import numpy as np
from model import CrystalGNN

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
        pred = model(graph_data)
        
        # Convert to dictionary
        properties = ['band_gap', 'formation_energy_per_atom', 'density']
        predictions = {prop: float(pred[0, i]) for i, prop in enumerate(properties)}
    
    return predictions
