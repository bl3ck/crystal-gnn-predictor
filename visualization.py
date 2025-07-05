import plotly.graph_objects as go
import torch
import numpy as np
from pymatgen.core import Structure
from typing import Dict

def plot_crystal_structure_3d(structure: Structure, title: str = "Crystal Structure"):
    """Create 3D visualization of crystal structure"""
    
    # Get atomic positions
    positions = structure.cart_coords
    species = [str(site.specie) for site in structure]
    
    # Color mapping for common elements
    color_map = {
        'Si': 'blue', 'Na': 'purple', 'Cl': 'green', 'Li': 'red', 'F': 'orange',
        'O': 'red', 'C': 'black', 'N': 'blue', 'H': 'white', 'Fe': 'brown'
    }
    
    colors = [color_map.get(s, 'gray') for s in species]
    
    # Create 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            opacity=0.8
        ),
        text=species,
        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))
    
    # Add unit cell edges
    lattice = structure.lattice
    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
    ]
    
    cart_vertices = [lattice.get_cartesian_coords(v) for v in vertices]
    
    # Define edges of the unit cell
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    for edge in edges:
        start, end = cart_vertices[edge[0]], cart_vertices[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='cube'
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def plot_explanation_3d(structure: Structure, graph_data, node_feat_mask: torch.Tensor, edge_mask: torch.Tensor, title: str = "GNN Prediction Explanation"):
    """Create 3D visualization of GNN explanation with atom and bond importance."""
    
    positions = structure.cart_coords
    species = [str(site.specie) for site in structure]
    
    # Normalize node feature mask to get per-atom importance (0 to 1)
    # We sum the importance across all features for each atom
    node_importance = node_feat_mask.sum(dim=1).detach().numpy()
    node_colors = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-6)

    # Create 3D scatter plot for atoms, colored by importance
    fig = go.Figure(data=go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=15,
            color=node_colors,
            colorscale='Reds',
            colorbar=dict(title='Atom Importance'),
            opacity=0.8
        ),
        text=species,
        hovertemplate='<b>%{text}</b><br>Importance: %{marker.color:.3f}<extra></extra>'
    ))

    # Add bonds (edges) colored by importance
    edge_index = graph_data.edge_index.cpu().numpy()
    edge_importance = edge_mask.cpu().detach().numpy()

    # Normalize edge importance for line width
    edge_widths = 2 + edge_importance * 8  # Scale width from 2 to 10

    for i in range(edge_index.shape[1]):
        start_idx, end_idx = edge_index[:, i]
        start_pos, end_pos = positions[start_idx], positions[end_idx]
        
        fig.add_trace(go.Scatter3d(
            x=[start_pos[0], end_pos[0]],
            y=[start_pos[1], end_pos[1]],
            z=[start_pos[2], end_pos[2]],
            mode='lines',
            line=dict(color='gray', width=edge_widths[i]),
            hoverinfo='none',
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='cube'
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig

def plot_property_predictions(predictions: Dict, actuals: Dict = None):
    """Plot predicted vs actual properties"""
    
    properties = list(predictions.keys())
    pred_values = list(predictions.values())
    
    if actuals:
        actual_values = [actuals.get(prop, 0) for prop in properties]
        
        fig = go.Figure(data=[
            go.Bar(name='Predicted', x=properties, y=pred_values, marker_color='lightblue'),
            go.Bar(name='Actual', x=properties, y=actual_values, marker_color='darkblue')
        ])
        
        fig.update_layout(
            title='Predicted vs Actual Properties',
            barmode='group',
            yaxis_title='Property Value',
            height=400
        )
    else:
        fig = go.Figure(data=go.Bar(
            x=properties,
            y=pred_values,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Predicted Properties',
            yaxis_title='Property Value',
            height=400
        )
    
    return fig
