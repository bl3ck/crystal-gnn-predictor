import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from typing import List, Tuple, Dict

class CrystalGraphCreator:
    """Convert crystal structures to graph representations"""
    
    def __init__(self, cutoff_radius=4.0):
        self.cutoff_radius = cutoff_radius
        self.crystal_nn = CrystalNN()
    
    def structure_to_graph(self, structure: Structure) -> Data:
        """Convert pymatgen Structure to PyTorch Geometric Data object"""
        
        # Get node features (atomic numbers and properties)
        atomic_numbers = [site.specie.Z for site in structure]
        node_features = torch.tensor(atomic_numbers, dtype=torch.float).view(-1, 1)
        
        # Add additional atomic features
        atomic_masses = [site.specie.atomic_mass for site in structure]
        atomic_radii = [site.specie.atomic_radius or 1.0 for site in structure]
        
        additional_features = torch.tensor([
            atomic_masses,
            atomic_radii
        ], dtype=torch.float).T
        
        node_features = torch.cat([node_features, additional_features], dim=1)
        
        # Get edge indices using structure graph
        try:
            sg = StructureGraph.with_local_env_strategy(structure, self.crystal_nn)
            edges = []
            edge_attrs = []
            
            for i, neighbors in enumerate(sg.graph.adjacency()):
                for neighbor_data in neighbors:
                    j = neighbor_data[0]
                    edge_data = neighbor_data[1]
                    
                    # Add edge
                    edges.append([i, j])
                    
                    # Edge features: distance
                    distance = structure.get_distance(i, j)
                    edge_attrs.append([distance])
            
            if len(edges) == 0:
                # Fallback: create edges based on distance cutoff
                edges, edge_attrs = self._distance_based_edges(structure)
            
            edge_index = torch.tensor(edges, dtype=torch.long).T
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            
        except:
            # Fallback method
            edge_index, edge_attr = self._distance_based_edges(structure)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def _distance_based_edges(self, structure: Structure) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback method to create edges based on distance cutoff"""
        edges = []
        edge_attrs = []
        
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                distance = structure.get_distance(i, j)
                if distance <= self.cutoff_radius:
                    edges.extend([[i, j], [j, i]])  # Undirected graph
                    edge_attrs.extend([[distance], [distance]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.empty((0, 1), dtype=torch.float)
        
        return edge_index, edge_attr

class MaterialsDataLoader:
    """Load and process materials data from Materials Project"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.graph_creator = CrystalGraphCreator()
    
    def get_sample_materials(self, num_samples=100) -> List[Dict]:
        """Get sample materials data for demonstration"""
        
        # For demo purposes, we'll use some common crystal systems
        sample_data = [
            {
                'material_id': 'mp-149',
                'formula': 'Si',
                'band_gap': 1.1,
                'formation_energy_per_atom': 0.0,
                'density': 2.33,
                'structure': self._get_silicon_structure()
            },
            {
                'material_id': 'mp-66',
                'formula': 'NaCl',
                'band_gap': 6.8,
                'formation_energy_per_atom': -2.1,
                'density': 2.17,
                'structure': self._get_nacl_structure()
            },
            {
                'material_id': 'mp-2',
                'formula': 'LiF',
                'band_gap': 12.7,
                'formation_energy_per_atom': -4.8,
                'density': 2.64,
                'structure': self._get_lif_structure()
            }
        ]
        
        return sample_data[:num_samples]
    
    def _get_silicon_structure(self) -> Structure:
        """Create silicon diamond structure"""
        from pymatgen.core import Lattice, Structure
        
        lattice = Lattice.cubic(5.431)
        species = ["Si", "Si", "Si", "Si", "Si", "Si", "Si", "Si"]
        coords = [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.0],
            [0.75, 0.75, 0.25],
            [0.5, 0.0, 0.5],
            [0.75, 0.25, 0.75],
            [0.0, 0.5, 0.5],
            [0.25, 0.75, 0.75]
        ]
        
        return Structure(lattice, species, coords)
    
    def _get_nacl_structure(self) -> Structure:
        """Create NaCl rock salt structure"""
        from pymatgen.core import Lattice, Structure
        
        lattice = Lattice.cubic(5.64)
        species = ["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"]
        coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5]
        ]
        
        return Structure(lattice, species, coords)
    
    def _get_lif_structure(self) -> Structure:
        """Create LiF rock salt structure"""
        from pymatgen.core import Lattice, Structure
        
        lattice = Lattice.cubic(4.03)
        species = ["Li", "F", "Li", "F", "Li", "F", "Li", "F"]
        coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5]
        ]
        
        return Structure(lattice, species, coords)
