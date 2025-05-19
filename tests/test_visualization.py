import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization import plot_network_graph, plot_map, plot_traffic_comparison, plot_route_visualization

class TestVisualization(unittest.TestCase):
    """Test cases for visualization module functions"""
    
    def setUp(self):
        """Set up test data for visualization tests"""
        # Create sample neighborhoods data
        self.neighborhoods = pd.DataFrame({
            'ID': ['1', '2', '3', '4'],
            'Name': ['Maadi', 'Nasr City', 'Downtown', 'New Cairo'],
            'Population': [250000, 500000, 100000, 300000],
            'Type': ['Residential', 'Mixed', 'Business', 'Residential'],
            'X': [31.25, 31.34, 31.24, 31.47],
            'Y': [29.96, 30.06, 30.04, 30.03]
        })
        
        # Create sample facilities data
        self.facilities = pd.DataFrame({
            'ID': ['F1', 'F2', 'F3'],
            'Name': ['Airport', 'Train Station', 'Hospital'],
            'Type': ['Airport', 'Transit Hub', 'Medical'],
            'X': [31.41, 31.25, 31.23],
            'Y': [30.11, 30.06, 30.03]
        })
        
        # Create sample existing roads data
        self.existing_roads = pd.DataFrame({
            'FromID': ['1', '1', '2', '3', 'F1', 'F2'],
            'ToID': ['3', '4', '3', 'F2', '2', '3'],
            'Distance(km)': [8.5, 22.8, 5.9, 2.5, 9.2, 2.5],
            'Current Capacity(vehicles/hour)': [3000, 3800, 2800, 2000, 3200, 2000],
            'Condition(1-10)': [7, 9, 8, 7, 8, 7]
        })
        
        # Create sample traffic flow data
        self.traffic_flow = pd.DataFrame({
            'FromID': ['1', '1', '2', '3', 'F1', 'F2'],
            'ToID': ['3', '4', '3', 'F2', '2', '3'],
            'Morning Peak(veh/h)': [2800, 3600, 2700, 1900, 3000, 1900],
            'Afternoon(veh/h)': [1500, 1800, 1400, 1600, 2000, 1600],
            'Evening Peak(veh/h)': [2600, 3300, 2500, 1800, 2800, 1800],
            'Night(veh/h)': [800, 750, 700, 900, 1100, 900]
        })
        
        # Create sample routes
        self.optimal_route = ['1', '3', 'F2', 'F1']
        self.alternative_route = ['1', '4', '2', 'F1']
    
    def test_plot_network_graph(self):
        """Test network graph plotting"""
        # Create a graph
        import networkx as nx
        G = nx.Graph()
        
        # Add nodes
        for _, row in self.neighborhoods.iterrows():
            G.add_node(row['ID'], 
                      name=row['Name'], 
                      population=row['Population'],
                      pos=(row['X'], row['Y']),
                      node_type='neighborhood')
        
        for _, row in self.facilities.iterrows():
            G.add_node(row['ID'], 
                      name=row['Name'], 
                      type=row['Type'],
                      pos=(row['X'], row['Y']),
                      node_type='facility')
        
        # Add edges
        for _, row in self.existing_roads.iterrows():
            G.add_edge(row['FromID'], row['ToID'], 
                      distance=row['Distance(km)'],
                      capacity=row['Current Capacity(vehicles/hour)'],
                      condition=row['Condition(1-10)'])
        
        # Test plot_network_graph
        fig = plot_network_graph(G)
        
        # Check that a figure is returned
        self.assertIsNotNone(fig)
        
        # Check that the figure has the expected data traces
        self.assertGreaterEqual(len(fig.data), 2)  # At least edges and nodes
    
    def test_plot_map(self):
        """Test map plotting"""
        # Test plot_map
        map_obj = plot_map(
            self.neighborhoods,
            self.facilities,
            self.existing_roads
        )
        
        # Check that a map is returned
        self.assertIsNotNone(map_obj)
        
        # Check that the map has the expected layers
        self.assertGreaterEqual(len(map_obj._children), 1)
    
    def test_plot_map_with_traffic(self):
        """Test map plotting with traffic visualization"""
        # Test plot_map with traffic
        map_obj = plot_map(
            self.neighborhoods,
            self.facilities,
            self.existing_roads,
            show_traffic=True,
            traffic_data=self.traffic_flow
        )
        
        # Check that a map is returned
        self.assertIsNotNone(map_obj)
    
    def test_plot_map_with_new_roads(self):
        """Test map plotting with new roads"""
        # Test plot_map with new roads
        new_roads = [('1', 'F1'), ('2', 'F3')]
        
        map_obj = plot_map(
            self.neighborhoods,
            self.facilities,
            self.existing_roads,
            new_roads=new_roads
        )
        
        # Check that a map is returned
        self.assertIsNotNone(map_obj)
    
    def test_plot_traffic_comparison(self):
        """Test traffic comparison plotting"""
        # Test plot_traffic_comparison
        fig = plot_traffic_comparison(
            self.traffic_flow,
            self.optimal_route,
            self.alternative_route,
            "Morning Peak(veh/h)"
        )
        
        # Check that a figure is returned
        self.assertIsNotNone(fig)
    
    def test_plot_route_visualization(self):
        """Test route visualization plotting"""
        # Test plot_route_visualization
        map_obj = plot_route_visualization(
            self.neighborhoods,
            self.facilities,
            self.existing_roads,
            self.optimal_route,
            self.alternative_route
        )
        
        # Check that a map is returned
        self.assertIsNotNone(map_obj)

if __name__ == '__main__':
    unittest.main()
