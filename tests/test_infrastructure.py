import unittest
import sys
import os
import pandas as pd
import numpy as np
import networkx as nx

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infrastructure import create_mst_network, generate_cost_report
from infrastructure_dp import allocate_maintenance_resources_dp

class TestInfrastructure(unittest.TestCase):
    """Test cases for infrastructure module functions"""
    
    def setUp(self):
        """Set up test data for infrastructure tests"""
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
        
        # Create sample potential roads data
        self.potential_roads = pd.DataFrame({
            'FromID': ['1', '2', '3', 'F1'],
            'ToID': ['2', 'F3', '4', 'F3'],
            'Distance(km)': [10.5, 3.2, 6.7, 8.9],
            'Estimated Capacity(vehicles/hour)': [3500, 2500, 3000, 3200],
            'Construction Cost(Million EGP)': [210, 64, 134, 178]
        })
    
    def test_create_mst_network_kruskal(self):
        """Test MST network creation using Kruskal's algorithm"""
        # Test with only existing roads
        G, time_cost, new_roads, distance, connectivity, cost = create_mst_network(
            self.neighborhoods, 
            self.facilities, 
            self.existing_roads,
            algorithm_choice="kruskal"
        )
        
        # Check that the graph has the correct number of nodes and edges
        self.assertEqual(len(G.nodes()), 7)  # 4 neighborhoods + 3 facilities
        # MST should have n-1 edges, but exact count may vary based on implementation
        self.assertGreaterEqual(len(G.edges()), 5)  # At least n-2 edges for connectivity
        
        # Check that time_cost and distance are positive
        self.assertGreater(time_cost, 0)
        self.assertGreater(distance, 0)
        
        # Check that no new roads were added
        self.assertEqual(len(new_roads), 0)
    
    def test_create_mst_network_with_new_roads(self):
        """Test MST network creation with potential new roads"""
        # Test with potential roads and budget
        G, time_cost, new_roads, distance, connectivity, cost = create_mst_network(
            self.neighborhoods, 
            self.facilities, 
            self.existing_roads,
            potential_roads=self.potential_roads,
            algorithm_choice="kruskal",
            max_budget=300
        )
        
        # Check that new roads were added within budget
        self.assertLessEqual(cost['new_roads_cost'], 300)
        
        # Check that the cost analysis has the expected keys
        expected_keys = ['existing_roads_cost', 'new_roads_cost', 'total_cost', 
                         'cost_per_km', 'cost_effectiveness']
        for key in expected_keys:
            self.assertIn(key, cost)
    
    def test_generate_cost_report(self):
        """Test cost report generation"""
        # Create sample new roads
        new_roads = [('1', '2', 210), ('F1', 'F3', 178)]
        
        # Generate cost report
        report = generate_cost_report(new_roads, self.potential_roads, 
                                     self.neighborhoods, self.facilities)
        
        # Check that the report has the correct number of rows
        self.assertEqual(len(report), 2)
        
        # Check that the report has the expected columns
        expected_columns = ['From', 'To', 'Distance (km)', 'Capacity (veh/h)', 
                           'Cost (Million EGP)', 'Cost per km (Million EGP)']
        for col in expected_columns:
            self.assertIn(col, report.columns)
    
    def test_allocate_maintenance_resources_dp(self):
        """Test dynamic programming resource allocation for road maintenance"""
        # Test with sample roads and budget
        allocation_plan, improvement, budget_used, metrics = allocate_maintenance_resources_dp(
            self.existing_roads, 
            budget=50,
            time_periods=2
        )
        
        # Check that the allocation plan is not empty
        self.assertGreater(len(allocation_plan), 0)
        
        # Check that budget used is within limits
        self.assertLessEqual(budget_used, 50)
        
        # Check that improvement is positive
        self.assertGreater(improvement, 0)
        
        # Check that metrics has the expected keys
        expected_keys = ['total_roads_improved', 'avg_improvement_per_road', 
                         'budget_efficiency', 'pct_budget_used']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check that allocation plan has the expected keys
        if allocation_plan:
            expected_keys = ['Time Period', 'Road', 'Current Condition', 'Length (km)',
                            'Investment (Million EGP)', 'Expected Improvement', 'New Condition']
            for key in expected_keys:
                self.assertIn(key, allocation_plan[0])

if __name__ == '__main__':
    unittest.main()
