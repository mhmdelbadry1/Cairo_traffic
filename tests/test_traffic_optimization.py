import unittest
import sys
import os
import pandas as pd
import numpy as np
import networkx as nx

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic_optimization import optimize_traffic_flow

class TestTrafficOptimization(unittest.TestCase):
    """Test cases for traffic optimization module functions"""
    
    def setUp(self):
        """Set up test data for traffic optimization tests"""
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
    
    def test_optimize_traffic_flow_basic(self):
        """Test basic traffic flow optimization"""
        # Test with default parameters
        optimal_path, optimal_distance, optimal_time, alternative_path, alt_distance, alt_time, error = optimize_traffic_flow(
            self.neighborhoods,
            self.facilities,
            self.existing_roads,
            self.traffic_flow,
            from_id='1',
            to_id='F1'
        )
        
        # Check that a path was found
        self.assertIsNone(error)
        self.assertGreater(len(optimal_path), 0)
        
        # Check that distance and time are positive
        self.assertGreater(optimal_distance, 0)
        self.assertGreater(optimal_time, 0)
    
    def test_optimize_traffic_flow_no_traffic(self):
        """Test traffic flow optimization without considering traffic"""
        # Test without considering traffic
        optimal_path, optimal_distance, optimal_time, alternative_path, alt_distance, alt_time, error = optimize_traffic_flow(
            self.neighborhoods,
            self.facilities,
            self.existing_roads,
            self.traffic_flow,
            from_id='1',
            to_id='F1',
            consider_traffic=False
        )
        
        # Check that a path was found
        self.assertIsNone(error)
        self.assertGreater(len(optimal_path), 0)
        
        # Check that distance and time are positive
        self.assertGreater(optimal_distance, 0)
        self.assertGreater(optimal_time, 0)
    
    def test_optimize_traffic_flow_different_time_periods(self):
        """Test traffic flow optimization with different time periods"""
        # Test with different time periods
        time_periods = ["Morning Peak(veh/h)", "Afternoon(veh/h)", "Evening Peak(veh/h)", "Night(veh/h)"]
        
        results = []
        for time_period in time_periods:
            optimal_path, optimal_distance, optimal_time, alternative_path, alt_distance, alt_time, error = optimize_traffic_flow(
                self.neighborhoods,
                self.facilities,
                self.existing_roads,
                self.traffic_flow,
                from_id='1',
                to_id='F1',
                time_column=time_period
            )
            
            # Check that a path was found
            self.assertIsNone(error)
            self.assertGreater(len(optimal_path), 0)
            
            results.append((optimal_path, optimal_time))
        
        # Check if at least one time period has a different optimal path or time
        # This verifies that time-dependent routing is working
        self.assertTrue(any(results[0][0] != result[0] or abs(results[0][1] - result[1]) > 0.1 
                           for result in results[1:]))
    
    def test_optimize_traffic_flow_nonexistent_path(self):
        """Test traffic flow optimization with nonexistent path"""
        # Create a disconnected graph by removing a critical edge
        disconnected_roads = self.existing_roads.iloc[1:].copy()
        
        # Test with disconnected graph
        optimal_path, optimal_distance, optimal_time, alternative_path, alt_distance, alt_time, error = optimize_traffic_flow(
            self.neighborhoods,
            self.facilities,
            disconnected_roads,
            self.traffic_flow,
            from_id='1',
            to_id='F1'
        )
        
        # Check that no path was found and error message is returned
        self.assertIsNotNone(error)
        self.assertEqual(len(optimal_path), 0)
        self.assertEqual(optimal_distance, 0)
        self.assertEqual(optimal_time, 0)
    
    def test_enhanced_alternate_route_handling(self):
        """Test enhanced alternate route handling during road closures"""
        # First get the optimal route
        optimal_path, optimal_distance, optimal_time, alternative_path, alt_distance, alt_time, error = optimize_traffic_flow(
            self.neighborhoods,
            self.facilities,
            self.existing_roads,
            self.traffic_flow,
            from_id='1',
            to_id='F1'
        )
        
        # Now simulate a road closure by removing one road from the optimal path
        if len(optimal_path) >= 3:
            closed_road_from = optimal_path[0]
            closed_road_to = optimal_path[1]
            
            # Create a copy of roads with the closure
            roads_with_closure = self.existing_roads.copy()
            closure_mask = ~((roads_with_closure['FromID'] == closed_road_from) & 
                           (roads_with_closure['ToID'] == closed_road_to) |
                           (roads_with_closure['FromID'] == closed_road_to) & 
                           (roads_with_closure['ToID'] == closed_road_from))
            roads_with_closure = roads_with_closure[closure_mask]
            
            # Test with road closure
            new_optimal_path, new_optimal_distance, new_optimal_time, new_alternative_path, new_alt_distance, new_alt_time, error = optimize_traffic_flow(
                self.neighborhoods,
                self.facilities,
                roads_with_closure,
                self.traffic_flow,
                from_id='1',
                to_id='F1'
            )
            
            # Check that a different path was found
            if error is None:
                self.assertNotEqual(optimal_path, new_optimal_path)
                self.assertGreater(new_optimal_time, optimal_time)  # Should be longer due to detour

if __name__ == '__main__':
    unittest.main()
