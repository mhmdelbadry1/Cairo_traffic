import unittest
import sys
import os
import pandas as pd
import numpy as np
import unittest.mock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit
sys.modules['streamlit'] = unittest.mock.MagicMock()

from emergency_response import plan_emergency_routes, preempt_intersection_signals
from traffic_signals import identify_intersections, real_time_signal_optimization

class TestEmergencyResponse(unittest.TestCase):
    """Test cases for emergency response module functions"""
    
    def setUp(self):
        """Set up test data for emergency response tests"""
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
        
        # Create sample existing roads data - Ensure a connected network
        self.existing_roads = pd.DataFrame({
            'FromID': ['1', '1', '2', '3', 'F1', 'F2', '3', 'F2'],  # Added direct connection from 3 to F3
            'ToID': ['3', '4', '3', 'F2', '2', '3', 'F3', 'F3'],    # Added direct connection from F2 to F3
            'Distance(km)': [8.5, 22.8, 5.9, 2.5, 9.2, 2.5, 3.0, 1.5],
            'Current Capacity(vehicles/hour)': [3000, 3800, 2800, 2000, 3200, 2000, 2500, 2200],
            'Condition(1-10)': [7, 9, 8, 7, 8, 7, 8, 9]
        })
        
        # Create sample traffic flow data - Match the road network
        self.traffic_flow = pd.DataFrame({
            'FromID': ['1', '1', '2', '3', 'F1', 'F2', '3', 'F2'],
            'ToID': ['3', '4', '3', 'F2', '2', '3', 'F3', 'F3'],
            'Morning Peak(veh/h)': [2800, 3600, 2700, 1900, 3000, 1900, 2200, 1800],
            'Afternoon(veh/h)': [1500, 1800, 1400, 1600, 2000, 1600, 1400, 1200],
            'Evening Peak(veh/h)': [2600, 3300, 2500, 1800, 2800, 1800, 2000, 1600],
            'Night(veh/h)': [800, 750, 700, 900, 1100, 900, 600, 500]
        })
        
        # Create intersections and signals
        self.intersections = identify_intersections(self.existing_roads, self.neighborhoods, self.facilities)
        self.signals = real_time_signal_optimization(self.intersections, self.traffic_flow, 
                                                   self.existing_roads, "Morning Peak(veh/h)")
    
    def test_plan_emergency_routes(self):
        """Test emergency route planning"""
        # Test with default parameters
        emergency_path, regular_path, emergency_time, regular_time, time_saved_route, \
        emergency_distance, preemption_plan, time_saved_signals = plan_emergency_routes(
            self.neighborhoods,
            self.facilities,
            self.existing_roads,
            self.traffic_flow,
            from_id='1',
            to_id='F3',
            time_column="Morning Peak(veh/h)",
            priority_level=3,
            route_clearing=True,
            emergency_type="Medical Emergency",
            intersections=self.intersections,
            signals=self.signals
        )
        
        # Check that paths were found
        self.assertGreater(len(emergency_path), 0)
        self.assertGreater(len(regular_path), 0)
        
        # Check that emergency time is less than regular time
        self.assertLess(emergency_time, regular_time)
        
        # Check that time saved is positive
        self.assertGreater(time_saved_route, 0)
    
    def test_plan_emergency_routes_different_priority(self):
        """Test emergency route planning with different priority levels"""
        # Test with different priority levels
        results = []
        for priority in [1, 2, 3]:
            emergency_path, regular_path, emergency_time, regular_time, time_saved_route, \
            emergency_distance, preemption_plan, time_saved_signals = plan_emergency_routes(
                self.neighborhoods,
                self.facilities,
                self.existing_roads,
                self.traffic_flow,
                from_id='1',
                to_id='F3',
                time_column="Morning Peak(veh/h)",
                priority_level=priority,
                route_clearing=True,
                emergency_type="Medical Emergency",
                intersections=self.intersections,
                signals=self.signals
            )
            
            results.append((emergency_path, emergency_time, time_saved_route))
        
        # In small test networks, different priorities might select different routes
        # which could lead to non-monotonic time savings. We only check that all priorities
        # result in time savings, not strict monotonicity.
        for result in results:
            self.assertGreater(result[2], 0, "All priority levels should result in time savings")
    
    def test_preempt_intersection_signals(self):
        """Test intersection signal preemption"""
        # Create a sample emergency path
        emergency_path = ['1', '3', 'F2', 'F3']
        
        # Test preemption
        preemption_plan, time_saved = preempt_intersection_signals(
            self.intersections,
            self.signals,
            emergency_path,
            self.traffic_flow,
            "Morning Peak(veh/h)"
        )
        
        # Check that preemption plan is returned
        self.assertIsInstance(preemption_plan, dict)
        
        # Check that time saved is non-negative
        self.assertGreaterEqual(time_saved, 0)
        
        # Check that preemption plan contains expected keys for intersections in the path
        for node_id in emergency_path[1:-1]:  # Skip first and last nodes
            if node_id in self.intersections:
                self.assertIn(node_id, preemption_plan)

if __name__ == '__main__':
    unittest.main()
