import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data

class TestDataLoader(unittest.TestCase):
    """Test cases for data loader module functions"""
    
    def test_load_data(self):
        """Test data loading functionality"""
        # Test load_data function
        neighborhoods, facilities, existing_roads, potential_roads, traffic_flow, metro_lines, bus_routes, transit_demand = load_data()
        
        # Check that all dataframes are returned
        self.assertIsInstance(neighborhoods, pd.DataFrame)
        self.assertIsInstance(facilities, pd.DataFrame)
        self.assertIsInstance(existing_roads, pd.DataFrame)
        self.assertIsInstance(potential_roads, pd.DataFrame)
        self.assertIsInstance(traffic_flow, pd.DataFrame)
        self.assertIsInstance(metro_lines, pd.DataFrame)
        self.assertIsInstance(bus_routes, pd.DataFrame)
        self.assertIsInstance(transit_demand, pd.DataFrame)
        
        # Check that dataframes have the expected columns
        self.assertIn('ID', neighborhoods.columns)
        self.assertIn('Name', neighborhoods.columns)
        self.assertIn('Population', neighborhoods.columns)
        self.assertIn('Type', neighborhoods.columns)
        self.assertIn('X', neighborhoods.columns)
        self.assertIn('Y', neighborhoods.columns)
        
        self.assertIn('ID', facilities.columns)
        self.assertIn('Name', facilities.columns)
        self.assertIn('Type', facilities.columns)
        self.assertIn('X', facilities.columns)
        self.assertIn('Y', facilities.columns)
        
        self.assertIn('FromID', existing_roads.columns)
        self.assertIn('ToID', existing_roads.columns)
        self.assertIn('Distance(km)', existing_roads.columns)
        self.assertIn('Current Capacity(vehicles/hour)', existing_roads.columns)
        self.assertIn('Condition(1-10)', existing_roads.columns)
        
        self.assertIn('FromID', potential_roads.columns)
        self.assertIn('ToID', potential_roads.columns)
        self.assertIn('Distance(km)', potential_roads.columns)
        self.assertIn('Estimated Capacity(vehicles/hour)', potential_roads.columns)
        self.assertIn('Construction Cost(Million EGP)', potential_roads.columns)
        
        self.assertIn('FromID', traffic_flow.columns)
        self.assertIn('ToID', traffic_flow.columns)
        self.assertIn('Morning Peak(veh/h)', traffic_flow.columns)
        self.assertIn('Afternoon(veh/h)', traffic_flow.columns)
        self.assertIn('Evening Peak(veh/h)', traffic_flow.columns)
        self.assertIn('Night(veh/h)', traffic_flow.columns)
        
        self.assertIn('LineID', metro_lines.columns)
        self.assertIn('Name', metro_lines.columns)
        self.assertIn('Stations(comma-separated IDs)', metro_lines.columns)
        self.assertIn('Daily Passengers', metro_lines.columns)
        
        self.assertIn('RouteID', bus_routes.columns)
        self.assertIn('Stops(comma-separated IDs)', bus_routes.columns)
        self.assertIn('Buses Assigned', bus_routes.columns)
        self.assertIn('Daily Passengers', bus_routes.columns)
        
        self.assertIn('FromID', transit_demand.columns)
        self.assertIn('ToID', transit_demand.columns)
        self.assertIn('Daily Passengers', transit_demand.columns)
        
        # Check that dataframes have data
        self.assertGreater(len(neighborhoods), 0)
        self.assertGreater(len(facilities), 0)
        self.assertGreater(len(existing_roads), 0)
        self.assertGreater(len(potential_roads), 0)
        self.assertGreater(len(traffic_flow), 0)
        self.assertGreater(len(metro_lines), 0)
        self.assertGreater(len(bus_routes), 0)
        self.assertGreater(len(transit_demand), 0)
        
        # Check data types
        self.assertEqual(neighborhoods['Population'].dtype, np.int64)
        self.assertEqual(neighborhoods['X'].dtype, np.float64)
        self.assertEqual(neighborhoods['Y'].dtype, np.float64)
        
        self.assertEqual(facilities['X'].dtype, np.float64)
        self.assertEqual(facilities['Y'].dtype, np.float64)
        
        self.assertEqual(existing_roads['Distance(km)'].dtype, np.float64)
        self.assertEqual(existing_roads['Current Capacity(vehicles/hour)'].dtype, np.int64)
        self.assertEqual(existing_roads['Condition(1-10)'].dtype, np.int64)
        
        self.assertEqual(potential_roads['Distance(km)'].dtype, np.float64)
        self.assertEqual(potential_roads['Estimated Capacity(vehicles/hour)'].dtype, np.int64)
        self.assertEqual(potential_roads['Construction Cost(Million EGP)'].dtype, np.int64)
        
        self.assertEqual(traffic_flow['Morning Peak(veh/h)'].dtype, np.int64)
        self.assertEqual(traffic_flow['Afternoon(veh/h)'].dtype, np.int64)
        self.assertEqual(traffic_flow['Evening Peak(veh/h)'].dtype, np.int64)
        self.assertEqual(traffic_flow['Night(veh/h)'].dtype, np.int64)
        
        self.assertEqual(metro_lines['Daily Passengers'].dtype, np.int64)
        
        self.assertEqual(bus_routes['Buses Assigned'].dtype, np.int64)
        self.assertEqual(bus_routes['Daily Passengers'].dtype, np.int64)
        
        self.assertEqual(transit_demand['Daily Passengers'].dtype, np.int64)

if __name__ == '__main__':
    unittest.main()
