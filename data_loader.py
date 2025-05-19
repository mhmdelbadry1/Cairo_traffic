import pandas as pd
import re

def load_data():
    """
    Load and parse Cairo traffic data with additional real-world routes and examples.
    
    Returns:
        tuple: A tuple containing DataFrames for neighborhoods, facilities, existing_roads, 
               potential_roads, traffic_flow, metro_lines, bus_routes, and transit_demand
    """
    # Geography Data - Neighborhoods and Districts (Expanded)
    neighborhoods_data = """
    1, Maadi, 250000, Residential, 31.25, 29.96
    2, Nasr City, 500000, Mixed, 31.34, 30.06
    3, Downtown Cairo, 100000, Business, 31.24, 30.04
    4, New Cairo, 300000, Residential, 31.47, 30.03
    5, Heliopolis, 200000, Mixed, 31.32, 30.09
    6, Zamalek, 50000, Residential, 31.22, 30.06
    7, 6th October City, 400000, Mixed, 30.98, 29.93
    8, Giza, 550000, Mixed, 31.21, 29.99
    9, Mohandessin, 180000, Business, 31.20, 30.05
    10, Dokki, 220000, Mixed, 31.21, 30.03
    11, Shubra, 450000, Residential, 31.24, 30.11
    12, Helwan, 350000, Industrial, 31.33, 29.85
    13, New Administrative Capital, 50000, Government, 31.80, 30.02
    14, Al Rehab, 120000, Residential, 31.49, 30.06
    15, Sheikh Zayed, 150000, Residential, 30.94, 30.01
    16, Imbaba, 300000, Residential, 31.21, 30.08
    17, Madinaty, 80000, Residential, 31.50, 30.09
    18, 15th of May City, 200000, Industrial, 31.36, 29.87
    19, Al Shorouk, 100000, Residential, 31.46, 30.12
    20, Obour City, 150000, Mixed, 31.40, 30.15
    21, Garden City, 60000, Business, 31.23, 30.03
    22, Agouza, 160000, Mixed, 31.21, 30.06
    23, Badr City, 70000, Residential, 31.72, 30.13
    24, Manial, 220000, Residential, 31.23, 30.02
    25, Sayeda Zeinab, 280000, Mixed, 31.24, 30.03
    """
    
    # Important Facilities (Expanded)
    facilities_data = """
    F1, Cairo International Airport, Airport, 31.41, 30.11
    F2, Ramses Railway Station, Transit Hub, 31.25, 30.06
    F3, Cairo University, Education, 31.21, 30.03
    F4, Al-Azhar University, Education, 31.26, 30.05
    F5, Egyptian Museum, Tourism, 31.23, 30.05
    F6, Cairo International Stadium, Sports, 31.30, 30.07
    F7, Smart Village, Business, 30.97, 30.07
    F8, Cairo Festival City, Commercial, 31.40, 30.03
    F9, Qasr El Aini Hospital, Medical, 31.23, 30.03
    F10, Maadi Military Hospital, Medical, 31.25, 29.95
    F11, Dar Al Fouad Hospital, Medical, 30.97, 29.94
    F12, Ain Shams University Hospital, Medical, 31.28, 30.07
    F13, Tahrir Square, Landmark, 31.24, 30.04
    F14, Giza Pyramids, Tourism, 31.13, 29.98
    F15, Mall of Egypt, Commercial, 30.97, 29.92
    F16, New Capital Train Station, Transit Hub, 31.80, 30.03
    F17, City Stars Mall, Commercial, 31.34, 30.07
    F18, Al Azhar Park, Tourism, 31.26, 30.06
    F19, Nile Corniche, Landmark, 31.23, 30.05
    F20, Sphinx International Airport, Airport, 30.92, 29.90
    """
    
    # Existing Roads (Expanded with Ring Road and more connections)
    existing_roads_data = """
    1, 3, 8.5, 3000, 7
    1, 8, 6.2, 2500, 6
    1, 24, 7.0, 2800, 6
    2, 3, 5.9, 2800, 8
    2, 5, 4.0, 3200, 9
    2, 17, 8.5, 3400, 8
    2, 19, 10.2, 3500, 7
    3, 5, 6.1, 3500, 7
    3, 6, 3.2, 2000, 8
    3, 9, 4.5, 2600, 6
    3, 10, 3.8, 2400, 7
    3, 21, 2.5, 2200, 8
    3, 22, 3.0, 2100, 7
    3, 25, 1.5, 2000, 6
    4, 2, 15.2, 3800, 9
    4, 14, 5.3, 3000, 10
    4, 17, 4.8, 3100, 9
    4, 19, 5.5, 3200, 8
    5, 11, 7.9, 3100, 7
    5, 20, 10.0, 3300, 8
    6, 9, 2.2, 1800, 8
    6, 22, 2.0, 1900, 7
    7, 8, 24.5, 3500, 8
    7, 15, 9.8, 3000, 9
    7, 16, 20.0, 3400, 7
    8, 10, 3.3, 2200, 7
    8, 12, 14.8, 2600, 5
    8, 16, 5.0, 2300, 6
    9, 10, 2.1, 1900, 7
    9, 22, 2.0, 2000, 8
    10, 11, 8.7, 2400, 6
    11, 16, 6.5, 2700, 7
    11, F2, 3.6, 2200, 7
    12, 1, 12.7, 2800, 6
    12, 18, 3.0, 2500, 5
    13, 4, 45.0, 4000, 10
    13, 14, 35.5, 3800, 9
    13, 23, 15.0, 3700, 8
    14, 17, 5.0, 3100, 9
    15, 7, 9.8, 3000, 9
    15, 16, 15.0, 3200, 8
    17, 19, 8.0, 3300, 8
    19, 20, 6.0, 3000, 7
    20, 23, 12.0, 3400, 7
    21, 24, 1.5, 2000, 8
    24, 25, 1.0, 1900, 7
    F1, 2, 9.2, 3200, 8
    F1, 5, 7.5, 3500, 9
    F1, 20, 12.5, 3600, 8
    F2, 3, 2.5, 2000, 7
    F2, 11, 3.0, 2100, 7
    F3, 8, 2.0, 2200, 8
    F3, 10, 1.0, 2000, 9
    F4, 3, 2.5, 2100, 7
    F5, 3, 1.0, 1800, 8
    F6, 2, 3.5, 2500, 7
    F7, 7, 8.0, 2800, 8
    F7, 15, 8.3, 2800, 8
    F8, 4, 6.1, 3000, 9
    F8, 2, 8.0, 3200, 8
    F9, 3, 1.0, 2000, 8
    F10, 1, 1.5, 2100, 7
    F11, 7, 5.0, 2600, 8
    F12, 2, 5.0, 2700, 7
    F13, 3, 0.5, 1800, 9
    F14, 8, 5.0, 2400, 7
    F15, 7, 6.0, 2700, 8
    F16, 13, 1.0, 2000, 9
    F17, 2, 2.0, 2500, 8
    F18, 3, 2.5, 2200, 7
    F19, 3, 1.0, 2000, 8
    F20, 7, 8.0, 2800, 8
    """
    
    # Potential New Roads (Expanded with Ring Road extensions)
    potential_roads_data = """
    1, 4, 22.8, 4000, 450
    1, 14, 25.3, 3800, 500
    1, 19, 30.0, 4200, 600
    2, 13, 48.2, 4500, 950
    2, 20, 15.0, 4000, 350
    3, 13, 56.7, 4500, 1100
    3, 16, 5.0, 3000, 150
    4, 23, 20.0, 3800, 400
    5, 4, 16.8, 3500, 320
    5, 19, 12.0, 3600, 300
    6, 8, 7.5, 2500, 150
    7, 13, 82.3, 4000, 1600
    7, 19, 35.0, 4200, 700
    8, 18, 15.0, 3400, 300
    9, 11, 6.9, 2800, 140
    10, F7, 27.4, 3200, 550
    11, 13, 62.1, 4200, 1250
    12, 14, 30.5, 3600, 610
    12, 23, 25.0, 3800, 500
    14, 5, 18.2, 3300, 360
    15, 9, 22.7, 3000, 450
    16, 19, 28.0, 4000, 550
    17, 20, 15.0, 3600, 350
    18, 23, 20.0, 3400, 400
    F1, 13, 40.2, 4000, 800
    F7, 9, 26.8, 3200, 540
    F14, 15, 25.0, 3500, 500
    F20, 15, 5.0, 3000, 200
    """
    
    # Traffic Flow Data (Expanded with new routes)
    traffic_flow_data = """
    1-3, 2800, 1500, 2600, 800
    1-8, 2200, 1200, 2100, 600
    1-24, 2500, 1300, 2300, 550
    2-3, 2700, 1400, 2500, 700
    2-5, 3000, 1600, 2800, 650
    2-17, 3200, 1700, 3000, 700
    2-19, 3400, 1800, 3100, 750
    3-5, 3200, 1700, 3100, 800
    3-6, 1800, 1400, 1900, 500
    3-9, 2400, 1300, 2200, 550
    3-10, 2300, 1200, 2100, 500
    3-21, 2000, 1100, 1900, 450
    3-22, 2100, 1200, 2000, 500
    3-25, 1900, 1000, 1800, 400
    4-2, 3600, 1800, 3300, 750
    4-14, 2800, 1600, 2600, 600
    4-17, 2900, 1500, 2700, 650
    4-19, 3000, 1600, 2800, 700
    5-11, 2900, 1500, 2700, 650
    5-20, 3100, 1600, 2900, 700
    6-9, 1700, 1300, 1800, 450
    6-22, 1800, 1400, 1900, 500
    7-8, 3200, 1700, 3000, 700
    7-15, 2800, 1500, 2600, 600
    7-16, 3300, 1800, 3100, 750
    8-10, 2000, 1100, 1900, 450
    8-12, 2400, 1300, 2200, 500
    8-16, 2100, 1200, 2000, 450
    9-10, 1800, 1200, 1700, 400
    9-22, 1900, 1300, 1800, 450
    10-11, 2200, 1300, 2100, 500
    11-16, 2500, 1400, 2300, 550
    11-F2, 2100, 1200, 2000, 450
    12-1, 2600, 1400, 2400, 550
    12-18, 2300, 1200, 2100, 500
    13-4, 3800, 2000, 3500, 800
    13-14, 3600, 1900, 3300, 750
    13-23, 3400, 1800, 3200, 700
    14-17, 2900, 1500, 2700, 650
    15-7, 2800, 1500, 2600, 600
    15-16, 3000, 1600, 2800, 650
    17-19, 3100, 1600, 2900, 700
    19-20, 2800, 1500, 2600, 600
    20-23, 3200, 1700, 3000, 650
    21-24, 1800, 1000, 1700, 400
    24-25, 1700, 900, 1600, 350
    F1-2, 3000, 2000, 2800, 1100
    F1-5, 3300, 2200, 3100, 1200
    F1-20, 3500, 2300, 3200, 1300
    F2-3, 1900, 1600, 1800, 900
    F2-11, 2000, 1700, 1900, 950
    F3-8, 2100, 1200, 2000, 500
    F3-10, 1900, 1100, 1800, 450
    F4-3, 2000, 1100, 1900, 500
    F5-3, 1700, 1000, 1600, 400
    F6-2, 2300, 1300, 2100, 550
    F7-7, 2600, 1500, 2400, 600
    F7-15, 2600, 1500, 2400, 550
    F8-4, 2800, 1600, 2600, 600
    F8-2, 2800, 1600, 2600, 600
    F9-3, 1800, 1000, 1700, 400
    F10-1, 1900, 1100, 1800, 450
    F11-7, 2400, 1300, 2200, 550
    F12-2, 2500, 1400, 2300, 600
    F13-3, 1600, 900, 1500, 350
    F14-8, 2200, 1200, 2000, 500
    F15-7, 2500, 1400, 2300, 600
    F16-13, 1800, 1000, 1700, 400
    F17-2, 2300, 1300, 2100, 550
    F18-3, 2000, 1100, 1900, 500
    F19-3, 1800, 1000, 1700, 400
    F20-7, 2600, 1500, 2400, 600
    """
    
    # Metro Lines Data (Expanded with new lines)
    metro_lines_data = """
    M1, Line 1 (Helwan-New Marg), "12,1,3,F2,11", 1500000
    M2, Line 2 (Shubra-Giza), "11,F2,3,10,8", 1200000
    M3, Line 3 (Airport-Imbaba), "F1,5,2,3,9", 800000
    M4, Line 4 (Nasr City-New Capital), "2,4,14,13", 600000
    M5, Line 5 (6th October-Imbaba), "7,15,16,8", 500000
    M6, Line 6 (Shorouk-Airport), "19,20,F1", 400000
    """
    
    # Bus Routes Data (Expanded with more routes)
    bus_routes_data = """
    B1, "1,3,6,9", 25, 35000
    B2, "7,15,8,10,3", 30, 42000
    B3, "2,5,F1", 20, 28000
    B4, "4,14,2,3", 22, 31000
    B5, "8,12,1", 18, 25000
    B6, "11,5,2", 24, 33000
    B7, "13,4,14", 15, 21000
    B8, "F7,15,7", 12, 17000
    B9, "1,8,10,9,6", 28, 39000
    B10, "F8,4,2,5", 20, 28000
    B11, "19,20,2,F1", 18, 26000
    B12, "7,16,8,F14", 22, 30000
    B13, "3,21,24,25", 15, 20000
    B14, "13,23,19,4", 20, 28000
    B15, "2,17,14,F8", 18, 25000
    B16, "F9,3,F5,F13", 12, 18000
    B17, "F11,7,F15,15", 15, 22000
    B18, "F17,2,5,11", 20, 27000
    B19, "F20,7,15,16", 14, 20000
    B20, "18,12,1,F10", 16, 23000
    """
    
    # Public Transportation Demand (Expanded with new pairs)
    transit_demand_data = """
    3, 5, 15000
    1, 3, 12000
    2, 3, 18000
    2, 17, 14000
    2, 19, 13000
    F2, 11, 25000
    F1, 3, 20000
    F1, 20, 15000
    7, 3, 14000
    7, 16, 12000
    4, 3, 16000
    4, 19, 11000
    8, 3, 22000
    8, 16, 13000
    3, 9, 13000
    3, 21, 10000
    3, 22, 11000
    3, 25, 12000
    5, 2, 17000
    5, 20, 14000
    11, 3, 24000
    11, 16, 15000
    12, 3, 11000
    12, 18, 9000
    1, 8, 9000
    1, 24, 8000
    7, F7, 18000
    7, F15, 15000
    4, F8, 12000
    13, 3, 8000
    13, 23, 10000
    14, 4, 7000
    14, 17, 6000
    15, 7, 11000
    15, F11, 13000
    17, 19, 10000
    19, 20, 9000
    20, 23, 8000
    21, 24, 7000
    24, 25, 6000
    F9, 3, 10000
    F10, 1, 8000
    F13, 3, 12000
    F14, 8, 11000
    F16, 13, 7000
    F17, 2, 13000
    F18, 3, 9000
    F19, 3, 10000
    F20, 7, 12000
    """
    
    # Process neighborhoods data
    neighborhoods_rows = [re.sub(r'\s+', ' ', line.strip()).split(', ') for line in neighborhoods_data.strip().split('\n')]
    neighborhoods = pd.DataFrame(neighborhoods_rows, columns=['ID', 'Name', 'Population', 'Type', 'X', 'Y'])
    neighborhoods['Population'] = neighborhoods['Population'].astype(int)
    neighborhoods['X'] = neighborhoods['X'].astype(float)
    neighborhoods['Y'] = neighborhoods['Y'].astype(float)
    
    # Process facilities data
    facilities_rows = [re.sub(r'\s+', ' ', line.strip()).split(', ') for line in facilities_data.strip().split('\n')]
    facilities = pd.DataFrame(facilities_rows, columns=['ID', 'Name', 'Type', 'X', 'Y'])
    facilities['X'] = facilities['X'].astype(float)
    facilities['Y'] = facilities['Y'].astype(float)
    
    # Process existing roads data
    existing_roads_rows = [re.sub(r'\s+', ' ', line.strip()).split(', ') for line in existing_roads_data.strip().split('\n')]
    existing_roads = pd.DataFrame(existing_roads_rows, columns=['FromID', 'ToID', 'Distance(km)', 'Current Capacity(vehicles/hour)', 'Condition(1-10)'])
    existing_roads['Distance(km)'] = existing_roads['Distance(km)'].astype(float)
    existing_roads['Current Capacity(vehicles/hour)'] = existing_roads['Current Capacity(vehicles/hour)'].astype(int)
    existing_roads['Condition(1-10)'] = existing_roads['Condition(1-10)'].astype(int)
    
    # Process potential roads data
    potential_roads_rows = [re.sub(r'\s+', ' ', line.strip()).split(', ') for line in potential_roads_data.strip().split('\n')]
    potential_roads = pd.DataFrame(potential_roads_rows, columns=['FromID', 'ToID', 'Distance(km)', 'Estimated Capacity(vehicles/hour)', 'Construction Cost(Million EGP)'])
    potential_roads['Distance(km)'] = potential_roads['Distance(km)'].astype(float)
    potential_roads['Estimated Capacity(vehicles/hour)'] = potential_roads['Estimated Capacity(vehicles/hour)'].astype(int)
    potential_roads['Construction Cost(Million EGP)'] = potential_roads['Construction Cost(Million EGP)'].astype(int)
    
    # Process traffic flow data
    traffic_flow_rows = []
    for line in traffic_flow_data.strip().split('\n'):
        parts = re.sub(r'\s+', ' ', line.strip()).split(', ')
        road_id_parts = parts[0].split('-')
        from_id, to_id = road_id_parts[0], road_id_parts[1]
        traffic_flow_rows.append([from_id, to_id] + parts[1:])
    
    traffic_flow = pd.DataFrame(traffic_flow_rows, columns=['FromID', 'ToID', 'Morning Peak(veh/h)', 'Afternoon(veh/h)', 'Evening Peak(veh/h)', 'Night(veh/h)'])
    traffic_flow['Morning Peak(veh/h)'] = traffic_flow['Morning Peak(veh/h)'].astype(int)
    traffic_flow['Afternoon(veh/h)'] = traffic_flow['Afternoon(veh/h)'].astype(int)
    traffic_flow['Evening Peak(veh/h)'] = traffic_flow['Evening Peak(veh/h)'].astype(int)
    traffic_flow['Night(veh/h)'] = traffic_flow['Night(veh/h)'].astype(int)
    
    # Process metro lines data
    metro_lines_rows = [re.sub(r'\s+', ' ', line.strip()).split(', ', 3) for line in metro_lines_data.strip().split('\n')]
    metro_lines = pd.DataFrame(metro_lines_rows, columns=['LineID', 'Name', 'Stations(comma-separated IDs)', 'Daily Passengers'])
    metro_lines['Daily Passengers'] = metro_lines['Daily Passengers'].astype(int)
    
    # Process bus routes data
    bus_routes_rows = [re.sub(r'\s+', ' ', line.strip()).split(', ', 3) for line in bus_routes_data.strip().split('\n')]
    bus_routes = pd.DataFrame(bus_routes_rows, columns=['RouteID', 'Stops(comma-separated IDs)', 'Buses Assigned', 'Daily Passengers'])
    bus_routes['Buses Assigned'] = bus_routes['Buses Assigned'].astype(int)
    bus_routes['Daily Passengers'] = bus_routes['Daily Passengers'].astype(int)
    
    # Process transit demand data
    transit_demand_rows = [re.sub(r'\s+', ' ', line.strip()).split(', ') for line in transit_demand_data.strip().split('\n')]
    transit_demand = pd.DataFrame(transit_demand_rows, columns=['FromID', 'ToID', 'Daily Passengers'])
    transit_demand['Daily Passengers'] = transit_demand['Daily Passengers'].astype(int)
    
    return neighborhoods, facilities, existing_roads, potential_roads, traffic_flow, metro_lines, bus_routes, transit_demand