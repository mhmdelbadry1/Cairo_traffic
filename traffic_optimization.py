import folium
import networkx as nx
import numpy as np
from collections import defaultdict

def optimize_traffic_flow(neighborhoods, facilities, existing_roads, traffic_flow, 
                         from_id, to_id, time_column="Morning Peak(veh/h)",
                         consider_traffic=True, congestion_factor=1.5, 
                         consider_road_quality=True):
    """
    Optimize traffic flow using Dijkstra's algorithm with time-dependent modifications.
    
    Args:
        neighborhoods (DataFrame): Neighborhoods and districts data
        facilities (DataFrame): Important facilities data
        existing_roads (DataFrame): Existing roads data
        traffic_flow (DataFrame): Traffic flow data for different times of day
        from_id (str): Origin node ID
        to_id (str): Destination node ID
        time_column (str): Column name for traffic time period to use
        consider_traffic (bool): Whether to consider traffic in route calculations
        congestion_factor (float): Factor for how much traffic affects travel time
        consider_road_quality (bool): Whether to consider road quality
        
    Returns:
        tuple: (optimal_path, optimal_distance, optimal_time, 
                alternative_path, alternative_distance, alternative_time, error_message)
    """
    G = nx.Graph()
    
    # Add nodes
    for _, row in neighborhoods.iterrows():
        G.add_node(row['ID'], name=row['Name'], population=row['Population'],
                  pos=(row['X'], row['Y']), node_type='neighborhood')
    
    for _, row in facilities.iterrows():
        G.add_node(row['ID'], name=row['Name'], type=row['Type'],
                  pos=(row['X'], row['Y']), node_type='facility')
    
    # Create traffic dictionary
    traffic_dict = {}
    for _, row in traffic_flow.iterrows():
        key = (row['FromID'], row['ToID'])
        reverse_key = (row['ToID'], row['FromID'])
        traffic_value = row[time_column]
        traffic_dict[key] = traffic_value
        traffic_dict[reverse_key] = traffic_value
    
    # Add edges
    for _, road in existing_roads.iterrows():
        condition_factor = 0.5 + (road['Condition(1-10)'] / 20) if consider_road_quality else 1.0
        base_speed = 60 * condition_factor
        base_time = (road['Distance(km)'] / base_speed) * 60
        
        traffic_adjustment = 1.0
        if consider_traffic:
            road_key = (road['FromID'], road['ToID'])
            reverse_key = (road['ToID'], road['FromID'])
            traffic_volume = traffic_dict.get(road_key, traffic_dict.get(reverse_key, 0.5 * road['Current Capacity(vehicles/hour)']))
            congestion_ratio = min(traffic_volume / road['Current Capacity(vehicles/hour)'], 1.0)
            traffic_adjustment = 1.0 + (congestion_ratio * 0.5) if congestion_ratio < 0.7 else 1.0 + (congestion_factor * (congestion_ratio ** 2))
        
        travel_time = base_time * traffic_adjustment
        
        G.add_edge(
            road['FromID'], road['ToID'], 
            distance=road['Distance(km)'],
            capacity=road['Current Capacity(vehicles/hour)'],
            condition=road['Condition(1-10)'],
            traffic=traffic_dict.get((road['FromID'], road['ToID']), 0),
            base_time=base_time,
            time=travel_time
        )
    
    try:
        # Optimal path
        optimal_path = nx.shortest_path(G, source=from_id, target=to_id, weight='time')
        optimal_distance = sum(G.get_edge_data(optimal_path[i], optimal_path[i+1])['distance'] 
                             for i in range(len(optimal_path)-1))
        optimal_time = sum(G.get_edge_data(optimal_path[i], optimal_path[i+1])['time'] 
                          for i in range(len(optimal_path)-1))
        
        # Alternative path: Remove one edge from optimal path to force a different route
        G_alt = G.copy()
        if len(optimal_path) > 2:
            # Remove the first edge after the start node
            u, v = optimal_path[1], optimal_path[2]
            G_alt.remove_edge(u, v)
        
        try:
            alternative_path = nx.shortest_path(G_alt, source=from_id, target=to_id, weight='time')
            alternative_distance = sum(G.get_edge_data(alternative_path[i], alternative_path[i+1])['distance'] 
                                     for i in range(len(alternative_path)-1))
            alternative_time = sum(G.get_edge_data(alternative_path[i], alternative_path[i+1])['time'] 
                                  for i in range(len(alternative_path)-1))
        except nx.NetworkXNoPath:
            alternative_path = []
            alternative_distance = 0
            alternative_time = 0
        
        return optimal_path, optimal_distance, optimal_time, alternative_path, alternative_distance, alternative_time, None
    
    except nx.NetworkXNoPath:
        return [], 0, 0, [], 0, 0, "No path exists between the selected origin and destination."

def plot_map(neighborhoods, facilities, existing_roads, new_roads=None, show_traffic=False, 
             traffic_data=None, highlight_emergency=False, highlight_transit=False, transit_stops=None):
    """
    Plot a map with neighborhoods, facilities, roads, and optional traffic visualization.
    
    Args:
        neighborhoods (DataFrame): Neighborhood data
        facilities (DataFrame): Facility data
        existing_roads (DataFrame): Existing roads data
        new_roads (list): List of new road tuples (from_id, to_id)
        show_traffic (bool): Whether to color roads based on traffic levels
        traffic_data (DataFrame): Traffic flow data
        highlight_emergency (bool): Highlight emergency facilities
        highlight_transit (bool): Highlight transit stops
        transit_stops (list): List of transit stop IDs
    
    Returns:
        folium.Map: Map object with all elements
    """
    # Initialize map centered on Cairo
    m = folium.Map(location=[30.0444, 31.2357], zoom_start=11)
    
    # Add neighborhoods
    for _, row in neighborhoods.iterrows():
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=5,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['Name']} (ID: {row['ID']})<br>Population: {row['Population']:,}"
        ).add_to(m)
    
    # Add facilities
    for _, row in facilities.iterrows():
        color = 'red' if highlight_emergency and row['Type'] in ['Hospital', 'Fire Station', 'Police Station'] else 'green'
        folium.Marker(
            location=[row['Y'], row['X']],
            popup=f"{row['Name']} (ID: {row['ID']})<br>Type: {row['Type']}",
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    # Create position lookup
    pos_lookup = {}
    for _, row in neighborhoods.iterrows():
        pos_lookup[row['ID']] = (row['Y'], row['X'])
    for _, row in facilities.iterrows():
        pos_lookup[row['ID']] = (row['Y'], row['X'])
    
    # Add existing roads
    if show_traffic and traffic_data is not None:
        traffic_dict = {}
        for _, row in traffic_data.iterrows():
            key = (row['FromID'], row['ToID'])
            reverse_key = (row['ToID'], row['FromID'])
            traffic_dict[key] = row
            traffic_dict[reverse_key] = row
    
    for _, road in existing_roads.iterrows():
        start_pos = pos_lookup.get(road['FromID'])
        end_pos = pos_lookup.get(road['ToID'])
        if start_pos and end_pos:
            color = 'gray'
            weight = 2
            if show_traffic and traffic_data is not None:
                road_key = (road['FromID'], road['ToID'])
                traffic_row = traffic_dict.get(road_key)
                if traffic_row:
                    congestion_ratio = traffic_row['Morning Peak(veh/h)'] / road['Current Capacity(vehicles/hour)']
                    if congestion_ratio < 0.5:
                        color = 'green'  # Low traffic
                    elif congestion_ratio < 0.8:
                        color = 'yellow'  # Moderate traffic
                    else:
                        color = 'red'  # Heavy traffic
                    weight = 4  # Thicker lines for traffic visualization
            folium.PolyLine(
                locations=[start_pos, end_pos],
                color=color,
                weight=weight,
                popup=f"Road: {road['FromID']} to {road['ToID']}<br>Distance: {road['Distance(km)']:.1f} km"
            ).add_to(m)
    
    # Add new roads
    if new_roads:
        for from_id, to_id in new_roads:
            start_pos = pos_lookup.get(from_id)
            end_pos = pos_lookup.get(to_id)
            if start_pos and end_pos:
                folium.PolyLine(
                    locations=[start_pos, end_pos],
                    color='purple',
                    weight=3,
                    dash_array='5, 5',
                    popup=f"New Road: {from_id} to {to_id}"
                ).add_to(m)
    
    # Highlight transit stops
    if highlight_transit and transit_stops:
        for stop_id in transit_stops:
            pos = pos_lookup.get(stop_id)
            if pos:
                folium.CircleMarker(
                    location=pos,
                    radius=7,
                    color='orange',
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"Transit Stop: {stop_id}"
                ).add_to(m)
    
    return m