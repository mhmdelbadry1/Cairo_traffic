import folium
import networkx as nx
from collections import defaultdict

def optimize_traffic_flow(neighborhoods, facilities, existing_roads, traffic_flow, 
                         from_id, to_id, time_column="Morning Peak(veh/h)",
                         consider_traffic=True, congestion_factor=1.5, 
                         consider_road_quality=True, max_alternatives=3, diversity_threshold=0.3):
    """
    Optimize traffic flow using Dijkstra's algorithm with time-dependent modifications
    and enhanced alternative route generation with proper G_alt handling.
    
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
        max_alternatives (int): Maximum number of alternative paths to return
        diversity_threshold (float): Minimum fraction of edges that must differ from optimal path
        
    Returns:
        tuple: (optimal_path, optimal_distance, optimal_time, 
                alternative_paths, alternative_distances, alternative_times, error_message)
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
        
        # Generate alternative paths
        alternative_paths = []
        alternative_distances = []
        alternative_times = []
        optimal_edges = set((min(optimal_path[i], optimal_path[i+1]), max(optimal_path[i], optimal_path[i+1])) 
                           for i in range(len(optimal_path)-1))
        excluded_edges = set()  # Track edges excluded across iterations
        
        for i in range(max_alternatives):
            # Create a fresh copy of the graph for each alternative path
            G_alt = G.copy()
            # Remove edges from the optimal path one at a time, ensuring diversity
            if i < len(optimal_path) - 1:
                u, v = optimal_path[i], optimal_path[i+1]
                edge = (min(u, v), max(u, v))
                if edge not in excluded_edges:
                    G_alt.remove_edge(u, v)
                    excluded_edges.add(edge)
            
            try:
                # Find a new path in the modified graph
                alt_path = nx.shortest_path(G_alt, source=from_id, target=to_id, weight='time')
                alt_edges = set((min(alt_path[j], alt_path[j+1]), max(alt_path[j], alt_path[j+1])) 
                               for j in range(len(alt_path)-1))
                
                # Check diversity: ensure the alternative path differs sufficiently
                common_edges = optimal_edges.intersection(alt_edges)
                diversity = 1.0 - (len(common_edges) / max(len(optimal_edges), len(alt_edges)))
                if (diversity >= diversity_threshold or not alternative_paths) and alt_path not in alternative_paths:
                    alt_distance = sum(G.get_edge_data(alt_path[j], alt_path[j+1])['distance'] 
                                      for j in range(len(alt_path)-1))
                    alt_time = sum(G.get_edge_data(alt_path[j], alt_path[j+1])['time'] 
                                  for j in range(len(alt_path)-1))
                    alternative_paths.append(alt_path)
                    alternative_distances.append(alt_distance)
                    alternative_times.append(alt_time)
            
            except nx.NetworkXNoPath:
                continue
        
        return (optimal_path, optimal_distance, optimal_time, 
                alternative_paths, alternative_distances, alternative_times, None)
    
    except nx.NetworkXNoPath:
        return [], 0, 0, [], [], [], "No path exists between the selected origin and destination."

def plot_map(neighborhoods, facilities, existing_roads, optimal_path=None, 
             alternative_paths=None, optimal_distance=None, optimal_time=None, 
             alternative_distances=None, alternative_times=None, new_roads=None, 
             show_traffic=False, traffic_data=None, highlight_emergency=False, 
             highlight_transit=False, transit_stops=None):
    """
    Plot a map with neighborhoods, facilities, roads, and optional traffic visualization,
    highlighting optimal and alternative paths.
    
    Args:
        neighborhoods (DataFrame): Neighborhood data
        facilities (DataFrame): Facility data
        existing_roads (DataFrame): Existing roads data
        optimal_path (list): Optimal path nodes from optimize_traffic_flow
        alternative_paths (list): List of alternative path nodes
        optimal_distance (float): Total distance of optimal path
        optimal_time (float): Total travel time of optimal path
        alternative_distances (list): List of distances for alternative paths
        alternative_times (list): List of travel times for alternative paths
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
    
    # Add optimal path
    if optimal_path and optimal_distance is not None and optimal_time is not None:
        path_locations = [pos_lookup.get(node) for node in optimal_path if pos_lookup.get(node)]
        if len(path_locations) == len(optimal_path):
            folium.PolyLine(
                locations=path_locations,
                color='blue',
                weight=5,
                opacity=0.8,
                popup=f"Optimal Path<br>Distance: {optimal_distance:.1f} km<br>Time: {optimal_time:.1f} min"
            ).add_to(m)
    
    # Add alternative paths
    if alternative_paths and alternative_distances and alternative_times:
        colors = ['green', 'yellow', 'red']  # Distinct colors for up to 3 alternative paths
        for i, (path, dist, time) in enumerate(zip(alternative_paths, alternative_distances, alternative_times)):
            path_locations = [pos_lookup.get(node) for node in path if pos_lookup.get(node)]
            if len(path_locations) == len(path):
                folium.PolyLine(
                    locations=path_locations,
                    color=colors[i % len(colors)],
                    weight=4,
                    opacity=0.7,
                    dash_array='5, 5',
                    popup=f"Alternative Path {i+1}<br>Distance: {dist:.1f} km<br>Time: {time:.1f} min"
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