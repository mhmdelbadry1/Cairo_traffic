import folium
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster

def plot_network_graph(G):
    """
    Create a network graph visualization using Plotly.
    
    Args:
        G (NetworkX Graph): Graph containing the road network
        
    Returns:
        plotly.graph_objects.Figure: Interactive network plot
    """
    # Extract node positions for plotting
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get node types for coloring
    node_types = nx.get_node_attributes(G, 'node_type')
    
    # Create edges for plotting
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge text for tooltips
        edge_text.append(f"Distance: {edge[2].get('distance', 0):.1f} km<br>" +
                        f"Capacity: {edge[2].get('capacity', 0)} veh/h<br>" +
                        f"Condition: {edge[2].get('condition', 0)}/10")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='text',
        text=edge_text * 3,  # Repeat text for each segment (x0,x1,None)
        mode='lines')
    
    # Create node traces - separate for neighborhoods and facilities
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    facility_x = []
    facility_y = []
    facility_text = []
    facility_color = []
    
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_type = node_types.get(node[0], 'unknown')
        
        if node_type == 'neighborhood':
            node_x.append(x)
            node_y.append(y)
            population = node[1].get('population', 0)
            node_text.append(f"{node[1].get('name', node[0])}<br>Population: {population:,}")
            node_color.append(min(255, int(population / 5000)))
            node_size.append(min(20, 5 + population / 50000))
        else:  # facility
            facility_x.append(x)
            facility_y.append(y)
            facility_text.append(f"{node[1].get('name', node[0])}<br>Type: {node[1].get('type', 'Unknown')}")
            facility_type = node[1].get('type', 'Unknown')
            
            # Color based on facility type
            if 'Hospital' in facility_type or 'Medical' in facility_type:
                facility_color.append('red')
            elif 'Education' in facility_type:
                facility_color.append('blue')
            elif 'Transit' in facility_type or 'Airport' in facility_type:
                facility_color.append('purple')
            elif 'Commercial' in facility_type or 'Business' in facility_type:
                facility_color.append('orange')
            else:
                facility_color.append('green')
    
    # Neighborhood nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            colorscale='Viridis',
            size=node_size,
            line_width=2))
    
    # Facility nodes
    facility_trace = go.Scatter(
        x=facility_x, y=facility_y,
        mode='markers',
        hoverinfo='text',
        text=facility_text,
        marker=dict(
            color=facility_color,
            size=15,
            symbol='diamond',
            line_width=2))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace, facility_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    return fig

def plot_map(neighborhoods, facilities, existing_roads, new_roads=None, 
             show_traffic=False, traffic_data=None, highlight_emergency=False,
             highlight_transit=False, transit_stops=None):
    """
    Create an interactive map visualization using Folium.
    
    Args:
        neighborhoods (DataFrame): Neighborhoods and districts data
        facilities (DataFrame): Important facilities data
        existing_roads (DataFrame): Existing roads data
        new_roads (list): List of (from_id, to_id) tuples for recommended new roads
        show_traffic (bool): Whether to color-code roads by traffic level
        traffic_data (DataFrame): Traffic flow data if show_traffic is True
        highlight_emergency (bool): Whether to highlight emergency facilities
        highlight_transit (bool): Whether to highlight transit facilities
        transit_stops (list): List of transit stop IDs to highlight
        
    Returns:
        folium.Map: Interactive map with layers
    """
    # Calculate center of map
    center_lat = neighborhoods['Y'].mean()
    center_lon = neighborhoods['X'].mean()
    
    # Create map
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Create layers
    neighborhood_layer = folium.FeatureGroup(name="Neighborhoods")
    facility_layer = folium.FeatureGroup(name="Facilities")
    road_layer = folium.FeatureGroup(name="Existing Roads")
    
    if new_roads:
        new_road_layer = folium.FeatureGroup(name="Recommended New Roads")
    
    # Add neighborhoods
    for _, row in neighborhoods.iterrows():
        # Scale circle size by population
        radius = 100 + (row['Population'] / 10000)  # Base radius + population factor
        
        # Different colors for different types
        if row['Type'] == 'Residential':
            color = 'blue'
        elif row['Type'] == 'Business':
            color = 'orange'
        elif row['Type'] == 'Industrial':
            color = 'purple'
        elif row['Type'] == 'Government':
            color = 'darkred'
        else:  # Mixed
            color = 'green'
        
        # Popup content
        popup_text = f"""
        <b>{row['Name']}</b><br>
        Type: {row['Type']}<br>
        Population: {row['Population']:,}<br>
        ID: {row['ID']}
        """
        
        # Add circle marker
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=min(15, radius/25),  # Cap max size
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=row['Name']
        ).add_to(neighborhood_layer)
    
    # Add facilities
    for _, row in facilities.iterrows():
        # Icon based on facility type
        if 'Hospital' in row['Type'] or 'Medical' in row['Type']:
            icon = 'plus'
            color = 'red'
        elif 'Education' in row['Type']:
            icon = 'book'
            color = 'blue'
        elif 'Airport' in row['Type']:
            icon = 'plane'
            color = 'purple'
        elif 'Transit' in row['Type']:
            icon = 'subway'
            color = 'darkgreen'
        elif 'Commercial' in row['Type'] or 'Business' in row['Type']:
            icon = 'briefcase'
            color = 'orange'
        elif 'Tourism' in row['Type'] or 'Sports' in row['Type']:
            icon = 'info-sign'
            color = 'green'
        else:
            icon = 'building'
            color = 'darkblue'
        
        # Special highlighting for emergency or transit
        if highlight_emergency and ('Hospital' in row['Type'] or 'Medical' in row['Type']):
            size = 'lg'
        elif highlight_transit and ('Transit' in row['Type'] or 'Airport' in row['Type']):
            size = 'lg'
        else:
            size = None
        
        # Popup content
        popup_text = f"""
        <b>{row['Name']}</b><br>
        Type: {row['Type']}<br>
        ID: {row['ID']}
        """
        
        # Add marker
        folium.Marker(
            location=[row['Y'], row['X']],
            icon=folium.Icon(color=color, icon=icon, prefix='fa', icon_size=size),
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=row['Name']
        ).add_to(facility_layer)
    
    # Create dictionaries for node positions
    positions = {}
    for _, row in neighborhoods.iterrows():
        positions[row['ID']] = (row['Y'], row['X'])
    
    for _, row in facilities.iterrows():
        positions[row['ID']] = (row['Y'], row['X'])
    
    # Create traffic dictionary if needed
    traffic_dict = {}
    if show_traffic and traffic_data is not None:
        for _, row in traffic_data.iterrows():
            key = (row['FromID'], row['ToID'])
            reverse_key = (row['ToID'], row['FromID'])
            
            # Use morning peak as default
            traffic_value = row['Morning Peak(veh/h)']
            traffic_dict[key] = traffic_value
            traffic_dict[reverse_key] = traffic_value
    
    # Add existing roads
    for _, row in existing_roads.iterrows():
        # Get coordinates
        if row['FromID'] in positions and row['ToID'] in positions:
            from_pos = positions[row['FromID']]
            to_pos = positions[row['ToID']]
            
            # Determine color based on traffic or condition
            if show_traffic and traffic_data is not None:
                # Look up traffic value
                key = (row['FromID'], row['ToID'])
                reverse_key = (row['ToID'], row['FromID'])
                
                if key in traffic_dict:
                    traffic = traffic_dict[key]
                elif reverse_key in traffic_dict:
                    traffic = traffic_dict[reverse_key]
                else:
                    traffic = 0
                
                # Color based on traffic vs capacity ratio
                ratio = min(1.0, traffic / row['Current Capacity(vehicles/hour)'])
                
                if ratio < 0.4:
                    color = 'green'
                    weight = 2
                elif ratio < 0.7:
                    color = 'orange'
                    weight = 3
                else:
                    color = 'red'
                    weight = 4
            else:
                # Color based on road condition
                condition = row['Condition(1-10)']
                
                if condition >= 8:
                    color = 'green'
                elif condition >= 5:
                    color = 'orange'
                else:
                    color = 'red'
                
                weight = 2
            
            # Popup content
            popup_text = f"""
            <b>Road</b><br>
            From: {row['FromID']}<br>
            To: {row['ToID']}<br>
            Distance: {row['Distance(km)']} km<br>
            Capacity: {row['Current Capacity(vehicles/hour)']} veh/h<br>
            Condition: {row['Condition(1-10)']}
            """
            if show_traffic and (key in traffic_dict or reverse_key in traffic_dict):
                popup_text += f"<br>Traffic: {traffic} veh/h"
            
            # Add line
            folium.PolyLine(
                locations=[from_pos, to_pos],
                color=color,
                weight=weight,
                opacity=0.8,
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"{row['Distance(km)']} km"
            ).add_to(road_layer)
    
    # Add new roads if provided
    if new_roads:
        for from_id, to_id in new_roads:
            # Get coordinates
            if from_id in positions and to_id in positions:
                from_pos = positions[from_id]
                to_pos = positions[to_id]
                
                # Add line with dashed style
                folium.PolyLine(
                    locations=[from_pos, to_pos],
                    color='blue',
                    weight=3,
                    opacity=0.8,
                    dash_array='5,10',
                    tooltip=f"Proposed new road"
                ).add_to(new_road_layer)
    
    # Add transit stops if provided
    if highlight_transit and transit_stops:
        transit_layer = folium.FeatureGroup(name="Transit Stops")
        
        for stop_id in transit_stops:
            if stop_id in positions:
                pos = positions[stop_id]
                
                # Get name if available
                name = ""
                if stop_id in neighborhoods['ID'].values:
                    name = neighborhoods[neighborhoods['ID'] == stop_id]['Name'].iloc[0]
                elif stop_id in facilities['ID'].values:
                    name = facilities[facilities['ID'] == stop_id]['Name'].iloc[0]
                
                # Add marker
                folium.CircleMarker(
                    location=pos,
                    radius=5,
                    color='purple',
                    fill=True,
                    fill_opacity=0.7,
                    tooltip=f"Transit Stop: {name}"
                ).add_to(transit_layer)
        
        transit_layer.add_to(map_obj)
    
    # Add all layers to map
    neighborhood_layer.add_to(map_obj)
    facility_layer.add_to(map_obj)
    road_layer.add_to(map_obj)
    
    if new_roads:
        new_road_layer.add_to(map_obj)
    
    # Add layer control
    folium.LayerControl().add_to(map_obj)
    
    return map_obj

def plot_traffic_comparison(traffic_flow, optimal_route, alternative_route, time_column):
    """
    Create a comparison chart of traffic levels for optimal and alternative routes.
    
    Args:
        traffic_flow (DataFrame): Traffic flow data
        optimal_route (list): List of node IDs in the optimal route
        alternative_route (list): List of node IDs in the alternative route
        time_column (str): Column name for the time period to show
        
    Returns:
        plotly.graph_objects.Figure: Traffic comparison chart
    """
    # Extract traffic data for segments in both routes
    optimal_segments = []
    for i in range(len(optimal_route) - 1):
        from_id = optimal_route[i]
        to_id = optimal_route[i+1]
        optimal_segments.append((from_id, to_id))
    
    alternative_segments = []
    for i in range(len(alternative_route) - 1):
        from_id = alternative_route[i]
        to_id = alternative_route[i+1]
        alternative_segments.append((from_id, to_id))
    
    # Get traffic values
    optimal_traffic = []
    optimal_labels = []
    
    for from_id, to_id in optimal_segments:
        # Look for this segment in traffic data (in either direction)
        traffic_row = traffic_flow[
            ((traffic_flow['FromID'] == from_id) & (traffic_flow['ToID'] == to_id)) |
            ((traffic_flow['FromID'] == to_id) & (traffic_flow['ToID'] == from_id))
        ]
        
        if not traffic_row.empty:
            optimal_traffic.append(traffic_row[time_column].iloc[0])
            optimal_labels.append(f"{from_id}-{to_id}")
        else:
            # Use a default value if no traffic data
            optimal_traffic.append(2000)  # Moderate traffic as default
            optimal_labels.append(f"{from_id}-{to_id}")
    
    alternative_traffic = []
    alternative_labels = []
    
    for from_id, to_id in alternative_segments:
        # Look for this segment in traffic data (in either direction)
        traffic_row = traffic_flow[
            ((traffic_flow['FromID'] == from_id) & (traffic_flow['ToID'] == to_id)) |
            ((traffic_flow['FromID'] == to_id) & (traffic_flow['ToID'] == from_id))
        ]
        
        if not traffic_row.empty:
            alternative_traffic.append(traffic_row[time_column].iloc[0])
            alternative_labels.append(f"{from_id}-{to_id}")
        else:
            # Use a default value if no traffic data
            alternative_traffic.append(2000)  # Moderate traffic as default
            alternative_labels.append(f"{from_id}-{to_id}")
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Bar(
        x=optimal_labels,
        y=optimal_traffic,
        name='Optimal Route',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        x=alternative_labels,
        y=alternative_traffic,
        name='Alternative Route',
        marker_color='orange'
    ))
    
    # Update layout
    fig.update_layout(
        title='Traffic Comparison by Road Segment',
        xaxis_title='Road Segments',
        yaxis_title=f'Traffic (vehicles/hour)',
        barmode='group',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def plot_route_visualization(neighborhoods, facilities, existing_roads, 
                            route_1, route_2=None, is_emergency=False):
    """
    Create a map visualization of one or two routes.
    
    Args:
        neighborhoods (DataFrame): Neighborhoods and districts data
        facilities (DataFrame): Important facilities data
        existing_roads (DataFrame): Existing roads data
        route_1 (list): List of node IDs in the primary route
        route_2 (list): List of node IDs in the secondary route (optional)
        is_emergency (bool): Whether this is an emergency route visualization
        
    Returns:
        folium.Map: Interactive map with routes
    """
    # Calculate center of map
    center_lat = neighborhoods['Y'].mean()
    center_lon = neighborhoods['X'].mean()
    
    # Create map
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Create layers
    neighborhood_layer = folium.FeatureGroup(name="Neighborhoods")
    facility_layer = folium.FeatureGroup(name="Facilities")
    route_1_layer = folium.FeatureGroup(name="Primary Route")
    if route_2:
        route_2_layer = folium.FeatureGroup(name="Secondary Route")
    
    # Create dictionaries for node positions and names
    positions = {}
    names = {}
    
    for _, row in neighborhoods.iterrows():
        positions[row['ID']] = (row['Y'], row['X'])
        names[row['ID']] = row['Name']
    
    for _, row in facilities.iterrows():
        positions[row['ID']] = (row['Y'], row['X'])
        names[row['ID']] = row['Name']
    
    # Add neighborhoods with minimal styling
    for _, row in neighborhoods.iterrows():
        # Only add neighborhoods that are in routes for clarity
        if row['ID'] in route_1 or (route_2 and row['ID'] in route_2):
            color = 'blue'
            
            folium.CircleMarker(
                location=[row['Y'], row['X']],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                tooltip=row['Name']
            ).add_to(neighborhood_layer)
    
    # Add facilities with minimal styling
    for _, row in facilities.iterrows():
        # Only add facilities that are in routes for clarity
        if row['ID'] in route_1 or (route_2 and row['ID'] in route_2):
            # Icon based on facility type
            if 'Hospital' in row['Type'] or 'Medical' in row['Type']:
                icon = 'plus'
                color = 'red'
            elif 'Transit' in row['Type']:
                icon = 'subway'
                color = 'darkgreen'
            else:
                icon = 'building'
                color = 'darkblue'
            
            folium.Marker(
                location=[row['Y'], row['X']],
                icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                tooltip=row['Name']
            ).add_to(facility_layer)
    
    # Add route 1
    route_1_coords = []
    for node_id in route_1:
        if node_id in positions:
            route_1_coords.append(positions[node_id])
    
    # Add route polyline
    if is_emergency:
        route_color = 'red'
        route_name = "Emergency Route"
    else:
        route_color = 'blue'
        route_name = "Optimal Route"
    
    if route_1_coords:
        folium.PolyLine(
            locations=route_1_coords,
            color=route_color,
            weight=4,
            opacity=0.8,
            tooltip=route_name
        ).add_to(route_1_layer)
        
        # Add markers for start and end
        folium.Marker(
            location=route_1_coords[0],
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
            tooltip=f"Start: {names.get(route_1[0], route_1[0])}"
        ).add_to(route_1_layer)
        
        folium.Marker(
            location=route_1_coords[-1],
            icon=folium.Icon(color='red', icon='stop', prefix='fa'),
            tooltip=f"End: {names.get(route_1[-1], route_1[-1])}"
        ).add_to(route_1_layer)
    
    # Add route 2 if provided
    if route_2:
        route_2_coords = []
        for node_id in route_2:
            if node_id in positions:
                route_2_coords.append(positions[node_id])
        
        if route_2_coords:
            if is_emergency:
                route_color = 'orange'
                route_name = "Regular Route"
            else:
                route_color = 'green'
                route_name = "Alternative Route"
                
            folium.PolyLine(
                locations=route_2_coords,
                color=route_color,
                weight=4,
                opacity=0.7,
                dash_array='5,10',
                tooltip=route_name
            ).add_to(route_2_layer)
    
    # Add all layers to map
    neighborhood_layer.add_to(map_obj)
    facility_layer.add_to(map_obj)
    route_1_layer.add_to(map_obj)
    if route_2:
        route_2_layer.add_to(map_obj)
    
    # Add layer control
    folium.LayerControl().add_to(map_obj)
    
    return map_obj

def plot_public_transit_routes(neighborhoods, facilities, existing_roads, 
                              metro_routes, bus_routes=None, proposed_routes=None):
    """
    Create a map visualization of metro, bus, and proposed routes.
    
    Args:
        neighborhoods (DataFrame): Neighborhoods and districts data
        facilities (DataFrame): Important facilities data
        existing_roads (DataFrame): Existing roads data
        metro_routes (list): List of lists, each containing station IDs for a metro line
        bus_routes (list): List of lists, each containing stop IDs for a bus route
        proposed_routes (list): List of lists, each containing stop IDs for proposed routes
    
    Returns:
        folium.Map: Interactive map with transit routes
    """
    # Calculate center of map
    center_lat = neighborhoods['Y'].mean()
    center_lon = neighborhoods['X'].mean()
    
    # Create map
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Create layers
    neighborhood_layer = folium.FeatureGroup(name="Neighborhoods")
    facility_layer = folium.FeatureGroup(name="Facilities")
    metro_layer = folium.FeatureGroup(name="Metro Lines")
    bus_layer = folium.FeatureGroup(name="Bus Routes") if bus_routes else None
    proposed_layer = folium.FeatureGroup(name="Proposed Routes") if proposed_routes else None
    
    # Create dictionaries for node positions and names
    positions = {}
    names = {}
    
    for _, row in neighborhoods.iterrows():
        positions[row['ID']] = (row['Y'], row['X'])
        names[row['ID']] = row['Name']
    
    for _, row in facilities.iterrows():
        positions[row['ID']] = (row['Y'], row['X'])
        names[row['ID']] = row['Name']
    
    # Collect all transit stops
    all_transit_stops = set()
    for route in metro_routes:
        all_transit_stops.update(route)
    
    if bus_routes:
        for route in bus_routes:
            all_transit_stops.update(route)
    
    if proposed_routes:
        for route in proposed_routes:
            all_transit_stops.update(route)
    
    # Add neighborhoods
    for _, row in neighborhoods.iterrows():
        # Use smaller markers for non-transit locations
        if row['ID'] in all_transit_stops:
            radius = 6
            color = 'blue'
            fill_opacity = 0.7
        else:
            radius = 3
            color = 'gray'
            fill_opacity = 0.4
            
        folium.CircleMarker(
            location=[row['Y'], row['X']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=fill_opacity,
            tooltip=row['Name']
        ).add_to(neighborhood_layer)
    
    # Add facilities
    for _, row in facilities.iterrows():
        # Only highlight transit-related facilities
        if 'Transit' in row['Type'] or 'Airport' in row['Type'] or row['ID'] in all_transit_stops:
            icon = 'subway' if 'Transit' in row['Type'] else 'building'
            color = 'green' if row['ID'] in all_transit_stops else 'gray'
            
            folium.Marker(
                location=[row['Y'], row['X']],
                icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                tooltip=row['Name']
            ).add_to(facility_layer)
    
    # Add metro routes
    metro_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
    
    for i, route in enumerate(metro_routes):
        route_coords = []
        for node_id in route:
            if node_id in positions:
                route_coords.append(positions[node_id])
        
        if route_coords:
            # Use a different color for each metro line
            color = metro_colors[i % len(metro_colors)]
            
            # Add route line
            folium.PolyLine(
                locations=route_coords,
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=f"Metro Line {i+1}"
            ).add_to(metro_layer)
            
            # Add stations
            for j, node_id in enumerate(route):
                if node_id in positions:
                    folium.CircleMarker(
                        location=positions[node_id],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_opacity=1.0,
                        tooltip=f"Station: {names.get(node_id, node_id)}"
                    ).add_to(metro_layer)
    
    # Add bus routes if provided
    if bus_routes:
        bus_colors = ['cadetblue', 'darkpurple', 'lightred', 'lightblue', 'lightgreen', 
                     'gray', 'black', 'lightgray', 'beige', 'darkgreen']
        
        for i, route in enumerate(bus_routes):
            route_coords = []
            for node_id in route:
                if node_id in positions:
                    route_coords.append(positions[node_id])
            
            if route_coords:
                # Use a different color for each bus route
                color = bus_colors[i % len(bus_colors)]
                
                # Add route line
                folium.PolyLine(
                    locations=route_coords,
                    color=color,
                    weight=2,
                    opacity=0.7,
                    dash_array='5,10',
                    tooltip=f"Bus Route {i+1}"
                ).add_to(bus_layer)
                
                # Add stops (first and last highlighted)
                for j, node_id in enumerate(route):
                    if node_id in positions:
                        if j == 0 or j == len(route) - 1:
                            # Terminal stops
                            folium.CircleMarker(
                                location=positions[node_id],
                                radius=4,
                                color=color,
                                fill=True,
                                fill_opacity=1.0,
                                tooltip=f"Terminal: {names.get(node_id, node_id)}"
                            ).add_to(bus_layer)
                        else:
                            # Regular stops
                            folium.CircleMarker(
                                location=positions[node_id],
                                radius=3,
                                color=color,
                                fill=True,
                                fill_opacity=0.7,
                                tooltip=f"Stop: {names.get(node_id, node_id)}"
                            ).add_to(bus_layer)
    
    # Add proposed routes if provided
    if proposed_routes:
        for i, route in enumerate(proposed_routes):
            route_coords = []
            for node_id in route:
                if node_id in positions:
                    route_coords.append(positions[node_id])
            
            if route_coords:
                # Use a distinct color and style for proposed routes
                folium.PolyLine(
                    locations=route_coords,
                    color='cyan',
                    weight=3,
                    opacity=0.8,
                    dash_array='10,10',
                    tooltip=f"Proposed Route {i+1}"
                ).add_to(proposed_layer)
                
                # Add stops
                for j, node_id in enumerate(route):
                    if node_id in positions:
                        if j == 0 or j == len(route) - 1:
                            # Terminal stops
                            folium.CircleMarker(
                                location=positions[node_id],
                                radius=5,
                                color='cyan',
                                fill=True,
                                fill_opacity=1.0,
                                tooltip=f"Proposed Terminal: {names.get(node_id, node_id)}"
                            ).add_to(proposed_layer)
                        else:
                            # Regular stops
                            folium.CircleMarker(
                                location=positions[node_id],
                                radius=4,
                                color='cyan',
                                fill=True,
                                fill_opacity=0.7,
                                tooltip=f"Proposed Stop: {names.get(node_id, node_id)}"
                            ).add_to(proposed_layer)
    
    # Add all layers to map
    neighborhood_layer.add_to(map_obj)
    facility_layer.add_to(map_obj)
    metro_layer.add_to(map_obj)
    if bus_routes:
        bus_layer.add_to(map_obj)
    if proposed_routes:
        proposed_layer.add_to(map_obj)
    
    # Add layer control
    folium.LayerControl().add_to(map_obj)
    
    return map_obj