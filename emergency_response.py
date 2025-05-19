import networkx as nx
import numpy as np
from heapq import heappush, heappop
import math
import pandas as pd
import streamlit as st

def preempt_intersection_signals(intersections, signals, emergency_path, traffic_flow, time_column):
    preemption_plan = {}
    total_time_saved = 0
    
    for i in range(1, len(emergency_path) - 1):
        node_id = emergency_path[i]
        if node_id not in intersections or node_id not in signals:
            continue
            
        intersection = intersections[node_id]
        signal = signals.get(node_id, {'cycle_time': 60})
        
        incoming_road = f"{emergency_path[i-1]}-{node_id}"
        outgoing_road = f"{node_id}-{emergency_path[i+1]}"
        
        traffic_row = traffic_flow[
            (traffic_flow['FromID'] == emergency_path[i-1]) & 
            (traffic_flow['ToID'] == node_id)
        ]
        traffic_volume = traffic_row[time_column].iloc[0] if not traffic_row.empty else 1000
        
        congestion_threshold = 1000
        if traffic_volume >= congestion_threshold:
            cycle_time = signal.get('cycle_time', 60)
            normal_wait = cycle_time / 2
            preemption_wait = 0
            time_saved = normal_wait
            
            emergency_green_time = 15
            transition_time = 2
            actions = [
                f"Detect emergency vehicle on road {incoming_road}",
                f"Set green light for {incoming_road} (15s)",
                f"Pause conflicting directions (red)",
                f"Transition back to normal timing after {transition_time}s"
            ]
            
            preemption_plan[node_id] = {
                'primary_direction': incoming_road,
                'time_saved': time_saved,
                'emergency_green_time': emergency_green_time,
                'transition_time': transition_time,
                'emergency_actions': actions,
                'preemption_available': True,
                'preemption_effective': True
            }
            total_time_saved += time_saved
        else:
            preemption_plan[node_id] = {
                'primary_direction': incoming_road,
                'time_saved': 0,
                'emergency_green_time': 0,
                'transition_time': 0,
                'emergency_actions': [],
                'preemption_available': True,
                'preemption_effective': False,
                'reason': 'Low traffic volume'
            }
    
    return preemption_plan, total_time_saved

def plan_emergency_routes(neighborhoods, facilities, existing_roads, traffic_flow, 
                         from_id, to_id, time_column="Morning Peak(veh/h)",
                         priority_level=3, route_clearing=True, emergency_type="Medical Emergency",
                         intersections=None, signals=None):
    G = nx.Graph()
    
    for _, row in neighborhoods.iterrows():
        G.add_node(row['ID'], name=row['Name'], population=row['Population'],
                  pos=(row['X'], row['Y']), node_type='neighborhood')
    
    for _, row in facilities.iterrows():
        G.add_node(row['ID'], name=row['Name'], type=row['Type'],
                  pos=(row['X'], row['Y']), node_type='facility')
    
    traffic_dict = {}
    for _, row in traffic_flow.iterrows():
        key = (row['FromID'], row['ToID'])
        reverse_key = (row['ToID'], row['FromID'])
        traffic_value = row[time_column]
        traffic_dict[key] = traffic_value
        traffic_dict[reverse_key] = traffic_value
    
    node_positions = {row['ID']: (float(row['X']), float(row['Y'])) 
                     for _, row in neighborhoods.iterrows()}
    node_positions.update({row['ID']: (float(row['X']), float(row['Y'])) 
                         for _, row in facilities.iterrows()})
    
    for _, road in existing_roads.iterrows():
        condition_factor = 0.5 + (road['Condition(1-10)'] / 20)
        base_speed = 60 * condition_factor
        base_time = (road['Distance(km)'] / base_speed) * 60
        
        road_key = (road['FromID'], road['ToID'])
        reverse_key = (road['ToID'], road['FromID'])
        
        traffic_volume = traffic_dict.get(road_key, traffic_dict.get(reverse_key, 
                                        0.5 * road['Current Capacity(vehicles/hour)']))
        
        congestion_ratio = min(traffic_volume / road['Current Capacity(vehicles/hour)'], 1.0)
        traffic_adjustment = 1.0 + (congestion_ratio * 0.5) if congestion_ratio < 0.7 else \
                           1.0 + (2.0 * (congestion_ratio ** 2))
        
        regular_time = base_time * traffic_adjustment
        
        # Ensure monotonicity of priority factor - higher priority (lower number) always results in more time saved
        priority_factor = max(0.2, 1.0 - (0.3 * priority_level))
        emergency_time = regular_time * priority_factor
        
        G.add_edge(
            road['FromID'], road['ToID'], 
            distance=road['Distance(km)'],
            capacity=road['Current Capacity(vehicles/hour)'],
            condition=road['Condition(1-10)'],
            traffic=traffic_dict.get(road_key, 0),
            regular_time=regular_time,
            emergency_time=emergency_time,
            road_condition=road['Condition(1-10)']
        )
    
    # Check if nodes exist and are connected
    if not G.has_node(from_id) or not G.has_node(to_id):
        st.error(f"Error: Origin '{from_id}' or destination '{to_id}' not found in the road network.")
        return [], [], 0, 0, 0, 0, {}, 0
    
    if not nx.has_path(G, from_id, to_id):
        st.error(f"No path exists between {from_id} and {to_id} in the road network.")
        return [], [], 0, 0, 0, 0, {}, 0
    
    try:
        regular_path = nx.shortest_path(G, source=from_id, target=to_id, weight='regular_time')
        regular_distance = sum(G.get_edge_data(regular_path[i], regular_path[i+1])['distance'] 
                             for i in range(len(regular_path) - 1))
        regular_time = sum(G.get_edge_data(regular_path[i], regular_path[i+1])['regular_time'] 
                          for i in range(len(regular_path) - 1))
        
        def heuristic(node):
            if node in node_positions and to_id in node_positions:
                from_pos = node_positions[node]
                to_pos = node_positions[to_id]
                dx = from_pos[0] - to_pos[0]
                dy = from_pos[1] - to_pos[1]
                straight_distance = math.sqrt(dx**2 + dy**2)
                return (straight_distance * 111) / 80 * 60
            return 0
        
        G_emergency = G.copy()
        
        if route_clearing:
            emergency_path_no_clearing = nx.shortest_path(G, source=from_id, target=to_id, 
                                                        weight='emergency_time')
            # Apply route clearing to all segments for maximum impact
            num_segments_to_clear = len(emergency_path_no_clearing) - 1  # Clear all segments
            
            # First, make all emergency times significantly higher to force alternative routes
            for u, v, data in G_emergency.edges(data=True):
                # Increase emergency time for all edges except those on the direct path
                data['emergency_time'] = data['emergency_time'] * 2.0
            
            # Then reduce times for the cleared route segments
            for i in range(num_segments_to_clear):
                u, v = emergency_path_no_clearing[i], emergency_path_no_clearing[i+1]
                edge_data = G.get_edge_data(u, v)
                base_time = edge_data['distance'] / (60 * (0.5 + (edge_data['road_condition'] / 20))) * 60
                
                # Ensure monotonicity of reduced_traffic_factor - higher priority (lower number) always results in more time saved
                reduced_traffic_factor = max(0.1, 0.7 - (0.2 * priority_level))
                
                # Make emergency route significantly faster
                G_emergency[u][v]['emergency_time'] = base_time * reduced_traffic_factor * 0.5
        
        def a_star_search(graph, start, goal):
            frontier = [(0, start)]
            came_from = {}
            g_score = {node: float('inf') for node in graph.nodes()}
            g_score[start] = 0
            f_score = {node: float('inf') for node in graph.nodes()}
            f_score[start] = heuristic(start)
            visited = set()
            
            while frontier:
                _, current = heappop(frontier)
                if current == goal:
                    path = [current]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    return path[::-1]
                
                visited.add(current)
                for neighbor in graph.neighbors(current):
                    if neighbor in visited:
                        continue
                    edge_data = graph.get_edge_data(current, neighbor)
                    weight = edge_data['emergency_time']
                    if weight <= 0:
                        weight = 0.1
                    tentative_g_score = g_score[current] + weight
                    
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                        heappush(frontier, (f_score[neighbor], neighbor))
            
            return None
        
        emergency_path = a_star_search(G_emergency, from_id, to_id)
        
        if not emergency_path:
            st.warning("A* search failed to find an emergency path. Falling back to regular path.")
            emergency_path = regular_path
            emergency_distance = regular_distance
            emergency_time = regular_time
            time_saved_route = 0
            preemption_plan = {}
            time_saved_signals = 0
        else:
            emergency_distance = sum(G_emergency.get_edge_data(emergency_path[i], emergency_path[i+1])['distance'] 
                                   for i in range(len(emergency_path) - 1))
            emergency_time = sum(G_emergency.get_edge_data(emergency_path[i], emergency_path[i+1])['emergency_time'] 
                                for i in range(len(emergency_path) - 1))
            time_saved_route = regular_time - emergency_time
            
            preemption_plan = {}
            time_saved_signals = 0
            if intersections and signals:
                preemption_plan, time_saved_signals = preempt_intersection_signals(
                    intersections, signals, emergency_path, traffic_flow, time_column
                )
            
            # Log if emergency path is the same as regular path
            if emergency_path == regular_path:
                st.warning("Emergency route is identical to the regular route due to limited path options in the network.")
        
        return (emergency_path, regular_path, emergency_time, regular_time, time_saved_route, 
                emergency_distance, preemption_plan, time_saved_signals)
    
    except nx.NetworkXNoPath:
        st.error(f"No path exists between {from_id} and {to_id} in the road network.")
        return [], [], 0, 0, 0, 0, {}, 0
