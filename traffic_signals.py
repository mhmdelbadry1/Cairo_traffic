import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

def identify_intersections(existing_roads, neighborhoods, facilities):
    """
    Identify major intersections based on road connectivity.
    
    Args:
        existing_roads (DataFrame): Existing roads data
        neighborhoods (DataFrame): Neighborhoods data
        facilities (DataFrame): Facilities data
    
    Returns:
        dict: Intersections with node ID, name, type, position, degree, and importance
    """
    # Build adjacency list
    adj_list = {}
    for _, row in existing_roads.iterrows():
        u, v = row['FromID'], row['ToID']
        adj_list.setdefault(u, set()).add(v)
        adj_list.setdefault(v, set()).add(u)
    
    # Node information
    node_info = {}
    for _, row in neighborhoods.iterrows():
        node_info[row['ID']] = {'name': row['Name'], 'type': 'neighborhood', 
                              'pos': (row['X'], row['Y'])}
    for _, row in facilities.iterrows():
        node_info[row['ID']] = {'name': row['Name'], 'type': 'facility', 
                              'pos': (row['X'], row['Y'])}
    
    # Identify intersections (nodes with degree >= 3)
    intersections = {}
    for node in adj_list:
        degree = len(adj_list[node])
        if degree >= 3 and node in node_info:
            importance = degree * (2 if node_info[node]['type'] == 'facility' else 1)
            intersections[node] = {
                'name': node_info[node]['name'],
                'type': node_info[node]['type'],
                'pos': node_info[node]['pos'],
                'degree': degree,
                'importance': importance
            }
    
    return intersections

def real_time_signal_optimization(intersections, traffic_flow, existing_roads, time_column):
    """
    Optimize signal timings using a greedy approach with dynamic cycle times and road-specific capacities.
    
    Args:
        intersections (dict): Intersection data
        traffic_flow (DataFrame): Traffic flow data
        existing_roads (DataFrame): Existing roads data for road-specific capacities
        time_column (str): Traffic time period
    
    Returns:
        dict: Signal timings for each intersection
    """
    signals = {}
    for node_id in intersections:
        # Get connected roads
        traffic_data = traffic_flow[
            (traffic_flow['ToID'] == node_id) | (traffic_flow['FromID'] == node_id)
        ]
        
        # Calculate total traffic volume
        total_volume = 0
        phases = []
        for _, row in traffic_data.iterrows():
            volume = row[time_column]
            total_volume += volume
            road_id = f"{row['FromID']}-{row['ToID']}" if row['ToID'] == node_id else \
                     f"{row['ToID']}-{row['FromID']}"
            # Fetch road-specific capacity
            road_data = existing_roads[
                ((existing_roads['FromID'] == row['FromID']) & (existing_roads['ToID'] == row['ToID'])) |
                ((existing_roads['FromID'] == row['ToID']) & (existing_roads['ToID'] == row['FromID']))
            ]
            capacity = road_data['Current Capacity(vehicles/hour)'].iloc[0] if not road_data.empty else 3000
            phases.append({
                'road_id': road_id,
                'volume': volume,
                'saturation': min(volume / capacity, 1.0),
                'urgency': 0
            })
        
        # Dynamic cycle time based on total traffic volume
        if total_volume < 2000:
            cycle_time = 90  # Low traffic
        elif total_volume <= 5000:
            cycle_time = 120  # Moderate traffic
        else:
            cycle_time = 180  # High traffic
        
        min_green = 10  # Seconds
        total_green = cycle_time - (len(phases) * 2)  # 2s yellow per phase
        green_times = []
        
        for phase in phases:
            if total_volume > 0:
                proportion = phase['volume'] / total_volume
                green_time = max(min_green, int(proportion * total_green))
            else:
                green_time = min_green
            phase['green_time'] = green_time
            green_times.append(green_time)
        
        # Calculate performance
        avg_wait = sum(phase['saturation'] * (cycle_time - phase['green_time']) / 2 
                      for phase in phases) / len(phases) if phases else 0
        efficiency = sum(phase['green_time'] * phase['volume'] for phase in phases) / \
                    (cycle_time * total_volume) if total_volume > 0 else 1.0
        queue_lengths = [phase['volume'] * (cycle_time - phase['green_time']) / 3600 
                        for phase in phases]
        
        signals[node_id] = {
            'cycle_time': cycle_time,
            'phases': phases,
            'performance': {
                'efficiency': efficiency,
                'avg_wait_time': avg_wait,
                'queue_lengths': queue_lengths
            }
        }
    
    return signals

def emergency_vehicle_preemption(intersections, signals, emergency_route, traffic_flow, time_column):
    """
    Simulate emergency vehicle preemption at all intersections along the route with congestion-based adjustments.
    
    Args:
        intersections (dict): Intersection data
        signals (dict): Signal timings
        emergency_route (list): List of node IDs in the route
        traffic_flow (DataFrame): Traffic flow data
        time_column (str): Traffic time period
    
    Returns:
        tuple: (preemption_plan, total_time_saved)
    """
    preemption_plan = {}
    total_time_saved = 0
    
    if not emergency_route or len(emergency_route) < 2:
        return preemption_plan, total_time_saved
    
    # Process all intersections along the route (excluding start and end nodes)
    for i in range(1, len(emergency_route) - 1):
        node_id = emergency_route[i]
        if node_id not in intersections or node_id not in signals:
            continue
        
        signal = signals[node_id]
        
        # Get traffic volume at this intersection
        traffic_data = traffic_flow[
            (traffic_flow['ToID'] == node_id) | (traffic_flow['FromID'] == node_id)
        ]
        total_volume = traffic_data[time_column].sum()
        
        # Congestion-based preemption
        base_green_time = 15
        transition_time = 2
        
        if total_volume > 1000:  # High congestion threshold
            # Scale green time based on congestion
            extra_time = int((total_volume - 1000) / 500)  # 1s per 500 veh/h above threshold
            emergency_green_time = base_green_time + extra_time
            normal_wait = signal['cycle_time'] / 2  # Assume red half the cycle
            preemption_wait = 0
            time_saved = normal_wait
            total_time_saved += time_saved
            
            preemption_plan[node_id] = {
                'preemption_effective': True,
                'primary_direction': f"{emergency_route[i-1]}-{emergency_route[i+1]}",
                'time_saved': time_saved,
                'emergency_green_time': emergency_green_time,
                'transition_time': transition_time,
                'emergency_actions': [
                    f"Detect emergency vehicle at {node_id}",
                    f"Set green light for {emergency_green_time}s",
                    f"Pause conflicting directions",
                    f"Transition back after {transition_time}s"
                ]
            }
        else:
            # No preemption needed in low congestion
            preemption_plan[node_id] = {
                'preemption_effective': False,
                'reason': 'Low congestion, no preemption needed',
                'time_saved': 0
            }
    
    return preemption_plan, total_time_saved

def optimal_signal_timing(intersection_id, traffic_data, time_column, cycle_time):
    """
    Compute optimal signal timings using linear programming to minimize wait times.
    
    Args:
        intersection_id (str): ID of the intersection
        traffic_data (DataFrame): Traffic data for the intersection
        time_column (str): Column name for the time period
        cycle_time (int): Total cycle time in seconds
    
    Returns:
        tuple: (optimal_green_times, optimal_wait_time)
    """
    prob = LpProblem("Signal_Optimization", LpMinimize)
    green_vars = {}
    
    # Create variables for green times
    for _, row in traffic_data.iterrows():
        road_id = f"{row['FromID']}-{row['ToID']}" if row['ToID'] == intersection_id else \
                 f"{row['ToID']}-{row['FromID']}"
        green_vars[road_id] = LpVariable(f"Green_{road_id}", 10, cycle_time)  # Min 10s
    
    # Constraint: Total green time (plus yellow times) <= cycle time
    total_yellow = len(green_vars) * 2  # 2s yellow per phase
    prob += lpSum(green_vars[road_id] for road_id in green_vars) <= cycle_time - total_yellow
    
    # Objective: Minimize total wait time
    wait_times = []
    for _, row in traffic_data.iterrows():
        road_id = f"{row['FromID']}-{row['ToID']}" if row['ToID'] == intersection_id else \
                 f"{row['ToID']}-{row['FromID']}"
        volume = row[time_column]
        # Wait time approximation: volume * (cycle_time - green_time) / 2
        wait_time = volume * (cycle_time - green_vars[road_id]) / 2
        wait_times.append(wait_time)
    
    prob += lpSum(wait_times)  # Minimize total wait time
    prob.solve()
    
    optimal_green = {road_id: value(green_vars[road_id]) for road_id in green_vars}
    optimal_wait = value(prob.objective) / len(traffic_data) if traffic_data.shape[0] > 0 else 0
    
    return optimal_green, optimal_wait