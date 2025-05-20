import networkx as nx
import numpy as np
from collections import defaultdict
from functools import lru_cache
from sklearn.cluster import KMeans

def optimize_public_transit(neighborhoods, facilities, existing_roads, 
                           metro_lines, bus_routes, transit_demand,
                           route_id, route_type, optimize_for="Passenger Capacity"):
    """
    Optimize public transportation routes and schedules using dynamic programming.
    Now with proper DP implementation for route optimization.
    
    Args:
        neighborhoods (DataFrame): Neighborhoods and districts data
        facilities (DataFrame): Important facilities data
        existing_roads (DataFrame): Existing roads data
        metro_lines (DataFrame): Metro lines data
        bus_routes (DataFrame): Bus routes data
        transit_demand (DataFrame): Public transportation demand data
        route_id (str): ID of the route to optimize
        route_type (str): "metro" or "bus"
        optimize_for (str): Optimization goal - "Passenger Capacity", "Travel Time", or "Resource Efficiency"
        
    Returns:
        tuple: (optimized_route, current_performance, optimized_performance, suggested_changes)
    """
    # Create graph representation of the road network
    G = nx.Graph()
    
    # Add all neighborhoods and facilities as nodes
    for _, row in neighborhoods.iterrows():
        G.add_node(row['ID'], 
                  name=row['Name'], 
                  population=row['Population'],
                  pos=(row['X'], row['Y']),
                  node_type='neighborhood')
    
    for _, row in facilities.iterrows():
        G.add_node(row['ID'], 
                  name=row['Name'], 
                  type=row['Type'],
                  pos=(row['X'], row['Y']),
                  node_type='facility')
    
    # Add existing roads as edges
    for _, road in existing_roads.iterrows():
        G.add_edge(
            road['FromID'], 
            road['ToID'], 
            distance=road['Distance(km)'],
            capacity=road['Current Capacity(vehicles/hour)'],
            condition=road['Condition(1-10)']
        )
    
    # Get the current route
    current_route = []
    if route_type == "metro":
        row = metro_lines[metro_lines['LineID'] == route_id].iloc[0]
        current_route = row['Stations(comma-separated IDs)'].strip('"').split(',')
        daily_passengers = row['Daily Passengers']
    else:  # bus
        row = bus_routes[bus_routes['RouteID'] == route_id].iloc[0]
        current_route = row['Stops(comma-separated IDs)'].strip('"').split(',')
        daily_passengers = row['Daily Passengers']
        buses_assigned = row['Buses Assigned']
    
    # Current performance metrics
    current_performance = {
        'passengers': daily_passengers
    }
    
    if route_type == "bus":
        current_performance['resources'] = buses_assigned
    
    # Create demand matrix
    demand_matrix = defaultdict(lambda: defaultdict(int))
    for _, row in transit_demand.iterrows():
        demand_matrix[row['FromID']][row['ToID']] = row['Daily Passengers']
    
    # Dynamic programming approach to optimize route
    def calculate_route_performance(route):
        """Calculate performance metrics for a given route using DP concepts"""
        total_demand = 0
        demand_cache = {}
        
        for i in range(len(route)):
            for j in range(i+1, len(route)):
                from_id = route[i]
                to_id = route[j]
                
                if (from_id, to_id) in demand_cache:
                    total_demand += demand_cache[(from_id, to_id)]
                else:
                    demand = demand_matrix[from_id][to_id] + demand_matrix[to_id][from_id]
                    demand_cache[(from_id, to_id)] = demand
                    demand_cache[(to_id, from_id)] = demand
                    total_demand += demand
        
        total_distance = 0
        distance_cache = {}
        
        for i in range(len(route) - 1):
            from_id = route[i]
            to_id = route[i+1]
            
            if (from_id, to_id) in distance_cache:
                total_distance += distance_cache[(from_id, to_id)]
                continue
                
            try:
                path = nx.shortest_path(G, from_id, to_id, weight='distance')
                segment_distance = sum(G[path[j]][path[j+1]]['distance'] 
                                  for j in range(len(path) - 1))
                
                distance_cache[(from_id, to_id)] = segment_distance
                distance_cache[(to_id, from_id)] = segment_distance
                total_distance += segment_distance
            except nx.NetworkXNoPath:
                total_distance += 100
                distance_cache[(from_id, to_id)] = 100
                distance_cache[(to_id, from_id)] = 100
        
        if route_type == "bus":
            estimated_buses = max(5, 
                                int(total_distance / 5) + 
                                int(total_demand / 10000))
            return total_demand, total_distance, estimated_buses
        else:
            estimated_trains = max(3, int(total_distance / 10) + int(total_demand / 20000))
            return total_demand, total_distance, estimated_trains
    
    def optimize_route_dp(current_route, optimize_for):
        """Dynamic programming approach to optimize the route"""
        all_nodes = list(G.nodes())
        
        # Modified to compute total route demand
        def route_demand(route):
            total = 0
            for i in range(len(route)):
                for j in range(i+1, len(route)):
                    total += demand_matrix[route[i]][route[j]] + demand_matrix[route[j]][route[i]]
            return total
        
        @lru_cache(maxsize=None)
        def dp(current_pos, remaining_stops, used_nodes, start_node):
            used_nodes = eval(used_nodes)  # Convert stringified tuple back to set
            if remaining_stops == 0:
                return (0, [])
            
            max_value = -1
            best_route = []
            
            for next_pos in all_nodes:
                if next_pos == current_pos or next_pos in used_nodes:
                    continue
                
                # Calculate value based on optimization goal
                if optimize_for == "Passenger Capacity":
                    # Temporarily add next_pos to evaluate route demand
                    temp_route = [start_node] + [n for n in used_nodes] + [current_pos, next_pos]
                    value = route_demand(temp_route)
                elif optimize_for == "Travel Time":
                    try:
                        path = nx.shortest_path(G, current_pos, next_pos, weight='distance')
                        value = -sum(G[path[j]][path[j+1]]['distance'] for j in range(len(path) - 1))
                    except:
                        value = -100
                else:
                    value = -1
                
                # Update used nodes for recursive call
                new_used_nodes = tuple(set(used_nodes) | {next_pos})
                remaining_value, sub_route = dp(next_pos, remaining_stops - 1, str(new_used_nodes), start_node)
                total_value = value + remaining_value
                
                if total_value > max_value:
                    max_value = total_value
                    best_route = [next_pos] + sub_route
            
            return (max_value, best_route)
        
        best_value = -1
        best_route = []
        
        # Allow up to one additional stop to capture high-demand areas
        max_stops = len(current_route) + 1
        
        for start_node in current_route:
            for num_stops in range(len(current_route), max_stops + 1):
                value, route = dp(start_node, num_stops - 1, str(tuple([start_node])), start_node)
                temp_route = [start_node] + route
                route_value = route_demand(temp_route) if optimize_for == "Passenger Capacity" else value
                if route_value > best_value:
                    best_value = route_value
                    best_route = temp_route
        
        return best_route
    
    current_demand, current_distance, current_resources = calculate_route_performance(current_route)
    
    optimized_route = None
    optimized_performance = {}
    suggested_changes = []
    
    try:
        optimized_route = optimize_route_dp(current_route, optimize_for)
        opt_demand, opt_distance, opt_resources = calculate_route_performance(optimized_route)
        
        optimized_performance = {
            'passengers': opt_demand,
            'distance': opt_distance
        }
        
        if route_type == "bus":
            optimized_performance['resources'] = opt_resources
        
        added_stops = set(optimized_route) - set(current_route)
        removed_stops = set(current_route) - set(optimized_route)
        
        if added_stops:
            suggested_changes.append(f"Add stops at: {', '.join(added_stops)} to better serve demand")
        if removed_stops:
            suggested_changes.append(f"Remove stops at: {', '.join(removed_stops)} to improve efficiency")
        
        if optimize_for == "Travel Time":
            time_saved = current_distance - opt_distance
            if time_saved > 0:
                suggested_changes.append(f"Estimated travel time reduction: {time_saved:.1f} km")
        
        if not added_stops and not removed_stops:
            suggested_changes.append("Current route is already optimal for selected criteria")
            
    except Exception as e:
        suggested_changes.append(f"Optimization failed: {str(e)}")
        optimized_route = None
    
    if optimize_for == "Resource Efficiency":
        if optimized_route:
            route_length = optimized_performance['distance']
            if route_type == "bus":
                optimal_buses = max(5, int(route_length / 4) + int(optimized_performance['passengers'] / 2000))
                if optimal_buses < buses_assigned:
                    efficiency_gain = ((buses_assigned - optimal_buses) / buses_assigned) * 100
                    suggested_changes.append(f"Reduce fleet from {buses_assigned} to {optimal_buses} buses")
                elif optimal_buses > buses_assigned:
                    suggested_changes.append(f"Increase fleet from {buses_assigned} to {optimal_buses} buses")
                else:
                    suggested_changes.append("Current bus allocation is optimal")
                optimized_performance['resources'] = optimal_buses
            else:
                optimal_trains = max(3, int(route_length / 10) + int(optimized_performance['passengers'] / 25000))
                suggested_changes.append(f"Recommended trains for metro: {optimal_trains}")
                optimized_performance['resources'] = optimal_trains
    
    return optimized_route, current_performance, optimized_performance, suggested_changes

def optimize_schedule_dp(route_id, route_type, metro_lines, bus_routes, transit_demand, peak_hours, resource_constraint=100):
    """
    Optimize transit schedules using a dynamic programming approach.
    
    Args:
        route_id (str): ID of the route to optimize
        route_type (str): "metro" or "bus"
        metro_lines (DataFrame): Metro lines data
        bus_routes (DataFrame): Bus routes data
        transit_demand (DataFrame): Public transportation demand data
        peak_hours (list): List of peak hour periods (e.g., ["6-9 AM", "3-6 PM"])
        resource_constraint (int): Percentage of available resources (70-130)
    
    Returns:
        tuple: (current_schedule, optimized_schedule)
    """
    time_periods = [
        "5-6 AM", "6-7 AM", "7-8 AM", "8-9 AM", "9-10 AM", "10-11 AM", "11-12 PM",
        "12-1 PM", "1-2 PM", "2-3 PM", "3-4 PM", "4-5 PM", "5-6 PM", "6-7 PM",
        "7-8 PM", "8-9 PM", "9-10 PM", "10-11 PM", "11-12 AM"
    ]
    
    if route_type == "metro":
        route_data = metro_lines[metro_lines['LineID'] == route_id].iloc[0]
        base_headway = 5 if route_id == "M1" else 6 if route_id == "M2" else 8
        base_capacity = 1500 if route_id == "M1" else 1200 if route_id == "M2" else 800
        base_resources = 8 if route_id == "M1" else 7 if route_id == "M2" else 5
        stops = route_data['Stations(comma-separated IDs)'].strip('"').split(',')
    else:
        route_data = bus_routes[bus_routes['RouteID'] == route_id].iloc[0]
        base_headway = 10
        base_capacity = 300
        base_resources = route_data['Buses Assigned']
        stops = route_data['Stops(comma-separated IDs)'].strip('"').split(',')
    
    demand_per_period = {}
    for period in time_periods:
        hour = int(period.split("-")[0].split(" ")[0])
        if "PM" in period and hour != 12:
            hour += 12
        if "AM" in period and hour == 12:
            hour = 0
        
        if 6 <= hour < 9 or 15 <= hour < 18:
            factor = 1.0
        elif 9 <= hour < 15 or 18 <= hour < 21:
            factor = 0.6
        else:
            factor = 0.3
        
        total_demand = 0
        for i in range(len(stops)):
            for j in range(i+1, len(stops)):
                from_id = stops[i]
                to_id = stops[j]
                demand = transit_demand[
                    (transit_demand['FromID'] == from_id) & (transit_demand['ToID'] == to_id)
                ]['Daily Passengers'].sum()
                total_demand += demand
        demand_per_period[period] = total_demand * factor / 24
    
    available_resources = int(base_resources * resource_constraint / 100)
    
    current_schedule = []
    for period in time_periods:
        is_peak = any(period in peak for peak in peak_hours)
        factor = 1.0 if is_peak else 0.6 if demand_per_period[period] > total_demand * 0.5 / 24 else 0.3
        headway = max(int(base_headway / factor), 3)
        resources = max(int(base_resources * factor), 2)
        current_schedule.append({
            "Time Period": period,
            "Headway (min)": headway,
            "Resources in Service": resources,
            "Capacity (passengers/hour)": int(resources * base_capacity * 60 / headway),
            "Peak": is_peak
        })
    
    @lru_cache(maxsize=None)
    def dp_schedule(period_idx, remaining_resources):
        if period_idx >= len(time_periods):
            return 0, []
        
        period = time_periods[period_idx]
        demand = demand_per_period[period]
        is_peak = any(period in peak for peak in peak_hours)
        
        best_cost = float('inf')
        best_schedule = []
        
        min_resources = 2 if route_type == "metro" else 1
        max_resources = min(available_resources, base_resources if is_peak else int(base_resources * 0.8))
        
        for resources in range(min_resources, max_resources + 1):
            headway = max(3, int(base_headway * base_resources / resources))
            capacity = resources * base_capacity * 60 / headway
            wait_time = headway / 2
            cost = wait_time * demand
            
            if capacity < demand:
                cost += (demand - capacity) * 10
            
            remaining = available_resources - resources
            if remaining < 0:
                continue
            sub_cost, sub_schedule = dp_schedule(period_idx + 1, remaining)
            total_cost = cost + sub_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_schedule = [{
                    "Time Period": period,
                    "Headway (min)": headway,
                    "Resources in Service": resources,
                    "Capacity (passengers/hour)": int(capacity),
                    "Peak": is_peak
                }] + sub_schedule
        
        return best_cost, best_schedule
    
    _, optimized_schedule = dp_schedule(0, available_resources)
    
    return current_schedule, optimized_schedule

def analyze_transfer_points(metro_lines, bus_routes, transit_demand, neighborhoods, facilities, existing_roads):
    """
    Analyze and optimize transfer points between metro and bus routes.
    
    Args:
        metro_lines (DataFrame): Metro lines data
        bus_routes (DataFrame): Bus routes data
        transit_demand (DataFrame): Public transportation demand data
        neighborhoods (DataFrame): Neighborhoods data
        facilities (DataFrame): Facilities data
        existing_roads (DataFrame): Existing roads data
    
    Returns:
        dict: Transfer point analysis and recommendations
    """
    G = nx.Graph()
    for _, row in neighborhoods.iterrows():
        G.add_node(row['ID'], pos=(row['X'], row['Y']))
    for _, row in facilities.iterrows():
        G.add_node(row['ID'], pos=(row['X'], row['Y']))
    for _, road in existing_roads.iterrows():
        G.add_edge(road['FromID'], road['ToID'], distance=road['Distance(km)'])
    
    metro_stops = set()
    for _, row in metro_lines.iterrows():
        stops = row['Stations(comma-separated IDs)'].strip('"').split(',')
        metro_stops.update(stops)
    
    bus_stops = set()
    for _, row in bus_routes.iterrows():
        stops = row['Stops(comma-separated IDs)'].strip('"').split(',')
        bus_stops.update(stops)
    
    transfer_points = metro_stops & bus_stops
    
    transfer_analysis = {}
    for point in transfer_points:
        demand = 0
        for _, row in transit_demand.iterrows():
            if row['FromID'] == point or row['ToID'] == point:
                demand += row['Daily Passengers']
        
        nearby_stops = set()
        for metro_row in metro_lines.itertuples():
            stops = metro_row.__getattribute__('Stations(comma-separated IDs)').strip('"').split(',')
            if point in stops:
                nearby_stops.update(stops)
        for bus_row in bus_routes.itertuples():
            stops = bus_row.__getattribute__('Stops(comma-separated IDs)').strip('"').split(',')
            if point in stops:
                nearby_stops.update(stops)
        
        avg_walking_distance = 0
        if nearby_stops:
            distances = []
            for other_stop in nearby_stops:
                if other_stop != point:
                    try:
                        path = nx.shortest_path(G, point, other_stop, weight='distance')
                        distance = sum(G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
                        distances.append(distance)
                    except nx.NetworkXNoPath:
                        continue
            avg_walking_distance = np.mean(distances) if distances else 0
        
        transfer_analysis[point] = {
            'demand': demand,
            'avg_walking_distance': avg_walking_distance,
            'recommendations': []
        }
        
        if avg_walking_distance > 1.0:
            transfer_analysis[point]['recommendations'].append(
                f"Reduce walking distance (currently {avg_walking_distance:.1f} km) by improving connectivity or adding shuttle services"
            )
        if demand > 5000:
            transfer_analysis[point]['recommendations'].append(
                f"High transfer demand ({demand:,} passengers). Consider dedicated transfer facilities or synchronized schedules"
            )
    
    return transfer_analysis

def design_integrated_network(
    neighborhoods, facilities, existing_roads, metro_lines, bus_routes, transit_demand,
    max_new_routes=5, max_budget=1000, fleet_availability=None
):
    """
    Design an integrated public transit network by proposing new routes to connect underserved areas.
    
    Args:
        neighborhoods (DataFrame): Neighborhoods data
        facilities (DataFrame): Facilities data
        existing_roads (DataFrame): Existing roads data
        metro_lines (DataFrame): Metro lines data
        bus_routes (DataFrame): Bus routes data
        transit_demand (DataFrame): Transit demand data
        max_new_routes (int): Maximum number of new routes to propose
        max_budget (float): Maximum budget for new route infrastructure (Million EGP)
        fleet_availability (dict): Available vehicles, e.g., {'bus': 50, 'metro': 10}
    
    Returns:
        tuple: (new_routes, coverage_stats, recommendations)
    """
    G = nx.Graph()
    for _, row in neighborhoods.iterrows():
        G.add_node(row['ID'], pos=(row['X'], row['Y']), population=row['Population'])
    for _, row in facilities.iterrows():
        G.add_node(row['ID'], pos=(row['X'], row['Y']))
    for _, road in existing_roads.iterrows():
        G.add_edge(road['FromID'], road['ToID'], distance=road['Distance(km)'])

    existing_stops = set()
    for _, row in metro_lines.iterrows():
        stops = row['Stations(comma-separated IDs)'].strip('"').split(',')
        existing_stops.update(stops)
    for _, row in bus_routes.iterrows():
        stops = row['Stops(comma-separated IDs)'].strip('"').split(',')
        existing_stops.update(stops)

    coords = neighborhoods[['X', 'Y']].values
    populations = neighborhoods['Population'].values
    kmeans = KMeans(n_clusters=min(10, len(neighborhoods)), random_state=0).fit(coords, sample_weight=populations)
    clusters = kmeans.labels_

    cluster_coverage = defaultdict(int)
    cluster_population = defaultdict(int)
    cluster_nodes = defaultdict(list)
    for idx, (cluster, row) in enumerate(zip(clusters, neighborhoods.itertuples())):
        node_id = row.ID
        cluster_nodes[cluster].append(node_id)
        cluster_population[cluster] += row.Population
        if node_id in existing_stops:
            cluster_coverage[cluster] += 1

    underserved_clusters = []
    cluster_info = []
    for cluster in set(clusters):
        coverage_ratio = cluster_coverage[cluster] / len(cluster_nodes[cluster]) if cluster_nodes[cluster] else 0
        total_population = cluster_population[cluster]
        cluster_info.append(f"Cluster {cluster}: Coverage Ratio = {coverage_ratio:.2f}, Population = {total_population:,}")
        if coverage_ratio < 0.95 and total_population > 1000:
            underserved_clusters.append((cluster, total_population, cluster_nodes[cluster]))
    
    underserved_clusters.sort(key=lambda x: (
        x[1],
        1 / (cluster_coverage[x[0]] / len(cluster_nodes[x[0]]) if cluster_nodes[x[0]] else 1)
    ), reverse=True)

    stop_demand = defaultdict(int)
    for _, row in transit_demand.iterrows():
        stop_demand[row['FromID']] += row['Daily Passengers']
        stop_demand[row['ToID']] += row['Daily Passengers']
    transit_hubs = sorted([(stop, demand) for stop, demand in stop_demand.items() if stop in existing_stops], 
                         key=lambda x: x[1], reverse=True)[:10]
    transit_hub_ids = [hub[0] for hub in transit_hubs]

    new_routes = []
    total_cost = 0
    recommendations = []
    
    considered_clusters = min(max_new_routes * 2, len(underserved_clusters))
    
    for cluster, population, nodes in underserved_clusters[:considered_clusters]:
        if len(new_routes) >= max_new_routes:
            break
            
        cluster_populations = [(node, neighborhoods[neighborhoods['ID'] == node]['Population'].iloc[0] if node in neighborhoods['ID'].values else 0) 
                              for node in nodes]
        if not cluster_populations:
            continue
            
        cluster_center = max(cluster_populations, key=lambda x: x[1])[0]

        min_distance = float('inf')
        closest_hub = None
        best_path = []
        
        if not transit_hub_ids:
            central_nodes = sorted([(node, sum(nx.shortest_path_length(G, node, target, weight='distance') 
                                  for target in G.nodes() if target != node and nx.has_path(G, node, target)))
                           for node in G.nodes() if node in existing_stops], 
                          key=lambda x: x[1])[:5]
            if central_nodes:
                transit_hub_ids = [node for node, _ in central_nodes]
            else:
                transit_hub_ids = [node for node in G.nodes() if G.degree(node) > 2][:5]
        
        for hub in transit_hub_ids[:5]:
            if hub == cluster_center:
                continue
                
            try:
                path = nx.shortest_path(G, cluster_center, hub, weight='distance')
                if len(path) < 2:
                    continue
                    
                distance = sum(G[path[i]][path[i+1]].get('distance', 1.0) for i in range(len(path)-1))
                
                if any(len(set(path).intersection(set(existing_route))) > len(path) * 0.5 for existing_route in new_routes):
                    continue
                    
                if distance < min_distance:
                    min_distance = distance
                    closest_hub = hub
                    best_path = path
            except (nx.NetworkXNoPath, KeyError):
                continue

        if closest_hub and best_path:
            if min_distance < 0.1:
                min_distance = 1.0
                
            cost = min_distance * 10
            
            if total_cost + cost > max_budget * 0.8 and len(new_routes) == 0:
                total_cost += cost
                new_routes.append(best_path)
                recommendations.append(f"Proposed new route from cluster {cluster} (Population: {population:,}) to hub {closest_hub}: {len(best_path)} stops, {min_distance:.1f} km, Cost: {cost:.1f} Million EGP")
            elif total_cost + cost <= max_budget:
                total_cost += cost
                new_routes.append(best_path)
                recommendations.append(f"Proposed new route from cluster {cluster} (Population: {population:,}) to hub {closest_hub}: {len(best_path)} stops, {min_distance:.1f} km, Cost: {cost:.1f} Million EGP")
            else:
                recommendations.append(f"Could not propose route to cluster {cluster} (Population: {population:,}) due to budget constraints.")
                continue

    if fleet_availability:
        required_buses = len(new_routes) * 5
        if 'bus' in fleet_availability and required_buses > fleet_availability['bus']:
            recommendations.append(f"Insufficient bus fleet: Need {required_buses} buses, but only {fleet_availability['bus']} available.")

    total_population = neighborhoods['Population'].sum()
    
    neighborhoods_with_stops = set()
    for stop_id in existing_stops:
        if stop_id in neighborhoods['ID'].values:
            neighborhoods_with_stops.add(stop_id)
    
    covered_population_before = neighborhoods[neighborhoods['ID'].isin(neighborhoods_with_stops)]['Population'].sum()
    
    new_covered_nodes = set()
    for route in new_routes:
        for node in route:
            if node in neighborhoods['ID'].values and node not in neighborhoods_with_stops:
                new_covered_nodes.add(node)
    
    newly_covered_population = neighborhoods[neighborhoods['ID'].isin(new_covered_nodes)]['Population'].sum()
    covered_population_after = covered_population_before + newly_covered_population

    coverage_stats = {
        'total_population': total_population,
        'covered_population_before': covered_population_before,
        'covered_population_after': covered_population_after,
        'coverage_percentage_before': (covered_population_before / total_population * 100),
        'coverage_percentage_after': (covered_population_after / total_population * 100),
        'total_cost': total_cost
    }

    if not new_routes:
        recommendations.append("No new routes proposed due to budget constraints or lack of underserved areas.")

    return new_routes, coverage_stats, recommendations