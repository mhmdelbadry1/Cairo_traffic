import networkx as nx
import pandas as pd
import numpy as np
from scipy.spatial import distance
from heapq import heappush, heappop

def create_mst_network(neighborhoods, facilities, existing_roads, potential_roads=None, 
                      algorithm_choice="kruskal", population_weight=0.5, 
                      facility_priority=1.5, max_budget=None):
    """
    Create an optimized road network using Minimum Spanning Tree algorithms,
    with modifications for population centers, critical facilities, and budget constraints.
    
    Args:
        neighborhoods (DataFrame): Neighborhoods and districts data
        facilities (DataFrame): Important facilities data
        existing_roads (DataFrame): Existing roads data
        potential_roads (DataFrame): Potential new roads data (or None if not considered)
        algorithm_choice (str): "kruskal" or "prim" - algorithm to use
        population_weight (float): Weight factor for population in calculations (0-1)
        facility_priority (float): Priority factor for connections to critical facilities (>=1)
        max_budget (float): Maximum budget for new road construction (Million EGP)
        
    Returns:
        tuple: (G, total_time_cost, new_roads_added, total_distance, connectivity_score, cost_analysis)
            G: The optimized network graph
            total_time_cost: Total time cost of the network (minutes)
            new_roads_added: List of tuples (from, to, cost) for recommended new roads
            total_distance: Total network distance (km)
            connectivity_score: Measure of network connectivity
            cost_analysis: Dictionary with cost breakdown and effectiveness metrics
    """
    # Create a graph for the road network
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
        condition_factor = (11 - road['Condition(1-10)']) / 5  # Higher condition = lower factor
        avg_speed = 60 * (1 - (condition_factor * 0.5))  # km/h, ranges from 30-60 km/h
        travel_time = (road['Distance(km)'] / avg_speed) * 60  # minutes

        G.add_edge(road['FromID'], 
                   road['ToID'], 
                   weight=travel_time,
                   time_cost=travel_time,
                   distance=road['Distance(km)'],
                   capacity=road['Current Capacity(vehicles/hour)'],
                   condition=road['Condition(1-10)'],
                   road_type='existing',
                   cost=0)

    # Initialize variables
    new_roads_added = []
    budget_used = 0
    cost_analysis = {
        'existing_roads_cost': 0,  # Maintenance cost assumed 0 for simplicity
        'new_roads_cost': 0,
        'total_cost': 0,
        'cost_per_km': 0,
        'cost_effectiveness': 0  # Population served per million EGP
    }

    if potential_roads is not None:
        G_potential = G.copy()
        edges = []

        # Add potential roads with modified weights
        for _, road in potential_roads.iterrows():
            if G_potential.has_edge(road['FromID'], road['ToID']):
                continue

            # Population factor
            population_factor = 1.0
            population_served = 0
            from_node_type = G_potential.nodes[road['FromID']].get('node_type', 'unknown')
            to_node_type = G_potential.nodes[road['ToID']].get('node_type', 'unknown')

            if from_node_type == 'neighborhood':
                population_served += G_potential.nodes[road['FromID']].get('population', 0)
            if to_node_type == 'neighborhood':
                population_served += G_potential.nodes[road['ToID']].get('population', 0)
            if population_served > 0:
                population_factor = 1.0 / (1.0 + population_weight * population_served / 1000000)

            # Facility priority (extra priority for medical and transit facilities)
            priority_factor = 1.0
            if from_node_type  facility_priority = 1.5
            if from_node_type == 'facility' or to_node_type == 'facility':
                priority_factor = 1.0 / facility_priority
                if (from_node_type == 'facility' and G_potential.nodes[road['FromID']].get('type', '') in ['Medical', 'Transit Hub']):
                    priority_factor = 1.0 / (facility_priority * 1.5)
                elif (to_node_type == 'facility' and G_potential.nodes[road['ToID']].get('type', '') in ['Medical', 'Transit Hub']):
                    priority_factor = 1.0 / (facility_priority * 1.5)

            # Travel time for new roads (assume condition 8/10)
            avg_speed = 60 * (1 - ((11 - 8) / 5 * 0.5))  # ~48 km/h
            travel_time = (road['Distance(km)'] / avg_speed) * 60  # minutes
            modified_time = travel_time * population_factor * priority_factor

            # Add potential road
            G_potential.add_edge(
                road['FromID'], 
                road['ToID'], 
                weight=modified_time,
                time_cost=travel_time,
                distance=road['Distance(km)'],
                capacity=road['Estimated Capacity(vehicles/hour)'],
                condition=8,
                road_type='potential',
                cost=road['Construction Cost(Million EGP)']
            )
            edges.append((modified_time, road['FromID'], road['ToID'], road['Distance(km)'], road['Construction Cost(Million EGP)'], population_served))

        # Custom Kruskal's Algorithm
        def kruskal(G_potential, edges):
            parent = {node: node for node in G_potential.nodes}
            rank = {node: 0 for node in G_potential.nodes}
            
            def find(node):
                if parent[node] != node:
                    parent[node] = find(parent[node])
                return parent[node]
            
            def union(node1, node2):
                root1, root2 = find(node1), find(node2)
                if root1 != root2:
                    if rank[root1] < rank[root2]:
                        parent[root1] = root2
                    elif rank[root1] > rank[root2]:
                        parent[root2] = root1
                    else:
                        parent[root2] = root1
                        rank[root1] += 1
            
            # Sort edges by weight, then by population served (descending) for tie-breaking
            sorted_edges = sorted(edges, key=lambda x: (x[0], -x[5]))
            mst_edges = []
            
            for weight, u, v, distance, cost, pop_served in sorted_edges:
                if find(u) != find(v):
                    union(u, v)
                    mst_edges.append((u, v, G_potential[u][v]))
            
            return mst_edges

        # Custom Prim's Algorithm
        def prim(G_potential):
            # Start from a high-priority facility (e.g., Medical or Transit Hub)
            start_node = None
            for node in G_potential.nodes:
                if G_potential.nodes[node].get('node_type') == 'facility':
                    if G_potential.nodes[node].get('type') in ['Medical', 'Transit Hub']:
                        start_node = node
                        break
            if not start_node:
                start_node = list(G_potential.nodes)[0]  # Fallback to first node
            
            visited = {start_node}
            heap = []
            mst_edges = []
            
            for v in G_potential.neighbors(start_node):
                heappush(heap, (G_potential[start_node][v]['weight'], start_node, v, G_potential[start_node][v]))
            
            while heap and len(visited) < len(G_potential.nodes):
                weight, u, v, data = heappop(heap)
                if v in visited:
                    continue
                visited.add(v)
                mst_edges.append((u, v, data))
                
                for w in G_potential.neighbors(v):
                    if w not in visited:
                        heappush(heap, (G_potential[v][w]['weight'], v, w, G_potential[v][w]))
            
            return mst_edges

        # Calculate MST
        if algorithm_choice == "kruskal":
            mst_edges = kruskal(G_potential, edges)
        else:  # prim
            mst_edges = prim(G_potential)

        # Add potential roads within budget
        # Sort by cost-effectiveness (population served per cost)
        sorted_edges = sorted(mst_edges, key=lambda x: x[2]['cost'] / (G_potential.nodes[x[0]].get('population', 0) + G_potential.nodes[x[1]].get('population', 0) + 1) if x[2]['cost'] > 0 else float('inf'))
        for u, v, data in sorted_edges:
            if data['road_type'] == 'potential' and (max_budget is None or budget_used + data['cost'] <= max_budget):
                G.add_edge(u, v, **data)
                new_roads_added.append((u, v, data['cost']))
                budget_used += data['cost']
                cost_analysis['new_roads_cost'] += data['cost']

    # Ensure connectivity for all neighborhoods
    neighborhoods_ids = set(neighborhoods['ID'])
    connected_nodes = set(nx.node_connected_component(G, list(neighborhoods_ids)[0]))
    missing_nodes = neighborhoods_ids - connected_nodes

    if missing_nodes:
        for node in missing_nodes:
            min_cost = float('inf')
            best_edge = None
            for connected_node in connected_nodes:
                pos1 = G.nodes[node]['pos']
                pos2 = G.nodes[connected_node]['pos']
                dist = distance.euclidean(pos1, pos2) * 100  # Approximate km
                est_cost = dist * 20  # 20M EGP per km
                if est_cost < min_cost:
                    min_cost = est_cost
                    best_edge = (node, connected_node)
            
            if best_edge:
                if max_budget is None or budget_used + min_cost <= max_budget:
                    avg_speed = 48  # New road, condition 8/10
                    travel_time = (min_cost / 20 / avg_speed) * 60
                    G.add_edge(
                        best_edge[0], best_edge[1],
                        weight=travel_time,
                        time_cost=travel_time,
                        distance=min_cost / 20,
                        capacity=3000,
                        condition=8,
                        road_type='potential',
                        cost=min_cost
                    )
                    new_roads_added.append((best_edge[0], best_edge[1], min_cost))
                    budget_used += min_cost
                    cost_analysis['new_roads_cost'] += min_cost
                    connected_nodes.add(node)

    # Calculate metrics
    total_time_cost = sum(nx.get_edge_attributes(G, 'time_cost').values())
    total_distance = sum(nx.get_edge_attributes(G, 'distance').values())
    connectivity_score = nx.average_clustering(G) * 100 if nx.number_of_nodes(G) > 2 else 0

    # Cost-effectiveness analysis
    cost_analysis['total_cost'] = cost_analysis['new_roads_cost']
    cost_analysis['cost_per_km'] = cost_analysis['new_roads_cost'] / total_distance if total_distance > 0 else 0
    # Benefit: population served per million EGP
    total_population_served = sum(G.nodes[node].get('population', 0) for node in G.nodes if G.nodes[node].get('node_type') == 'neighborhood')
    cost_analysis['cost_effectiveness'] = total_population_served / cost_analysis['total_cost'] if cost_analysis['total_cost'] > 0 else 0

    return G, total_time_cost, new_roads_added, total_distance, connectivity_score, cost_analysis

def generate_cost_report(new_roads_added, potential_roads, neighborhoods, facilities):
    """
    Generate a detailed cost report for new roads added to the network.
    
    Args:
        new_roads_added: List of tuples (from, to, cost)
        potential_roads: DataFrame of potential roads
        neighborhoods: DataFrame of neighborhoods
        facilities: DataFrame of facilities
    
    Returns:
        DataFrame: Cost report with road details
    """
    report_data = []
    for from_id, to_id, cost in new_roads_added:
        # Get road details
        road_data = potential_roads[
            ((potential_roads['FromID'] == from_id) & (potential_roads['ToID'] == to_id)) |
            ((potential_roads['FromID'] == to_id) & (potential_roads['ToID'] == from_id))
        ]
        distance = road_data['Distance(km)'].iloc[0] if not road_data.empty else 0
        capacity = road_data['Estimated Capacity(vehicles/hour)'].iloc[0] if not road_data.empty else 3000

        # Get node names
        from_name = neighborhoods[neighborhoods['ID'] == from_id]['Name'].iloc[0] if from_id in neighborhoods['ID'].values else None
        from_name = facilities[facilities['ID'] == from_id]['Name'].iloc[0] if from_id in facilities['ID'].values else from_name
        to_name = neighborhoods[neighborhoods['ID'] == to_id]['Name'].iloc[0] if to_id in neighborhoods['ID'].values else \
                 facilities[facilities['ID'] == to_id]['Name'].iloc[0]

        report_data.append({
            'From': f"{from_id} ({from_name})",
            'To': f"{to_id} ({to_name})",
            'Distance (km)': distance,
            'Capacity (veh/h)': capacity,
            'Cost (Million EGP)': cost,
            'Cost per km (Million EGP)': cost / distance if distance > 0 else 0
        })

    return pd.DataFrame(report_data)
