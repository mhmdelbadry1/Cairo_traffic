import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from data_loader import load_data
from infrastructure import create_mst_network, generate_cost_report
from traffic_optimization import optimize_traffic_flow
from emergency_response import plan_emergency_routes
from traffic_signals import identify_intersections, real_time_signal_optimization, emergency_vehicle_preemption
from visualization import (
    plot_network_graph, 
    plot_map, 
    plot_traffic_comparison, 
    plot_route_visualization,
    plot_public_transit_routes
)
from public_transit import optimize_public_transit, optimize_schedule_dp, analyze_transfer_points, design_integrated_network

# Set page config
st.set_page_config(
    page_title="Cairo Traffic Optimization System",
    page_icon="🚗",
    layout="wide"
)

# Main title and description
st.title("Cairo Traffic Optimization System")
st.markdown("""
This system implements multiple graph algorithms to optimize traffic in Cairo:
* **Infrastructure Network Design** - Minimum Spanning Tree algorithms
* **Traffic Flow Optimization** - Dijkstra's algorithm with time modifications
* **Emergency Response Planning** - A* search algorithm with preemption
* **Public Transit Optimization** - Dynamic programming
* **Traffic Signal Optimization** - Greedy approach with emergency vehicle preemption
""")

# Load data
neighborhoods, facilities, existing_roads, potential_roads, traffic_flow, metro_lines, bus_routes, transit_demand = load_data()

# Sidebar for selecting the module
st.sidebar.title("Navigation")
module = st.sidebar.radio(
    "Select Module",
    ["Overview", "Infrastructure Network Design", "Traffic Flow Optimization", 
     "Emergency Response Planning", "Public Transit Optimization", "Greedy Traffic Signal Control"]
)

# Overview module
if module == "Overview":
    st.header("Cairo Traffic System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Geographic Distribution")
        st.markdown("Map of neighborhoods, districts, and key facilities in Cairo")
        
        folium_map = plot_map(neighborhoods, facilities, existing_roads)
        folium_static(folium_map, width=600, height=400)
        
        st.subheader("Statistics")
        
        # Basic statistics
        total_population = neighborhoods['Population'].sum()
        total_roads = len(existing_roads)
        total_facilities = len(facilities)
        
        st.metric("Total Population", f"{total_population:,}")
        st.metric("Existing Roads", total_roads)
        st.metric("Key Facilities", total_facilities)
    
    with col2:
        st.subheader("Road Network")
        
        # Create a network graph
        G = nx.Graph()
        
        # Add neighborhoods and facilities as nodes
        for _, row in neighborhoods.iterrows():
            G.add_node(row['ID'], 
                       name=row['Name'], 
                       population=row['Population'],
                       pos=(row['X'], row['Y']),
                       node_type='neighborhood')
        
        for _, row in facilities.iterrows():
            G.add_node(row['ID'], 
                       name=row['Name'], 
                       pos=(row['X'], row['Y']),
                       node_type='facility')
        
        # Add existing roads as edges
        for _, row in existing_roads.iterrows():
            G.add_edge(row['FromID'], row['ToID'], 
                       weight=row['Distance(km)'],
                       capacity=row['Current Capacity(vehicles/hour)'],
                       condition=row['Condition(1-10)'])
        
        fig = plot_network_graph(G)
        st.plotly_chart(fig, use_container_width=True)
        
        # Road condition distribution
        st.subheader("Road Condition Distribution")
        condition_counts = existing_roads['Condition(1-10)'].value_counts().sort_index()
        
        fig = px.bar(x=condition_counts.index, 
                     y=condition_counts.values,
                     labels={'x': 'Condition (1-10)', 'y': 'Number of Roads'},
                     color=condition_counts.index,
                     color_continuous_scale='RdYlGn')
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Traffic Flow Patterns")
    # Average traffic by time of day
    traffic_by_time = pd.DataFrame({
        'Time Period': ['Morning Peak', 'Afternoon', 'Evening Peak', 'Night'],
        'Average Traffic': [
            traffic_flow['Morning Peak(veh/h)'].mean(),
            traffic_flow['Afternoon(veh/h)'].mean(),
            traffic_flow['Evening Peak(veh/h)'].mean(),
            traffic_flow['Night(veh/h)'].mean()
        ]
    })
    
    fig = px.line(traffic_by_time, x='Time Period', y='Average Traffic', 
                 markers=True, line_shape='linear')
    fig.update_layout(yaxis_title="Average Traffic (vehicles/hour)")
    st.plotly_chart(fig, use_container_width=True)
    
# Infrastructure Network Design module
if module == "Infrastructure Network Design":
    st.header("Infrastructure Network Design")
    st.markdown("""
    This module uses Minimum Spanning Tree (MST) algorithms to design an optimal road network that connects 
    all areas of Cairo while minimizing travel time and construction costs. The algorithm prioritizes high-population 
    areas and critical facilities, ensuring full connectivity within budget constraints.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("MST Parameters")
        
        algorithm = st.selectbox("Algorithm", ["Kruskal's Algorithm", "Prim's Algorithm"], index=0)
        population_weight = st.slider(
            "Population Weight Factor", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            help="Higher values prioritize connections to high-population areas"
        )
        
        facility_priority = st.slider(
            "Critical Facility Priority", 
            min_value=1.0, 
            max_value=3.0, 
            value=1.5,
            help="Higher values prioritize connections to critical facilities like hospitals"
        )
        
        include_new_roads = st.checkbox("Include Potential New Roads", value=True)
        
        max_budget = st.number_input(
            "Max Budget (Million EGP)", 
            min_value=0, 
            max_value=10000, 
            value=5000,
            help="Maximum budget for new road construction"
        ) if include_new_roads else None
            
        st.write("---")
        
        run_button = st.button("Run MST Analysis")
    
    with col1:
        if run_button:
            with st.spinner("Calculating optimal road network..."):
                # Run MST algorithm
                mst_result = create_mst_network(
                    neighborhoods, 
                    facilities, 
                    existing_roads, 
                    potential_roads if include_new_roads else None,
                    algorithm_choice="kruskal" if "Kruskal" in algorithm else "prim",
                    population_weight=population_weight,
                    facility_priority=facility_priority,
                    max_budget=max_budget
                )
                
                # Unpack results
                optimized_network, total_time_cost, new_roads_added, total_distance, connectivity_score, cost_analysis = mst_result
                
                # Display the optimized network
                st.subheader("Optimized Road Network")
                optimized_map = plot_map(
                    neighborhoods, 
                    facilities, 
                    existing_roads,
                    new_roads=[(r[0], r[1]) for r in new_roads_added] if include_new_roads else None
                )
                folium_static(optimized_map, width=700, height=500)
                
                # Display network metrics
                st.subheader("Network Analysis Results")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Total Travel Time Cost", f"{total_time_cost:.2f} minutes")
                with metric_col2:
                    st.metric("Total Network Distance", f"{total_distance:.2f} km")
                with metric_col3:
                    st.metric("Connectivity Score", f"{connectivity_score:.2f}")
                
                # Display cost analysis
                st.subheader("Cost Analysis")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Total Cost (Million EGP)", f"{cost_analysis['total_cost']:.2f}")
                with metric_col2:
                    st.metric("Cost per km (Million EGP)", f"{cost_analysis['cost_per_km']:.2f}")
                with metric_col3:
                    st.metric("Cost Effectiveness", f"{cost_analysis['cost_effectiveness']:.4f}")
                
                # Display new roads table if any were added
                if include_new_roads and len(new_roads_added) > 0:
                    st.subheader("Recommended New Roads")
                    cost_report = generate_cost_report(new_roads_added, potential_roads, neighborhoods, facilities)
                    st.dataframe(cost_report)
                    
                    # Cost distribution chart
                    st.subheader("Cost Distribution")
                    fig = px.bar(
                        cost_report,
                        x='From',
                        y='Cost (Million EGP)',
                        text='Cost (Million EGP)',
                        title="Cost of New Roads"
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Adjust the parameters and click 'Run MST Analysis' to see the results.")
            # Show the existing road network
            st.subheader("Current Road Network")
            current_map = plot_map(neighborhoods, facilities, existing_roads)
            folium_static(current_map, width=700, height=500)


elif module == "Traffic Flow Optimization":
    st.header("Traffic Flow Optimization")
    st.markdown("""
    This module uses Dijkstra's algorithm with time-dependent modifications to analyze and optimize 
    traffic flow based on different times of the day.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Route Parameters")
        
        origin_options = []
        for _, row in neighborhoods.iterrows():
            origin_options.append(f"{row['ID']} - {row['Name']}")
        for _, row in facilities.iterrows():
            origin_options.append(f"{row['ID']} - {row['Name']}")
            
        origin = st.selectbox("Origin", origin_options)
        destination = st.selectbox("Destination", origin_options, index=min(1, len(origin_options)-1))
        
        origin_id = origin.split(" - ")[0]
        destination_id = destination.split(" - ")[0]
        
        time_period = st.selectbox(
            "Time Period", 
            ["Morning Peak", "Afternoon", "Evening Peak", "Night"],
            index=0
        )
        
        consider_traffic = st.checkbox("Consider Traffic Conditions", value=True)
        congestion_factor = 1.5
        if consider_traffic:
            congestion_factor = st.slider(
                "Congestion Factor", 
                min_value=1.0, 
                max_value=3.0, 
                value=1.5,
                help="Higher values reflect heavier congestion impact"
            )
        
        consider_road_quality = st.checkbox("Consider Road Quality", value=True)
        
        st.write("---")
        
        find_route_button = st.button("Find Optimal Route")
    
    with col1:
        if find_route_button:
            with st.spinner("Finding optimal routes..."):
                time_column = {
                    "Morning Peak": "Morning Peak(veh/h)",
                    "Afternoon": "Afternoon(veh/h)",
                    "Evening Peak": "Evening Peak(veh/h)",
                    "Night": "Night(veh/h)"
                }[time_period]
                
                optimal_route_results = optimize_traffic_flow(
                    neighborhoods, 
                    facilities, 
                    existing_roads, 
                    traffic_flow,
                    origin_id, 
                    destination_id,
                    time_column=time_column,
                    consider_traffic=consider_traffic,
                    congestion_factor=congestion_factor,
                    consider_road_quality=consider_road_quality
                )
                
                optimal_route, optimal_distance, optimal_time, alternative_route, alt_distance, alt_time, error_message = optimal_route_results
                
                if error_message:
                    st.error(error_message)
                else:
                    st.subheader("Optimal Route Visualization")
                    route_map = plot_route_visualization(
                        neighborhoods, 
                        facilities, 
                        existing_roads,
                        optimal_route,
                        alternative_route
                    )
                    folium_static(route_map, width=700, height=500)
                    
                    st.subheader("Route Analysis")
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Optimal Route Distance", f"{optimal_distance:.2f} km")
                        st.metric("Estimated Travel Time", f"{optimal_time:.2f} min")
                    
                    with metric_col2:
                        st.metric("Alternative Route Distance", f"{alt_distance:.2f} km")
                        st.metric("Alternative Travel Time", f"{alt_time:.2f} min")
                    
                    st.subheader(f"Traffic Comparison ({time_period})")
                    traffic_chart = plot_traffic_comparison(
                        traffic_flow,
                        optimal_route,
                        alternative_route,
                        time_column
                    )
                    st.plotly_chart(traffic_chart, use_container_width=True)
                    
                    st.subheader("Route Directions")
                    tab1, tab2 = st.tabs(["Optimal Route", "Alternative Route"])
                    
                    with tab1:
                        optimal_steps = []
                        for i in range(len(optimal_route) - 1):
                            from_id = optimal_route[i]
                            to_id = optimal_route[i+1]
                            from_name = neighborhoods[neighborhoods['ID'] == from_id]['Name'].iloc[0] if from_id in neighborhoods['ID'].values \
                                       else facilities[facilities['ID'] == from_id]['Name'].iloc[0]
                            to_name = neighborhoods[neighborhoods['ID'] == to_id]['Name'].iloc[0] if to_id in neighborhoods['ID'].values \
                                     else facilities[facilities['ID'] == to_id]['Name'].iloc[0]
                            road_data = existing_roads[
                                ((existing_roads['FromID'] == from_id) & (existing_roads['ToID'] == to_id)) |
                                ((existing_roads['FromID'] == to_id) & (existing_roads['ToID'] == from_id))
                            ].iloc[0]
                            optimal_steps.append(f"From {from_name} to {to_name} ({road_data['Distance(km)']:.1f} km)")
                        
                        for step in optimal_steps:
                            st.markdown(f"- {step}")
                    
                    with tab2:
                        alt_steps = []
                        for i in range(len(alternative_route) - 1):
                            from_id = alternative_route[i]
                            to_id = alternative_route[i+1]
                            from_name = neighborhoods[neighborhoods['ID'] == from_id]['Name'].iloc[0] if from_id in neighborhoods['ID'].values \
                                       else facilities[facilities['ID'] == from_id]['Name'].iloc[0]
                            to_name = neighborhoods[neighborhoods['ID'] == to_id]['Name'].iloc[0] if to_id in neighborhoods['ID'].values \
                                     else facilities[facilities['ID'] == to_id]['Name'].iloc[0]
                            road_data = existing_roads[
                                ((existing_roads['FromID'] == from_id) & (existing_roads['ToID'] == to_id)) |
                                ((existing_roads['FromID'] == to_id) & (existing_roads['ToID'] == from_id))
                            ].iloc[0]
                            alt_steps.append(f"From {from_name} to {to_name} ({road_data['Distance(km)']:.1f} km)")
                        
                        for step in alt_steps:
                            st.markdown(f"- {step}")
                
        else:
            st.info("Select origin, destination, and parameters, then click 'Find Optimal Route' to see the results.")
            st.subheader("Traffic Flow Map")
            traffic_map = plot_map(
                neighborhoods, 
                facilities, 
                existing_roads, 
                show_traffic=True, 
                traffic_data=traffic_flow
            )
            folium_static(traffic_map, width=700, height=500)
# Emergency Response Planning module
elif module == "Emergency Response Planning":
    st.header("Emergency Response Planning")
    st.markdown("""
    This module uses the A* search algorithm with signal preemption to plan emergency response routes 
    that minimize response time in heavy traffic conditions.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Emergency Parameters")
        
        emergency_type = st.selectbox(
            "Emergency Type", 
            ["Medical Emergency", "Fire", "Police"],
            index=0
        )
        
        facility_options = [f"{row['ID']} - {row['Name']}" for _, row in facilities.iterrows()]
        default_facility_index = next(
            (i for i, facility in enumerate(facility_options) if "Hospital" in facility and emergency_type == "Medical Emergency"),
            0
        )
        emergency_origin = st.selectbox("Emergency Response Origin", facility_options, index=default_facility_index)
        
        neighborhood_options = [f"{row['ID']} - {row['Name']}" for _, row in neighborhoods.iterrows()]
        incident_location = st.selectbox("Incident Location", neighborhood_options)
        
        origin_id = emergency_origin.split(" - ")[0]
        destination_id = incident_location.split(" - ")[0]
        
        time_period = st.selectbox(
            "Time Period", 
            ["Morning Peak", "Afternoon", "Evening Peak", "Night"],
            index=0
        )
        time_column = {
            "Morning Peak": "Morning Peak(veh/h)",
            "Afternoon": "Afternoon(veh/h)",
            "Evening Peak": "Evening Peak(veh/h)",
            "Night": "Night(veh/h)"
        }[time_period]
        
        priority_level = st.slider("Priority Level", min_value=1, max_value=5, value=3)
        route_clearing = st.checkbox("Enable Route Clearing", value=True)
        
        find_route_button = st.button("Find Emergency Route")
    
    with col1:
        if find_route_button:
            with st.spinner("Planning emergency response route..."):
                # Identify intersections and optimize signals
                intersections = identify_intersections(existing_roads, neighborhoods, facilities)
                signals = real_time_signal_optimization(intersections, traffic_flow, existing_roads, time_column)
                
                # Plan emergency route with preemption
                results = plan_emergency_routes(
                    neighborhoods, facilities, existing_roads, traffic_flow,
                    origin_id, destination_id, time_column, priority_level,
                    route_clearing, emergency_type, intersections, signals
                )
                
                (emergency_route, regular_route, emergency_time, regular_time, time_saved_route, 
                 distance, preemption_plan, time_saved_signals) = results
                
                # Display route on map
                st.subheader("Emergency Route Visualization")
                emergency_map = plot_route_visualization(
                    neighborhoods, facilities, existing_roads,
                    emergency_route, regular_route, is_emergency=True
                )
                folium_static(emergency_map, width=700, height=500)
                
                # Display metrics
                st.subheader("Emergency Response Analysis")
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Emergency Route Time", f"{emergency_time:.1f} min", delta=f"-{time_saved_route:.1f} min")
                with col_stats2:
                    st.metric("Regular Route Time", f"{regular_time:.1f} min")
                with col_stats3:
                    total_time_saved = time_saved_route + (time_saved_signals / 60)
                    st.metric("Total Time Saved", f"{total_time_saved:.1f} min")
                
                # Display preemption details
                st.subheader("Signal Preemption Details")
                preemption_df = []
                for node_id, plan in preemption_plan.items():
                    name = intersections[node_id]['name'] if node_id in intersections else "Unknown"
                    preemption_df.append({
                        'Intersection': f"{node_id} - {name}",
                        'Normal Wait Time': f"{plan['time_saved']:.1f} sec" if plan['preemption_effective'] else "N/A",
                        'Preemption Wait': "0.0 sec" if plan['preemption_effective'] else "N/A",
                        'Time Saved': f"{plan['time_saved']:.1f} sec" if plan['preemption_effective'] else "0.0 sec",
                        'Status': "Active Preemption" if plan['preemption_effective'] else plan.get('reason', 'No Preemption')
                    })
                
                if preemption_df:
                    st.dataframe(pd.DataFrame(preemption_df), use_container_width=True)
                else:
                    st.info("No intersections required preemption on this route.")
                
                # Display route steps
                st.subheader("Emergency Route Directions")
                tab1, tab2 = st.tabs(["Emergency Route", "Regular Route"])
                
                with tab1:
                    emergency_steps = []
                    for i in range(len(emergency_route) - 1):
                        from_id, to_id = emergency_route[i], emergency_route[i+1]
                        from_name = neighborhoods[neighborhoods['ID'] == from_id]['Name'].iloc[0] if from_id in neighborhoods['ID'].values \
                                   else facilities[facilities['ID'] == from_id]['Name'].iloc[0]
                        to_name = neighborhoods[neighborhoods['ID'] == to_id]['Name'].iloc[0] if to_id in neighborhoods['ID'].values \
                                 else facilities[facilities['ID'] == to_id]['Name'].iloc[0]
                        road_data = existing_roads[
                            ((existing_roads['FromID'] == from_id) & (existing_roads['ToID'] == to_id)) |
                            ((existing_roads['FromID'] == to_id) & (existing_roads['ToID'] == from_id))
                        ].iloc[0]
                        route_clearing_note = " (with route clearing)" if route_clearing and i < 2 else ""
                        emergency_steps.append(f"From {from_name} to {to_name} ({road_data['Distance(km)']:.1f} km){route_clearing_note}")
                    
                    for step in emergency_steps:
                        st.markdown(f"- {step}")
                
                with tab2:
                    regular_steps = []
                    for i in range(len(regular_route) - 1):
                        from_id, to_id = regular_route[i], regular_route[i+1]
                        from_name = neighborhoods[neighborhoods['ID'] == from_id]['Name'].iloc[0] if from_id in neighborhoods['ID'].values \
                                   else facilities[facilities['ID'] == from_id]['Name'].iloc[0]
                        to_name = neighborhoods[neighborhoods['ID'] == to_id]['Name'].iloc[0] if to_id in neighborhoods['ID'].values \
                                 else facilities[facilities['ID'] == to_id]['Name'].iloc[0]
                        road_data = existing_roads[
                            ((existing_roads['FromID'] == from_id) & (existing_roads['ToID'] == to_id)) |
                            ((existing_roads['FromID'] == to_id) & (existing_roads['ToID'] == from_id))
                        ].iloc[0]
                        regular_steps.append(f"From {from_name} to {to_name} ({road_data['Distance(km)']:.1f} km)")
                    
                    for step in regular_steps:
                        st.markdown(f"- {step}")
                
                # Critical intersections
                st.subheader("Critical Intersections for Priority")
                critical_nodes = [emergency_route[i] for i in range(1, min(len(emergency_route) - 1, 3))]
                for i, node_id in enumerate(critical_nodes):
                    node_name = neighborhoods[neighborhoods['ID'] == node_id]['Name'].iloc[0] if node_id in neighborhoods['ID'].values \
                               else facilities[facilities['ID'] == node_id]['Name'].iloc[0]
                    st.markdown(f"{i+1}. **{node_id} - {node_name}**")
        else:
            st.info("Select emergency parameters, then click 'Find Emergency Route' to see the results.")
            st.subheader("Emergency Facilities and Access Routes")
            emergency_map = plot_map(neighborhoods, facilities, existing_roads, highlight_emergency=True)
            folium_static(emergency_map, width=700, height=500)

# Public Transit Optimization module
elif module == "Public Transit Optimization":
    st.header("Public Transit Optimization")
    st.markdown("""
    This module uses dynamic programming to optimize public transportation routes and schedules,
    considering passenger demand and resource allocation. It also includes tools to design an integrated network.
    """)
    
    tabs = st.tabs(["Route Analysis", "Schedule Optimization", "Transit Demand", "Integrated Network Design"])
    
    with tabs[0]:  # Route Analysis
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Transit Parameters")
            
            transit_type = st.selectbox(
                "Transit Type",
                ["Metro", "Bus", "All"],
                index=0
            )
            
            # Options based on transit type
            route_options = []
            if transit_type == "Metro" or transit_type == "All":
                for _, row in metro_lines.iterrows():
                    route_options.append(f"Metro - {row['LineID']}: {row['Name']}")
            
            if transit_type == "Bus" or transit_type == "All":
                for _, row in bus_routes.iterrows():
                    route_options.append(f"Bus - {row['RouteID']}")
            
            selected_route = st.selectbox("Select Route", route_options)
            
            # Parameters for optimization
            optimize_for = st.selectbox(
                "Optimize For",
                ["Passenger Capacity", "Travel Time", "Resource Efficiency"],
                index=0
            )
            
            if "Bus" in selected_route:
                bus_count = st.slider(
                    "Number of Buses", 
                    min_value=5, 
                    max_value=40, 
                    value=20
                )
            
            analyze_button = st.button("Analyze Route")
        
        with col1:
            if analyze_button:
                with st.spinner("Analyzing transit route..."):
                    # Process the selected route
                    route_id = selected_route.split(" - ")[1].split(":")[0]
                    
                    # Get route data
                    if "Metro" in selected_route:
                        route_data = metro_lines[metro_lines['LineID'] == route_id].iloc[0]
                        route_type = "metro"
                        stations = route_data['Stations(comma-separated IDs)'].strip('"').split(',')
                        passengers = route_data['Daily Passengers']
                    else:  # Bus
                        route_data = bus_routes[bus_routes['RouteID'] == route_id].iloc[0]
                        route_type = "bus"
                        stations = route_data['Stops(comma-separated IDs)'].strip('"').split(',')
                        passengers = route_data['Daily Passengers']
                    
                    # Analyze the route
                    route_analysis = optimize_public_transit(
                        neighborhoods,
                        facilities,
                        existing_roads,
                        metro_lines,
                        bus_routes,
                        transit_demand,
                        route_id,
                        route_type,
                        optimize_for=optimize_for
                    )
                    
                    # Unpack results
                    optimized_route, current_performance, optimized_performance, suggested_changes = route_analysis
                    
                    # Display route map
                    st.subheader(f"Transit Route: {selected_route}")
                    
                    transit_map = plot_public_transit_routes(
                        neighborhoods,
                        facilities,
                        existing_roads,
                        [stations],  # Current route
                        [optimized_route] if optimized_route else []  # Optimized route
                    )
                    folium_static(transit_map, width=700, height=500)
                    
                    # Performance metrics
                    st.subheader("Route Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Current Performance**")
                        st.metric("Daily Passengers", f"{passengers:,}")
                        st.metric("Stops/Stations", len(stations))
                        
                        if route_type == "bus":
                            bus_count = bus_routes[bus_routes['RouteID'] == route_id].iloc[0]['Buses Assigned']
                            st.metric("Buses Assigned", bus_count)
                            st.metric("Passengers per Bus", f"{passengers / bus_count:.1f}")
                    
                    with col2:
                        st.markdown("**Optimized Performance**")
                        passengers_diff = optimized_performance.get('passengers', passengers) - passengers
                        st.metric("Projected Passengers", f"{optimized_performance.get('passengers', passengers):,}", delta=f"{passengers_diff:,}")
                        
                        if optimized_route:
                            st.metric("Recommended Stops", len(optimized_route), delta=len(optimized_route) - len(stations))
                        
                        if route_type == "bus" and "resources" in optimized_performance:
                            bus_count = bus_routes[bus_routes['RouteID'] == route_id].iloc[0]['Buses Assigned']
                            st.metric("Recommended Buses", optimized_performance['resources'], delta=int(optimized_performance['resources'] - bus_count))
                            st.metric("Projected Passengers per Bus", f"{optimized_performance.get('passengers', passengers) / optimized_performance['resources']:.1f}")
                    # Suggested changes
                    if suggested_changes:
                        st.subheader("Recommended Changes")
                        
                        for change in suggested_changes:
                            st.markdown(f"- {change}")
                    
                    # Route details
                    st.subheader("Route Details")
                    
                    # Create a table of stops/stations with details
                    stop_details = []
                    
                    for stop_id in stations:
                        stop_name = ""
                        stop_type = ""
                        population = 0
                        
                        if stop_id in neighborhoods['ID'].values:
                            stop_name = neighborhoods[neighborhoods['ID'] == stop_id]['Name'].iloc[0]
                            stop_type = "Neighborhood"
                            population = neighborhoods[neighborhoods['ID'] == stop_id]['Population'].iloc[0]
                        elif stop_id in facilities['ID'].values:
                            stop_name = facilities[facilities['ID'] == stop_id]['Name'].iloc[0]
                            stop_type = facilities[facilities['ID'] == stop_id]['Type'].iloc[0]
                        
                        # Calculate boarding/alighting passengers (simplified)
                        boarding = sum([row['Daily Passengers'] for _, row in transit_demand.iterrows() 
                                     if row['FromID'] == stop_id]) / 10  # Divide for simplification
                        
                        alighting = sum([row['Daily Passengers'] for _, row in transit_demand.iterrows() 
                                      if row['ToID'] == stop_id]) / 10  # Divide for simplification
                        
                        stop_details.append({
                            'Stop ID': stop_id,
                            'Name': stop_name,
                            'Type': stop_type,
                            'Population': population if population > 0 else "-",
                            'Est. Boarding': int(boarding),
                            'Est. Alighting': int(alighting)
                        })
                    
                    stop_details_df = pd.DataFrame(stop_details)
                    st.dataframe(stop_details_df)
            else:
                st.info("Select a transit route and parameters, then click 'Analyze Route' to see the results.")
                # Show transit network map
                st.subheader("Public Transit Network")
                
                # Extract all metro and bus routes
                metro_routes_list = []
                for _, row in metro_lines.iterrows():
                    stations = row['Stations(comma-separated IDs)'].strip('"').split(',')
                    metro_routes_list.append(stations)
                
                bus_routes_list = []
                for _, row in bus_routes.iterrows():
                    stops = row['Stops(comma-separated IDs)'].strip('"').split(',')
                    bus_routes_list.append(stops)
                
                transit_map = plot_public_transit_routes(
                    neighborhoods,
                    facilities,
                    existing_roads,
                    metro_routes_list,
                    bus_routes_list
                )
                folium_static(transit_map, width=700, height=500)
    
    with tabs[1]:  # Schedule Optimization
        st.subheader("Transit Schedule Optimization")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Schedule Parameters")
            
            transit_line = st.selectbox(
                "Transit Line",
                ["M1 - Line 1 (Helwan-New Marg)", 
                 "M2 - Line 2 (Shubra-Giza)", 
                 "M3 - Line 3 (Airport-Imbaba)"],
                index=0
            )
            
            day_type = st.selectbox(
                "Day Type",
                ["Weekday", "Weekend", "Holiday"],
                index=0
            )
            
            peak_hours = st.multiselect(
                "Peak Hours",
                ["6-9 AM", "9-12 PM", "12-3 PM", "3-6 PM", "6-9 PM", "9-12 AM"],
                default=["6-9 AM", "3-6 PM"]
            )
            
            resource_constraint = st.slider(
                "Resource Constraint (%)",
                min_value=70,
                max_value=130,
                value=100,
                help="Percentage of available transit vehicles"
            )
            
            optimize_schedule_button = st.button("Optimize Schedule")
        
        with col1:
            if optimize_schedule_button:
                with st.spinner("Optimizing transit schedule..."):
                    # Get the line ID
                    line_id = transit_line.split(" - ")[0]
                    
                    # Use the new optimize_schedule_dp function
                    current_schedule, optimized_schedule = optimize_schedule_dp(
                        route_id=line_id,
                        route_type="metro",
                        metro_lines=metro_lines,
                        bus_routes=bus_routes,
                        transit_demand=transit_demand,
                        peak_hours=peak_hours,
                        resource_constraint=resource_constraint
                    )
                    
                    # Display schedules
                    st.subheader(f"Schedule Optimization for {transit_line}")
                    
                    tab1, tab2 = st.tabs(["Current Schedule", "Optimized Schedule"])
                    
                    with tab1:
                        current_schedule_df = pd.DataFrame(current_schedule)
                        st.dataframe(current_schedule_df)
                        
                        # Chart for current schedule
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=current_schedule_df["Time Period"],
                            y=current_schedule_df["Capacity (passengers/hour)"],
                            mode='lines+markers',
                            name='Capacity',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=current_schedule_df["Time Period"],
                            y=current_schedule_df["Resources in Service"],
                            name='Resources',
                            marker_color='lightblue',
                            opacity=0.7,
                            yaxis='y2'
                        ))
                        
                        fig.update_layout(
                            title="Current Schedule - Capacity and Resource Count",
                            xaxis_title="Time Period",
                            yaxis_title="Capacity (passengers/hour)",
                            yaxis2=dict(
                                title="Resources in Service",
                                overlaying='y',
                                side='right'
                            ),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        optimized_schedule_df = pd.DataFrame(optimized_schedule)
                        st.dataframe(optimized_schedule_df)
                        
                        # Chart for optimized schedule
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=optimized_schedule_df["Time Period"],
                            y=optimized_schedule_df["Capacity (passengers/hour)"],
                            mode='lines+markers',
                            name='Capacity',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=optimized_schedule_df["Time Period"],
                            y=optimized_schedule_df["Resources in Service"],
                            name='Resources',
                            marker_color='lightgreen',
                            opacity=0.7,
                            yaxis='y2'
                        ))
                        
                        fig.update_layout(
                            title="Optimized Schedule - Capacity and Resource Count",
                            xaxis_title="Time Period",
                            yaxis_title="Capacity (passengers/hour)",
                            yaxis2=dict(
                                title="Resources in Service",
                                overlaying='y',
                                side='right'
                            ),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Comparison of schedules
                    st.subheader("Schedule Comparison")
                    
                    # Calculate metrics
                    current_total_capacity = current_schedule_df["Capacity (passengers/hour)"].sum()
                    optimized_total_capacity = optimized_schedule_df["Capacity (passengers/hour)"].sum()
                    
                    current_total_resources = current_schedule_df["Resources in Service"].sum()
                    optimized_total_resources = optimized_schedule_df["Resources in Service"].sum()
                    
                    current_efficiency = current_total_capacity / current_total_resources if current_total_resources > 0 else 0
                    optimized_efficiency = optimized_total_capacity / optimized_total_resources if optimized_total_resources > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Daily Capacity", 
                            f"{optimized_total_capacity:,}", 
                            delta=f"{optimized_total_capacity - current_total_capacity:,}"
                        )
                    
                    with col2:
                        st.metric(
                            "Total Resource-Hours", 
                            f"{optimized_total_resources}", 
                            delta=f"{optimized_total_resources - current_total_resources}"
                        )
                    
                    with col3:
                        st.metric(
                            "Efficiency (Capacity/Resource)", 
                            f"{optimized_efficiency:.1f}", 
                            delta=f"{optimized_efficiency - current_efficiency:.1f}"
                        )
                    
                    # Comparison chart
                    comparison_df = pd.DataFrame({
                        "Time Period": current_schedule_df["Time Period"],
                        "Current Capacity": current_schedule_df["Capacity (passengers/hour)"],
                        "Optimized Capacity": optimized_schedule_df["Capacity (passengers/hour)"],
                        "Current Resources": current_schedule_df["Resources in Service"],
                        "Optimized Resources": optimized_schedule_df["Resources in Service"]
                    })
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=comparison_df["Time Period"],
                        y=comparison_df["Current Capacity"],
                        mode='lines',
                        name='Current Capacity',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=comparison_df["Time Period"],
                        y=comparison_df["Optimized Capacity"],
                        mode='lines',
                        name='Optimized Capacity',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Capacity Comparison - Current vs. Optimized",
                        xaxis_title="Time Period",
                        yaxis_title="Capacity (passengers/hour)",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select schedule parameters and click 'Optimize Schedule' to see the results.")
                
                # Display sample schedule data
                st.subheader("Transit Line Information")
                
                metro_data = pd.DataFrame({
                    "Line": ["M1 - Line 1", "M2 - Line 2", "M3 - Line 3"],
                    "Route": ["Helwan-New Marg", "Shubra-Giza", "Airport-Imbaba"],
                    "Daily Passengers": ["1,500,000", "1,200,000", "800,000"],
                    "Stations": [5, 5, 5],
                    "Length (km)": [44, 21, 34]
                })
                
                st.dataframe(metro_data)
                
                # Simple chart of daily passengers by line
                fig = px.bar(
                    x=["M1 - Line 1", "M2 - Line 2", "M3 - Line 3"],
                    y=[1500000, 1200000, 800000],
                    labels={'x': 'Metro Line', 'y': 'Daily Passengers'},
                    color=[1500000, 1200000, 800000],
                    text=[1500000, 1200000, 800000],
                    color_continuous_scale='Blues'
                )
                
                fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                fig.update_layout(title="Daily Passengers by Metro Line")
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:  # Transit Demand
        st.subheader("Transit Demand Analysis")
        
        # Create a heatmap of transit demand between neighborhoods
        
        # Filter and process demand data
        demand_matrix = pd.pivot_table(
            transit_demand,
            values='Daily Passengers',
            index='FromID',
            columns='ToID',
            fill_value=0
        )
        
        # Get names for nodes
        node_names = {}
        for _, row in neighborhoods.iterrows():
            node_names[row['ID']] = row['Name']
        for _, row in facilities.iterrows():
            node_names[row['ID']] = row['Name']
        
        # Replace IDs with names where possible
        demand_matrix_named = demand_matrix.copy()
        demand_matrix_named.index = [node_names.get(idx, idx) for idx in demand_matrix_named.index]
        demand_matrix_named.columns = [node_names.get(col, col) for col in demand_matrix_named.columns]
        
        # Create heatmap
        fig = px.imshow(
            demand_matrix_named,
            labels=dict(x="Destination", y="Origin", color="Daily Passengers"),
            color_continuous_scale="blues"
        )
        
        fig.update_layout(
            title="Transit Demand Heat Map",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top demand routes
        st.subheader("Top Demand Routes")
        
        # Create a flat dataframe of all demands
        flat_demand = []
        for from_id in demand_matrix.index:
            for to_id in demand_matrix.columns:
                passengers = demand_matrix.loc[from_id, to_id]
                if passengers > 0:
                    from_name = node_names.get(from_id, from_id)
                    to_name = node_names.get(to_id, to_id)
                    flat_demand.append({
                        'Origin': f"{from_id} - {from_name}",
                        'Destination': f"{to_id} - {to_name}",
                        'Daily Passengers': passengers
                    })
        
        flat_demand_df = pd.DataFrame(flat_demand)
        flat_demand_df = flat_demand_df.sort_values('Daily Passengers', ascending=False)
        
        st.dataframe(flat_demand_df.head(10))
        
        # Bar chart of top routes
        top_routes = flat_demand_df.head(10)
        
        fig = px.bar(
            top_routes,
            x='Daily Passengers',
            y='Origin',
            orientation='h',
            color='Daily Passengers',
            color_continuous_scale='blues',
            text='Daily Passengers'
        )
        
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            title="Top 10 Transit Demand Routes",
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Coverage analysis
        st.subheader("Transit Coverage Analysis")
        
        # Create a list of all bus and metro stops
        all_transit_stops = set()
        
        for _, row in metro_lines.iterrows():
            stations = row['Stations(comma-separated IDs)'].strip('"').split(',')
            all_transit_stops.update(stations)
        
        for _, row in bus_routes.iterrows():
            stops = row['Stops(comma-separated IDs)'].strip('"').split(',')
            all_transit_stops.update(stops)
        
        # Calculate coverage statistics
        total_population = neighborhoods['Population'].sum()
        covered_population = 0
        
        for _, row in neighborhoods.iterrows():
            if row['ID'] in all_transit_stops:
                covered_population += row['Population']
        
        coverage_percentage = covered_population / total_population * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Population", f"{total_population:,}")
            st.metric("Covered Population", f"{covered_population:,}", delta=f"{coverage_percentage:.1f}%")
        
        with col2:
            # Create a pie chart of coverage
            fig = px.pie(
                values=[covered_population, total_population - covered_population],
                names=["Covered by Transit", "Not Covered"],
                color_discrete_sequence=['#2166ac', '#b2182b']
            )
            
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(title="Transit Coverage")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Transit stop map
        st.subheader("Transit Stop Coverage Map")
        
        transit_coverage_map = plot_map(
            neighborhoods, 
            facilities, 
            existing_roads,
            highlight_transit=True,
            transit_stops=list(all_transit_stops)
        )
        
        folium_static(transit_coverage_map, width=700, height=500)
    
    with tabs[3]:  # Integrated Network Design
        st.subheader("Integrated Network Design")
        st.markdown("""
        Design an integrated public transit network by proposing new routes to connect underserved areas,
        optimizing coverage and connectivity.
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Design Parameters")
            
            max_new_routes = st.slider(
                "Maximum New Routes",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of new routes to propose"
            )
            
            max_budget = st.number_input(
                "Maximum Budget (Million EGP)",
                min_value=100.0,
                max_value=5000.0,
                value=1000.0,
                help="Budget for new route infrastructure"
            )
            
            fleet_availability = {}
            include_fleet = st.checkbox("Specify Fleet Availability", value=False)
            if include_fleet:
                fleet_availability['bus'] = st.number_input(
                    "Available Buses",
                    min_value=0,
                    max_value=100,
                    value=50
                )
            
            design_button = st.button("Design Integrated Network")
        
        with col1:
            # Always display a default map even if button is not pressed
            if not design_button:
                st.subheader("Integrated Transit Network")
                st.write("Click 'Design Integrated Network' to generate an optimized transit network.")
                
                # Create a basic map showing existing transit infrastructure
                metro_routes_list = []
                for _, row in metro_lines.iterrows():
                    stations = row['Stations(comma-separated IDs)'].strip('"').split(',')
                    metro_routes_list.append(stations)
                
                bus_routes_list = []
                for _, row in bus_routes.iterrows():
                    stops = row['Stops(comma-separated IDs)'].strip('"').split(',')
                    bus_routes_list.append(stops)
                
                basic_map = plot_public_transit_routes(
                    neighborhoods,
                    facilities,
                    existing_roads,
                    metro_routes_list,
                    bus_routes_list,
                    proposed_routes=[]
                )
                folium_static(basic_map, width=700, height=500)
                
                # Display default coverage statistics
                st.subheader("Coverage Analysis")
                total_population = neighborhoods['Population'].sum()
                
                # Calculate existing coverage
                covered_nodes = set()
                for _, row in metro_lines.iterrows():
                    stops = row['Stations(comma-separated IDs)'].strip('"').split(',')
                    covered_nodes.update(stops)
                for _, row in bus_routes.iterrows():
                    stops = row['Stops(comma-separated IDs)'].strip('"').split(',')
                    covered_nodes.update(stops)
                
                # Get IDs of neighborhoods with existing stops
                neighborhoods_with_stops = set()
                for stop_id in covered_nodes:
                    if stop_id in neighborhoods['ID'].values:
                        neighborhoods_with_stops.add(stop_id)
                
                covered_population = neighborhoods[neighborhoods['ID'].isin(neighborhoods_with_stops)]['Population'].sum()
                coverage_percentage = (covered_population / total_population * 100)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Population", f"{total_population:,}")
                with col2:
                    st.metric("Covered Population (Before)", f"{covered_population:,}", 
                             delta=f"{coverage_percentage:.1f}%")
                with col3:
                    st.metric("Covered Population (After)", f"{covered_population:,}", 
                             delta=f"{coverage_percentage:.1f}%")
                
                st.subheader("Cost and Recommendations")
                st.write("Run the network design to see cost estimates and recommendations.")
                
            elif design_button:
                with st.spinner("Designing integrated transit network..."):
                    # Extract all metro and bus routes
                    metro_routes_list = []
                    for _, row in metro_lines.iterrows():
                        stations = row['Stations(comma-separated IDs)'].strip('"').split(',')
                        metro_routes_list.append(stations)
                    
                    bus_routes_list = []
                    for _, row in bus_routes.iterrows():
                        stops = row['Stops(comma-separated IDs)'].strip('"').split(',')
                        bus_routes_list.append(stops)
                    
                    # Run the network design
                    new_routes, coverage_stats, recommendations = design_integrated_network(
                        neighborhoods,
                        facilities,
                        existing_roads,
                        metro_lines,
                        bus_routes,
                        transit_demand,
                        max_new_routes=max_new_routes,
                        max_budget=max_budget,
                        fleet_availability=fleet_availability if include_fleet else None
                    )
                    
                    # Display the integrated network map
                    st.subheader("Integrated Transit Network")
                    transit_map = plot_public_transit_routes(
                        neighborhoods,
                        facilities,
                        existing_roads,
                        metro_routes_list,
                        bus_routes_list,
                        proposed_routes=new_routes
                    )
                    folium_static(transit_map, width=700, height=500)
                    
                    # Display coverage statistics
                    st.subheader("Coverage Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Population", f"{coverage_stats['total_population']:,}")
                    with col2:
                        st.metric("Covered Population (Before)", f"{coverage_stats['covered_population_before']:,}", 
                                 delta=f"{coverage_stats['coverage_percentage_before']:.1f}%")
                    with col3:
                        st.metric("Covered Population (After)", f"{coverage_stats['covered_population_after']:,}", 
                                 delta=f"{coverage_stats['coverage_percentage_after']:.1f}%")
                    
                    # Pie chart for coverage comparison
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=["Covered (Before)", "Not Covered (Before)"],
                        values=[coverage_stats['covered_population_before'], 
                               coverage_stats['total_population'] - coverage_stats['covered_population_before']],
                        name="Before",
                        marker_colors=['#2166ac', '#b2182b'],
                        domain={'x': [0, 0.45]}
                    ))
                    fig.add_trace(go.Pie(
                        labels=["Covered (After)", "Not Covered (After)"],
                        values=[coverage_stats['covered_population_after'], 
                               coverage_stats['total_population'] - coverage_stats['covered_population_after']],
                        name="After",
                        marker_colors=['#67a9cf', '#b2182b'],
                        domain={'x': [0.55, 1.0]}
                    ))
                    fig.update_layout(
                        title="Transit Coverage Before vs. After",
                        annotations=[
                            dict(text="Before", x=0.2, y=1.1, showarrow=False),
                            dict(text="After", x=0.8, y=1.1, showarrow=False)
                        ]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display cost and recommendations
                    st.subheader("Cost and Recommendations")
                    st.metric("Total Cost of New Routes", f"{coverage_stats['total_cost']:.1f} Million EGP")
                    
                    if recommendations:
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    
                    # Display proposed routes
                    if new_routes:
                        st.subheader("Proposed Routes")
                        for i, route in enumerate(new_routes, 1):
                            route_names = []
                            for node_id in route:
                                if node_id in neighborhoods['ID'].values:
                                    name = neighborhoods[neighborhoods['ID'] == node_id]['Name'].iloc[0]
                                elif node_id in facilities['ID'].values:
                                    name = facilities[facilities['ID'] == node_id]['Name'].iloc[0]
                                else:
                                    name = node_id
                                route_names.append(name)
                            st.markdown(f"**Route {i}:** {' -> '.join(route_names)}")
            else:
                st.info("Set design parameters and click 'Design Integrated Network' to see the results.")
                
                # Show current transit network
                st.subheader("Current Transit Network")
                metro_routes_list = []
                for _, row in metro_lines.iterrows():
                    stations = row['Stations(comma-separated IDs)'].strip('"').split(',')
                    metro_routes_list.append(stations)
                
                bus_routes_list = []
                for _, row in bus_routes.iterrows():
                    stops = row['Stops(comma-separated IDs)'].strip('"').split(',')
                    bus_routes_list.append(stops)
                
                transit_map = plot_public_transit_routes(
                    neighborhoods,
                    facilities,
                    existing_roads,
                    metro_routes_list,
                    bus_routes_list
                )
                folium_static(transit_map, width=700, height=500)
# Greedy Traffic Signal Control module
elif module == "Greedy Traffic Signal Control":
    st.header("Real-time Traffic Signal Control")
    st.markdown("""
    This module implements a greedy approach for real-time traffic signal optimization at major Cairo intersections, 
    including a priority-based system for emergency vehicle preemption during high congestion periods.
    """)
    
    tabs = st.tabs(["Intersection Analysis", "Signal Optimization", "Emergency Preemption"])
    
    with st.spinner("Identifying major intersections..."):
        intersections = identify_intersections(existing_roads, neighborhoods, facilities)
    
    with tabs[0]:
        st.subheader("Major Intersections Analysis")
        intersection_data = [
            {
                'ID': node_id,
                'Name': intersection['name'],
                'Type': intersection['type'].capitalize(),
                'Connecting Roads': intersection['degree'],
                'Importance Score': round(intersection['importance'], 2)
            }
            for node_id, intersection in intersections.items()
        ]
        intersection_df = pd.DataFrame(intersection_data).sort_values(by='Importance Score', ascending=False).reset_index(drop=True)
        st.dataframe(intersection_df, use_container_width=True)
        
        st.subheader("Intersection Locations")
        m = folium.Map(location=[30.0444, 31.2357], zoom_start=11)
        for node_id, intersection in intersections.items():
            color = 'blue' if intersection['type'] == 'neighborhood' else 'red'
            radius = max(5, min(15, intersection['importance'] * 2))
            folium.CircleMarker(
                location=intersection['pos'][::-1],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{intersection['name']} (ID: {node_id})<br>Importance: {intersection['importance']:.2f}<br>Connecting Roads: {intersection['degree']}"
            ).add_to(m)
        folium_static(m, width=700, height=500)
    
    with tabs[1]:
        st.subheader("Traffic Signal Optimization")
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### Optimization Parameters")
            time_period = st.selectbox(
                "Time Period", 
                ["Morning Peak", "Afternoon", "Evening Peak", "Night"],
                index=0,
                key="signal_time_period"
            )
            time_column = {
                "Morning Peak": "Morning Peak(veh/h)",
                "Afternoon": "Afternoon(veh/h)",
                "Evening Peak": "Evening Peak(veh/h)",
                "Night": "Night(veh/h)"
            }[time_period]
            emergency_preemption_enabled = st.checkbox("Enable Emergency Preemption", value=True)
            adaptive_timing = st.checkbox("Use Adaptive Signal Timing", value=True)
            min_importance = st.slider("Minimum Intersection Importance", 0.0, 10.0, 3.0)
            optimize_button = st.button("Optimize Traffic Signals")
        
        with col1:
            if optimize_button:
                important_intersections = {
                    node_id: intersection for node_id, intersection in intersections.items()
                    if intersection['importance'] >= min_importance
                }
                with st.spinner("Optimizing traffic signals..."):
                    st.session_state.time_column = time_column  # Store time_column in session state
                    optimized_signals = real_time_signal_optimization(
                        important_intersections, traffic_flow, existing_roads, time_column
                    )
                    st.session_state.optimized_signals = optimized_signals
                    st.session_state.intersection_stats = important_intersections
                    
                    st.subheader("Signal Timing Results")
                    results_data = []
                    for node_id, signal in optimized_signals.items():
                        intersection_name = intersections[node_id]['name'] if node_id in intersections else "Unknown"
                        performance = signal['performance']
                        avg_saturation = sum(phase['saturation'] for phase in signal['phases']) / len(signal['phases'])
                        total_green_time = sum(phase['green_time'] for phase in signal['phases'])
                        results_data.append({
                            'Intersection': f"{node_id} - {intersection_name}",
                            'Average Saturation': f"{avg_saturation*100:.1f}%",
                            'Signal Efficiency': f"{performance['efficiency']*100:.1f}%",
                            'Average Wait Time': f"{performance['avg_wait_time']:.1f} sec",
                            'Max Queue Length': f"{max(performance['queue_lengths']):.1f} vehicles",
                            'Total Green Time': f"{total_green_time} sec"
                        })
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    st.subheader("Signal Timing Details")
                    intersection_options = [f"{node_id} - {intersections[node_id]['name']}" 
                                          for node_id in optimized_signals.keys() if node_id in intersections]
                    selected_intersection = st.selectbox("Select Intersection", intersection_options)
                    selected_node_id = selected_intersection.split(" - ")[0]
                    
                    signal = optimized_signals[selected_node_id]
                    performance = signal['performance']
                    avg_saturation = sum(phase['saturation'] for phase in signal['phases']) / len(signal['phases'])
                    st.markdown(f"**Cycle Time:** {signal['cycle_time']} seconds")
                    st.markdown(f"**Average Saturation:** {avg_saturation*100:.1f}%")
                    st.markdown(f"**Signal Efficiency:** {performance['efficiency']*100:.1f}%")
                    st.markdown(f"**Average Wait Time:** {performance['avg_wait_time']:.1f} seconds")
                    st.markdown(f"**Maximum Queue Length:** {max(performance['queue_lengths']):.1f} vehicles")
                    
                    phases_df = pd.DataFrame([
                        {
                            'Road': f"Road to {phase['road_id']}",
                            'Green Time': phase['green_time'],
                            'Traffic Volume': phase['volume'],
                            'Saturation': phase['saturation'],
                            'Urgency': phase.get('urgency', 0)
                        }
                        for phase in signal['phases']
                    ])
                    
                    fig1 = px.pie(phases_df, values='Green Time', names='Road',
                                 title="Green Time Allocation", color_discrete_sequence=px.colors.qualitative.G10)
                    fig1.update_traces(textinfo='percent+label+value')
                    
                    fig2 = px.bar(phases_df, x='Road', y=['Traffic Volume', 'Green Time'],
                                 title="Traffic Volume vs Green Time", barmode='group')
                    
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col_chart2:
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    if emergency_preemption_enabled:
                        st.subheader("Emergency Preemption Configuration")
                        test_route = [selected_node_id]
                        preemption_plan, time_saved = emergency_vehicle_preemption(
                            {selected_node_id: intersections[selected_node_id]}, optimized_signals, test_route, traffic_flow, time_column
                        )
                        if preemption_plan and selected_node_id in preemption_plan:
                            plan = preemption_plan[selected_node_id]
                            st.markdown(f"**Primary Direction:** {plan['primary_direction']}")
                            st.markdown(f"**Time Saved:** {plan['time_saved']:.1f} seconds")
                            st.markdown(f"**Emergency Green Time:** {plan['emergency_green_time']} seconds")
                            st.markdown(f"**Transition Time:** {plan['transition_time']} seconds")
                            st.subheader("Emergency Actions")
                            for i, action in enumerate(plan['emergency_actions']):
                                st.markdown(f"{i+1}. {action}")
                        else:
                            st.info("No emergency preemption configured for this intersection.")
            else:
                st.info("Click 'Optimize Traffic Signals' to run the greedy optimization algorithm.")
    
    with tabs[2]:
        st.subheader("Emergency Vehicle Preemption")
        if 'optimized_signals' not in st.session_state:
            st.warning("Please run Signal Optimization first.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("### Emergency Route Parameters")
                emergency_type = st.selectbox("Emergency Type", ["Medical Emergency", "Fire", "Police"], index=0)
                facility_options = [f"{row['ID']} - {row['Name']}" for _, row in facilities.iterrows()]
                default_facility_index = next(
                    (i for i, facility in enumerate(facility_options) if "Hospital" in facility and emergency_type == "Medical Emergency"),
                    0
                )
                emergency_origin = st.selectbox("Emergency Vehicle Origin", facility_options, index=default_facility_index)
                neighborhood_options = [f"{row['ID']} - {row['Name']}" for _, row in neighborhoods.iterrows()]
                incident_location = st.selectbox("Incident Location", neighborhood_options)
                
                origin_id = emergency_origin.split(" - ")[0]
                destination_id = incident_location.split(" - ")[0]
                
                time_period = st.selectbox("Time Period", ["Morning Peak", "Afternoon", "Evening Peak", "Night"],
                                          index=0, key="emergency_time_period")
                time_column = {
                    "Morning Peak": "Morning Peak(veh/h)",
                    "Afternoon": "Afternoon(veh/h)",
                    "Evening Peak": "Evening Peak(veh/h)",
                    "Night": "Night(veh/h)"
                }[time_period]
                
                simulate_button = st.button("Simulate Emergency Response")
            
            with col1:
                if simulate_button:
                    with st.spinner("Planning emergency route and simulating preemption..."):
                        results = plan_emergency_routes(
                            neighborhoods, facilities, existing_roads, traffic_flow,
                            origin_id, destination_id, time_column, priority_level=3,
                            route_clearing=True, emergency_type=emergency_type,
                            intersections=st.session_state.intersection_stats,
                            signals=st.session_state.optimized_signals
                        )
                        
                        (emergency_path, regular_path, emergency_time, regular_time, time_saved_route, 
                         distance, preemption_plan, time_saved_signals) = results
                        
                        st.subheader("Emergency Response Route")
                        emergency_map = plot_route_visualization(
                            neighborhoods, facilities, existing_roads,
                            emergency_path, regular_path, is_emergency=True
                        )
                        folium_static(emergency_map, width=700, height=500)
                        
                        st.subheader("Emergency Response Analysis")
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("Emergency Route Time", f"{emergency_time:.1f} min", delta=f"-{time_saved_route:.1f} min")
                        with col_stats2:
                            st.metric("Regular Route Time", f"{regular_time:.1f} min")
                        with col_stats3:
                            total_time_saved = time_saved_route + (time_saved_signals / 60)
                            st.metric("Total Time Saved", f"{total_time_saved:.1f} min")
                        
                        st.subheader("Signal Preemption Details")
                        preemption_df = []
                        for node_id, plan in preemption_plan.items():
                            name = intersections[node_id]['name'] if node_id in intersections else "Unknown"
                            preemption_df.append({
                                'Intersection': f"{node_id} - {name}",
                                'Normal Wait Time': f"{plan['time_saved']:.1f} sec" if plan['preemption_effective'] else "N/A",
                                'Preemption Wait': "0.0 sec" if plan['preemption_effective'] else "N/A",
                                'Time Saved': f"{plan['time_saved']:.1f} sec" if plan['preemption_effective'] else "0.0 sec",
                                'Status': "Active Preemption" if plan['preemption_effective'] else plan.get('reason', 'No Preemption')
                            })
                        
                        if preemption_df:
                            st.dataframe(pd.DataFrame(preemption_df), use_container_width=True)
                        else:
                            st.info("No intersections required preemption on this route.")
                else:
                    st.info("Set emergency route parameters and click 'Simulate Emergency Response' to analyze signal preemption.")