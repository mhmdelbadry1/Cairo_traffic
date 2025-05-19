import numpy as np
from functools import lru_cache

def allocate_maintenance_resources_dp(roads, budget, time_periods=4):
    """
    Allocate road maintenance resources using dynamic programming to maximize road condition improvement
    while staying within budget constraints.
    
    Args:
        roads (DataFrame): Road data with columns 'FromID', 'ToID', 'Distance(km)', 'Condition(1-10)'
        budget (float): Total maintenance budget in Million EGP
        time_periods (int): Number of time periods to plan for (quarters, months, etc.)
        
    Returns:
        tuple: (allocation_plan, expected_condition_improvement, budget_used, allocation_metrics)
            allocation_plan: DataFrame with road IDs and allocated budget for each time period
            expected_condition_improvement: Overall expected improvement in road conditions
            budget_used: Total budget used
            allocation_metrics: Dictionary with additional metrics about the allocation
    """
    # Prepare road data for DP
    road_data = []
    for i, road in roads.iterrows():
        # Calculate maintenance cost based on length and current condition
        # Lower condition roads cost more to repair
        condition = road['Condition(1-10)']
        length = road['Distance(km)']
        
        # Base cost per km based on condition (worse roads cost more)
        base_cost_per_km = 10 - condition  # 1-9 million EGP per km
        
        # Calculate potential improvement for different investment levels
        # We'll create 5 investment levels for each road
        investment_levels = [0]  # No investment
        improvements = [0]  # No improvement
        
        # Calculate 4 possible investment levels and their improvements
        for level in range(1, 5):
            # Investment as a percentage of max possible investment
            investment_pct = level / 4
            investment = base_cost_per_km * length * investment_pct
            
            # Improvement depends on current condition and investment
            # Roads in worse condition show more improvement per investment
            if condition <= 3:  # Poor condition
                improvement = 3 * investment_pct
            elif condition <= 6:  # Medium condition
                improvement = 2 * investment_pct
            else:  # Good condition
                improvement = 1 * investment_pct
                
            investment_levels.append(investment)
            improvements.append(improvement)
        
        road_data.append({
            'id': i,
            'from_id': road['FromID'],
            'to_id': road['ToID'],
            'length': length,
            'condition': condition,
            'investment_levels': investment_levels,
            'improvements': improvements
        })
    
    # Convert budget to integer (in thousands) for DP table
    budget_units = int(budget * 100)  # Convert to units of 10,000 EGP
    
    # Initialize DP table
    # dp[t][b] = max improvement for time periods 0...t with budget b
    dp = [[-1 for _ in range(budget_units + 1)] for _ in range(time_periods)]
    
    # Initialize decisions table to track allocations
    decisions = [[-1 for _ in range(budget_units + 1)] for _ in range(time_periods)]
    
    # Helper function to calculate total improvement for a given allocation
    @lru_cache(maxsize=None)
    def calculate_improvement(allocation_tuple):
        total_improvement = 0
        for road_idx, level in enumerate(allocation_tuple):
            total_improvement += road_data[road_idx]['improvements'][level]
        return total_improvement
    
    # Helper function to calculate total cost for a given allocation
    @lru_cache(maxsize=None)
    def calculate_cost(allocation_tuple):
        total_cost = 0
        for road_idx, level in enumerate(allocation_tuple):
            total_cost += road_data[road_idx]['investment_levels'][level]
        return total_cost * 100  # Convert to budget units
    
    # Generate all possible allocations for each time period
    # For simplicity, we'll consider allocating to at most 3 roads per period
    # with at most level 3 investment per road
    road_indices = list(range(len(road_data)))
    
    # For each time period, calculate the best allocation
    for t in range(time_periods):
        # For each possible budget
        for b in range(budget_units + 1):
            max_improvement = 0
            best_allocation = None
            
            # Try different road combinations (up to 3 roads)
            for r1 in range(len(road_data)):
                for level1 in range(min(4, len(road_data[r1]['investment_levels']))):
                    cost1 = int(road_data[r1]['investment_levels'][level1] * 100)
                    if cost1 > b:
                        continue
                    
                    improvement1 = road_data[r1]['improvements'][level1]
                    
                    # Try allocating to just one road
                    if improvement1 > max_improvement:
                        max_improvement = improvement1
                        best_allocation = [(r1, level1)]
                    
                    # Try two roads
                    for r2 in range(r1 + 1, len(road_data)):
                        for level2 in range(min(4, len(road_data[r2]['investment_levels']))):
                            cost2 = int(road_data[r2]['investment_levels'][level2] * 100)
                            if cost1 + cost2 > b:
                                continue
                            
                            improvement2 = improvement1 + road_data[r2]['improvements'][level2]
                            
                            if improvement2 > max_improvement:
                                max_improvement = improvement2
                                best_allocation = [(r1, level1), (r2, level2)]
                            
                            # Try three roads
                            for r3 in range(r2 + 1, len(road_data)):
                                for level3 in range(min(4, len(road_data[r3]['investment_levels']))):
                                    cost3 = int(road_data[r3]['investment_levels'][level3] * 100)
                                    if cost1 + cost2 + cost3 > b:
                                        continue
                                    
                                    improvement3 = improvement2 + road_data[r3]['improvements'][level3]
                                    
                                    if improvement3 > max_improvement:
                                        max_improvement = improvement3
                                        best_allocation = [(r1, level1), (r2, level2), (r3, level3)]
            
            dp[t][b] = max_improvement
            decisions[t][b] = best_allocation
    
    # Reconstruct the solution
    allocation_plan = []
    remaining_budget = budget_units
    total_improvement = 0
    
    for t in range(time_periods):
        period_allocation = decisions[t][remaining_budget]
        period_improvement = dp[t][remaining_budget]
        
        # Calculate cost of this period's allocation
        period_cost = 0
        if period_allocation:
            for road_idx, level in period_allocation:
                period_cost += road_data[road_idx]['investment_levels'][level] * 100
        
        # Update remaining budget
        remaining_budget -= int(period_cost)
        total_improvement += period_improvement
        
        # Add to allocation plan
        for road_idx, level in period_allocation if period_allocation else []:
            road = road_data[road_idx]
            allocation_plan.append({
                'Time Period': t + 1,
                'Road': f"{road['from_id']}-{road['to_id']}",
                'Current Condition': road['condition'],
                'Length (km)': road['length'],
                'Investment (Million EGP)': road['investment_levels'][level],
                'Expected Improvement': road['improvements'][level],
                'New Condition': min(10, road['condition'] + road['improvements'][level])
            })
    
    # Calculate metrics
    budget_used = budget - (remaining_budget / 100)
    
    allocation_metrics = {
        'total_roads_improved': len(set(item['Road'] for item in allocation_plan)),
        'avg_improvement_per_road': total_improvement / max(1, len(set(item['Road'] for item in allocation_plan))),
        'budget_efficiency': total_improvement / max(0.1, budget_used),
        'pct_budget_used': (budget_used / budget) * 100 if budget > 0 else 0
    }
    
    return allocation_plan, total_improvement, budget_used, allocation_metrics
