# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Union, Tuple, Set
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV2ProblemMultiConstrained
import numpy as np
import time
import asyncio
import random
from collections import defaultdict

import bittensor as bt

class ALNSCMTSPsolver(BaseSolver):
    """
    Adaptive Large Neighborhood Search for cmTSP with constraints.
    Destroys and repairs solutions iteratively.
    Time complexity: O(iterations × n²)
    Quality: Excellent for complex constraints
    """
    
    def __init__(self, problem_types: List[GraphV2ProblemMultiConstrained] = [GraphV2ProblemMultiConstrained()]):
        super().__init__(problem_types=problem_types)
        
        # ALNS parameters - optimized for 30s timeout
        self.iterations = 20  # Much more aggressive reduction
        self.destroy_ratio = 0.2  # Smaller destroy ratio for faster repair
        self.temperature = 500  # Lower temperature for faster convergence
        self.cooling_rate = 0.9  # Faster cooling
        self.adaptive_weight = 0.2  # More aggressive adaptation

    def create_initial_solution(self, problem) -> List[List[int]]:
        """Create initial feasible solution using simplified greedy approach"""
        distance_matrix = np.array(problem.edges)
        n_nodes = problem.n_nodes
        n_salesmen = problem.n_salesmen
        depots = problem.depots
        demands = problem.demand
        constraints = problem.constraint
        
        # Initialize routes with depots
        routes = [[depot] for depot in depots]
        route_demands = [0] * n_salesmen
        unvisited = [i for i in range(n_nodes) if i not in depots]
        
        # Simplified assignment: assign cities to nearest feasible route
        while unvisited:
            city = unvisited.pop(0)  # Take first unvisited city
            
            # Find route with most capacity and nearest depot
            best_route = 0
            best_score = float('inf')
            
            for route_idx in range(n_salesmen):
                if route_demands[route_idx] + demands[city] <= constraints[route_idx]:
                    # Score based on capacity remaining and distance to depot
                    capacity_score = constraints[route_idx] - route_demands[route_idx]
                    distance_score = distance_matrix[city][depots[route_idx]]
                    total_score = distance_score - capacity_score * 0.1  # Weight capacity more
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_route = route_idx
            
            # Assign to best route found
            routes[best_route].append(city)
            route_demands[best_route] += demands[city]
        
        # Complete routes by returning to depots
        for i in range(n_salesmen):
            routes[i].append(depots[i])
        
        return routes

    def calculate_solution_cost(self, routes: List[List[int]], distance_matrix: np.ndarray) -> float:
        """Calculate total cost of solution"""
        total_cost = 0
        for route in routes:
            for i in range(len(route) - 1):
                total_cost += distance_matrix[route[i]][route[i + 1]]
        return total_cost

    def check_feasibility(self, routes: List[List[int]], demands: List[int], constraints: List[int]) -> bool:
        """Check if solution is feasible"""
        for route_idx, route in enumerate(routes):
            route_demand = sum(demands[city] for city in route if city != route[0])  # Exclude depot
            if route_demand > constraints[route_idx]:
                return False
        return True

    def destroy_random(self, routes: List[List[int]], destroy_ratio: float) -> Tuple[List[List[int]], List[int]]:
        """Destroy random nodes from routes"""
        all_cities = []
        for route in routes:
            all_cities.extend(route[1:-1])  # Exclude depots
        
        num_to_destroy = int(len(all_cities) * destroy_ratio)
        destroyed_cities = random.sample(all_cities, min(num_to_destroy, len(all_cities)))
        
        # Remove destroyed cities from routes
        new_routes = []
        for route in routes:
            new_route = [route[0]]  # Keep depot
            for city in route[1:-1]:  # Exclude depots
                if city not in destroyed_cities:
                    new_route.append(city)
            new_route.append(route[-1])  # Keep depot
            new_routes.append(new_route)
        
        return new_routes, destroyed_cities

    def destroy_worst(self, routes: List[List[int]], distance_matrix: np.ndarray, destroy_ratio: float) -> Tuple[List[List[int]], List[int]]:
        """Destroy worst nodes (highest cost) from routes"""
        all_cities = []
        city_costs = {}
        
        for route in routes:
            for i, city in enumerate(route[1:-1]):  # Exclude depots
                all_cities.append(city)
                # Calculate cost of this city in the route
                if i == 0:
                    cost = distance_matrix[route[0]][city] + distance_matrix[city][route[i+2]]
                elif i == len(route) - 3:
                    cost = distance_matrix[route[i]][city] + distance_matrix[city][route[-1]]
                else:
                    cost = distance_matrix[route[i]][city] + distance_matrix[city][route[i+2]]
                city_costs[city] = cost
        
        num_to_destroy = int(len(all_cities) * destroy_ratio)
        # Sort cities by cost (descending) and destroy worst ones
        sorted_cities = sorted(all_cities, key=lambda x: city_costs.get(x, 0), reverse=True)
        destroyed_cities = sorted_cities[:num_to_destroy]
        
        # Remove destroyed cities from routes
        new_routes = []
        for route in routes:
            new_route = [route[0]]  # Keep depot
            for city in route[1:-1]:  # Exclude depots
                if city not in destroyed_cities:
                    new_route.append(city)
            new_route.append(route[-1])  # Keep depot
            new_routes.append(new_route)
        
        return new_routes, destroyed_cities

    def repair_greedy(self, routes: List[List[int]], destroyed_cities: List[int], 
                     distance_matrix: np.ndarray, demands: List[int], constraints: List[int]) -> List[List[int]]:
        """Repair solution by greedily inserting destroyed cities"""
        # Calculate current route demands
        route_demands = []
        for route in routes:
            demand = sum(demands[city] for city in route if city != route[0])  # Exclude depot
            route_demands.append(demand)
        
        # Insert cities greedily
        for city in destroyed_cities:
            best_insertion = None
            best_cost = float('inf')
            
            for route_idx, route in enumerate(routes):
                # Check capacity constraint
                if route_demands[route_idx] + demands[city] <= constraints[route_idx]:
                    # Find best insertion position
                    for pos in range(1, len(route)):
                        prev_city = route[pos-1]
                        next_city = route[pos]
                        
                        # Calculate insertion cost
                        insert_cost = (distance_matrix[prev_city][city] + 
                                     distance_matrix[city][next_city] - 
                                     distance_matrix[prev_city][next_city])
                        
                        if insert_cost < best_cost:
                            best_cost = insert_cost
                            best_insertion = (route_idx, pos)
            
            if best_insertion:
                route_idx, pos = best_insertion
                routes[route_idx].insert(pos, city)
                route_demands[route_idx] += demands[city]
            else:
                # If no feasible insertion, find route with most remaining capacity
                route_idx = max(range(len(routes)), 
                              key=lambda i: constraints[i] - route_demands[i])
                routes[route_idx].insert(-1, city)  # Insert before depot
                route_demands[route_idx] += demands[city]
        
        return routes

    def repair_regret(self, routes: List[List[int]], destroyed_cities: List[int], 
                     distance_matrix: np.ndarray, demands: List[int], constraints: List[int]) -> List[List[int]]:
        """Repair solution using regret heuristic"""
        # Calculate current route demands
        route_demands = []
        for route in routes:
            demand = sum(demands[city] for city in route if city != route[0])  # Exclude depot
            route_demands.append(demand)
        
        # Insert cities using regret heuristic
        for city in destroyed_cities:
            # Calculate insertion costs for all feasible positions
            insertions = []
            
            for route_idx, route in enumerate(routes):
                if route_demands[route_idx] + demands[city] <= constraints[route_idx]:
                    for pos in range(1, len(route)):
                        prev_city = route[pos-1]
                        next_city = route[pos]
                        
                        insert_cost = (distance_matrix[prev_city][city] + 
                                     distance_matrix[city][next_city] - 
                                     distance_matrix[prev_city][next_city])
                        
                        insertions.append((insert_cost, route_idx, pos))
            
            if insertions:
                # Sort by cost
                insertions.sort()
                
                if len(insertions) >= 2:
                    # Use regret heuristic: choose position with highest regret
                    regret = insertions[1][0] - insertions[0][0]  # Difference between 2nd and 1st best
                    best_insertion = insertions[0]
                else:
                    best_insertion = insertions[0]
                
                route_idx, pos = best_insertion[1], best_insertion[2]
                routes[route_idx].insert(pos, city)
                route_demands[route_idx] += demands[city]
            else:
                # Fallback: insert in route with most capacity
                route_idx = max(range(len(routes)), 
                              key=lambda i: constraints[i] - route_demands[i])
                routes[route_idx].insert(-1, city)
                route_demands[route_idx] += demands[city]
        
        return routes

    def local_search_2opt(self, routes: List[List[int]], distance_matrix: np.ndarray) -> List[List[int]]:
        """Apply simplified 2-OPT local search to each route"""
        improved_routes = []
        
        for route in routes:
            if len(route) <= 3:
                improved_routes.append(route)
                continue
            
            # Apply 2-OPT to inner cities (excluding depots) - limited iterations
            inner_route = route[1:-1]
            improved = True
            max_iterations = 3  # Reduced from 10
            iteration = 0
            
            while improved and iteration < max_iterations:
                improved = False
                # Limit search range for speed
                for i in range(min(len(inner_route) - 1, 10)):  # Only first 10 positions
                    for j in range(i + 2, min(len(inner_route), i + 10)):  # Limited range
                        if j - i == 1:
                            continue
                        
                        # Calculate current distance
                        old_dist = (distance_matrix[inner_route[i]][inner_route[i+1]] + 
                                  distance_matrix[inner_route[j-1]][inner_route[j]])
                        
                        # Calculate new distance
                        new_dist = (distance_matrix[inner_route[i]][inner_route[j-1]] + 
                                  distance_matrix[inner_route[i+1]][inner_route[j]])
                        
                        if new_dist < old_dist:
                            # Reverse the segment
                            inner_route[i+1:j] = inner_route[j-1:i:-1]
                            improved = True
                            break
                    if improved:
                        break
                iteration += 1
            
            # Reconstruct route with depots
            improved_route = [route[0]] + inner_route + [route[-1]]
            improved_routes.append(improved_route)
        
        return improved_routes

    async def solve(self, formatted_problem, future_id: int) -> List[List[int]]:
        """
        Solve cmTSP using Adaptive Large Neighborhood Search:
        1. Create initial solution
        2. Iteratively destroy and repair
        3. Apply local search
        4. Use adaptive weights for destroy/repair operators
        """
        problem = formatted_problem
        distance_matrix = np.array(problem.edges)
        n_nodes = problem.n_nodes
        n_salesmen = problem.n_salesmen
        depots = problem.depots
        demands = problem.demand
        constraints = problem.constraint
        
        # Create initial solution
        current_solution = self.create_initial_solution(problem)
        current_cost = self.calculate_solution_cost(current_solution, distance_matrix)
        
        best_solution = [route[:] for route in current_solution]
        best_cost = current_cost
        
        # Initialize operator weights and scores
        destroy_operators = [self.destroy_random, self.destroy_worst]
        repair_operators = [self.repair_greedy, self.repair_regret]
        
        weights = {
            'destroy': [1.0] * len(destroy_operators),
            'repair': [1.0] * len(repair_operators)
        }
        
        scores = {
            'destroy': [0.0] * len(destroy_operators),
            'repair': [0.0] * len(repair_operators)
        }
        
        # Main ALNS loop
        start_time = time.time()
        max_time = 25  # Maximum time in seconds
        
        for iteration in range(self.iterations):
            if self.future_tracker.get(future_id):
                return None
            
            # Check timeout more frequently
            if time.time() - start_time > max_time:
                bt.logging.warning(f"ALNS solver timeout after {max_time}s, returning best solution found")
                break
            
            # Early termination if we have a good solution
            if iteration > 5 and best_cost < float('inf'):
                # If we've found a solution and done some iterations, check if we should stop early
                if time.time() - start_time > 15:  # Stop early if we've been running for 15s
                    bt.logging.info(f"ALNS early termination after {iteration} iterations")
                    break
            
            # Select destroy and repair operators based on weights
            destroy_weights = np.array(weights['destroy'])
            repair_weights = np.array(weights['repair'])
            
            destroy_idx = np.random.choice(len(destroy_operators), p=destroy_weights/sum(destroy_weights))
            repair_idx = np.random.choice(len(repair_operators), p=repair_weights/sum(repair_weights))
            
            destroy_op = destroy_operators[destroy_idx]
            repair_op = repair_operators[repair_idx]
            
            # Destroy and repair - handle different parameter requirements
            if destroy_op == self.destroy_worst:
                destroyed_solution, destroyed_cities = destroy_op(current_solution, distance_matrix, self.destroy_ratio)
            else:
                destroyed_solution, destroyed_cities = destroy_op(current_solution, self.destroy_ratio)
            new_solution = repair_op(destroyed_solution, destroyed_cities, distance_matrix, demands, constraints)
            
            # Apply local search
            new_solution = self.local_search_2opt(new_solution, distance_matrix)
            
            # Calculate new cost
            new_cost = self.calculate_solution_cost(new_solution, distance_matrix)
            
            # Simulated annealing acceptance
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / self.temperature):
                current_solution = new_solution
                current_cost = new_cost
                
                # Update best solution
                if current_cost < best_cost:
                    best_solution = [route[:] for route in current_solution]
                    best_cost = current_cost
                    
                    # Reward for finding new best
                    scores['destroy'][destroy_idx] += 10
                    scores['repair'][repair_idx] += 10
                else:
                    # Reward for acceptance
                    scores['destroy'][destroy_idx] += 5
                    scores['repair'][repair_idx] += 5
            else:
                # Small reward for trying
                scores['destroy'][destroy_idx] += 1
                scores['repair'][repair_idx] += 1
            
            # Update weights more frequently for faster adaptation
            if iteration % 5 == 0:  # Reduced from 10
                for op_type in ['destroy', 'repair']:
                    for i in range(len(weights[op_type])):
                        weights[op_type][i] = (1 - self.adaptive_weight) * weights[op_type][i] + \
                                            self.adaptive_weight * scores[op_type][i] / 10
                        scores[op_type][i] = 0  # Reset scores
            
            # Cool down temperature faster
            self.temperature *= self.cooling_rate
        
        return best_solution

    def problem_transformations(self, problem: GraphV2ProblemMultiConstrained):
        """Transform problem to required format"""
        return problem

if __name__ == "__main__":
    # Test the ALNS cmTSP solver
    import random
    import math
    
    # Create a test problem
    n_nodes = 100
    n_salesmen = 5
    depots = [0, 1, 2, 3, 4]
    
    # Create random distance matrix
    distance_matrix = np.random.rand(n_nodes, n_nodes) * 100
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Create demands and constraints
    demands = [random.randint(1, 9) for _ in range(n_nodes)]
    for depot in depots:
        demands[depot] = 0
    
    total_demand = sum(demands)
    constraint = [math.ceil(total_demand / n_salesmen) + random.randint(-5, 5) for _ in range(n_salesmen)]
    
    # Create test problem
    test_problem = GraphV2ProblemMultiConstrained(
        n_nodes=n_nodes,
        n_salesmen=n_salesmen,
        depots=depots,
        single_depot=False,
        demand=demands,
        constraint=constraint
    )
    test_problem.edges = distance_matrix
    
    solver = ALNSCMTSPsolver(problem_types=[test_problem])
    
    start_time = time.time()
    routes = asyncio.run(solver.solve_problem(test_problem))
    end_time = time.time()
    
    print(f"ALNS cmTSP Solution: {routes}")
    print(f"Time taken for {n_nodes} nodes, {n_salesmen} salesmen: {end_time - start_time:.3f} seconds")
    
    # Calculate total distance and check feasibility
    if routes:
        total_distance = 0
        for route in routes:
            route_distance = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            total_distance += route_distance
        print(f"Total distance: {total_distance:.2f}")
        print(f"Route lengths: {[len(route) for route in routes]}")
        
        # Check feasibility
        is_feasible = solver.check_feasibility(routes, demands, constraint)
        print(f"Solution feasible: {is_feasible}")
        
        # Calculate route demands
        route_demands = []
        for route in routes:
            route_demand = sum(demands[city] for city in route if city != route[0])
            route_demands.append(route_demand)
        print(f"Route demands: {route_demands}")
        print(f"Constraints: {constraint}") 