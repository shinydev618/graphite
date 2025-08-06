from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV2ProblemMultiConstrained
import elkai
import numpy as np
import random
from sklearn.cluster import KMeans
import time

class LKHcmTSPSolver(BaseSolver):
    def __init__(self, problem_types:List[GraphV2ProblemMultiConstrained]=[GraphV2ProblemMultiConstrained()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int) -> List[List[int]]:
        start_time = time.time()
        problem = formatted_problem
        n = problem.n_nodes
        m = problem.n_salesmen
        depots = problem.depots
        demand = problem.demand
        constraint = problem.constraint
        distance_matrix = np.array(problem.edges)
        nodes = list(range(n))
        non_depot_nodes = [node for node in nodes if node not in depots]

        # Fast clustering with capacity awareness
        if problem.nodes is not None and len(problem.nodes) > 0:
            coords = np.array([problem.nodes[i] for i in non_depot_nodes])
        else:
            # Fallback: use random coordinates
            np.random.seed(42)
            coords = np.random.rand(len(non_depot_nodes), 2)
        
        # Fast capacity-aware clustering
        cluster_assignments = self._fast_capacity_clustering(coords, m, demand, constraint)

        # Assign nodes to salesmen
        groups = self._assign_nodes_fast(non_depot_nodes, cluster_assignments, demand, constraint, m)

        # Construct tours quickly
        tours = []
        for i in range(m):
            group = groups[i]
            depot = depots[i]
            if not group:
                tours.append([depot, depot])
                continue
            
            # Use fast construction method
            tour = self._construct_fast_tour(group, depot, demand, constraint[i], distance_matrix)
            tours.append(tour)

        # Quick local optimization (limited iterations)
        tours = self._quick_local_optimization(tours, distance_matrix, depots, demand, constraint, start_time)
        
        return tours

    def _fast_capacity_clustering(self, coords, m, demand, constraint):
        """Fast capacity-aware clustering using K-means with capacity balancing"""
        # Simple K-means clustering
        kmeans = KMeans(n_clusters=m, n_init=3, random_state=42)  # Reduced n_init for speed
        assignments = kmeans.fit_predict(coords)
        
        # Quick capacity balancing
        assignments = self._quick_capacity_balance(assignments, demand, constraint, m)
        
        return assignments

    def _quick_capacity_balance(self, assignments, demand, constraint, m):
        """Quick capacity balancing without extensive optimization"""
        cluster_demands = [0] * m
        for i, cluster_id in enumerate(assignments):
            cluster_demands[cluster_id] += demand[i]
        
        # Simple balancing: move excess nodes to underloaded clusters
        for cluster_id in range(m):
            if cluster_demands[cluster_id] > constraint[cluster_id]:
                excess = cluster_demands[cluster_id] - constraint[cluster_id]
                cluster_nodes = [i for i, c in enumerate(assignments) if c == cluster_id]
                
                # Sort by demand (largest first)
                cluster_nodes.sort(key=lambda x: demand[x], reverse=True)
                
                for node in cluster_nodes:
                    if cluster_demands[cluster_id] <= constraint[cluster_id]:
                        break
                    
                    # Find underloaded cluster
                    for target_cluster in range(m):
                        if (target_cluster != cluster_id and 
                            cluster_demands[target_cluster] + demand[node] <= constraint[target_cluster]):
                            assignments[node] = target_cluster
                            cluster_demands[cluster_id] -= demand[node]
                            cluster_demands[target_cluster] += demand[node]
                            break
        
        return assignments

    def _assign_nodes_fast(self, non_depot_nodes, cluster_assignments, demand, constraint, m):
        """Fast node assignment respecting capacity constraints"""
        groups = [[] for _ in range(m)]
        group_demands = [0] * m
        
        for idx, cluster_id in enumerate(cluster_assignments):
            node = non_depot_nodes[idx]
            node_demand = demand[node]
            
            # Try intended cluster first
            if group_demands[cluster_id] + node_demand <= constraint[cluster_id]:
                groups[cluster_id].append(node)
                group_demands[cluster_id] += node_demand
            else:
                # Find any cluster with capacity
                for i in range(m):
                    if group_demands[i] + node_demand <= constraint[i]:
                        groups[i].append(node)
                        group_demands[i] += node_demand
                        break
                else:
                    # If no capacity, assign to least loaded
                    least_loaded = np.argmin(group_demands)
                    groups[least_loaded].append(node)
                    group_demands[least_loaded] += node_demand
        
        return groups

    def _construct_fast_tour(self, nodes, depot, demand, capacity, distance_matrix):
        """Fast tour construction using nearest neighbor with capacity check"""
        if not nodes:
            return [depot, depot]
        
        # Try LKH first if small enough
        if len(nodes) <= 50:  # LKH is fast for small problems
            try:
                tour = self._lkh_tour_fast(nodes, depot, demand, capacity, distance_matrix)
                if tour and self._validate_capacity(tour, demand, capacity):
                    return tour
            except:
                pass
        
        # Fallback to nearest neighbor
        return self._nearest_neighbor_fast(nodes, depot, demand, capacity, distance_matrix)

    def _lkh_tour_fast(self, nodes, depot, demand, capacity, distance_matrix):
        """Fast LKH tour construction"""
        if not nodes:
            return [depot, depot]
            
        sub_nodes = [depot] + nodes
        sub_matrix = distance_matrix[np.ix_(sub_nodes, sub_nodes)]
        
        # Convert to integer matrix for LKH
        int_matrix = np.rint(sub_matrix).astype(int).tolist()
        tour = elkai.solve_int_matrix(int_matrix)
        
        # Map back to original node indices
        mapped_tour = [sub_nodes[idx] for idx in tour]
        
        # Ensure start/end at depot
        if mapped_tour[0] != depot:
            idx0 = mapped_tour.index(depot)
            mapped_tour = mapped_tour[idx0:] + mapped_tour[1:idx0+1]
        if mapped_tour[-1] != depot:
            mapped_tour.append(depot)
        
        return mapped_tour

    def _nearest_neighbor_fast(self, nodes, depot, demand, capacity, distance_matrix):
        """Fast nearest neighbor construction with capacity check"""
        if not nodes:
            return [depot, depot]
            
        tour = [depot]
        unvisited = nodes.copy()
        current_demand = 0
        
        while unvisited:
            current = tour[-1]
            nearest = None
            min_dist = float('inf')
            
            for city in unvisited:
                if current_demand + demand[city] <= capacity:
                    dist = distance_matrix[current][city]
                    if dist < min_dist:
                        min_dist = dist
                        nearest = city
            
            if nearest is None:
                break
                
            tour.append(nearest)
            unvisited.remove(nearest)
            current_demand += demand[nearest]
        
        tour.append(depot)
        return tour

    def _validate_capacity(self, tour, demand, capacity):
        """Validate capacity constraint"""
        if not tour:
            return True
            
        total_demand = sum(demand[node] for node in tour if node != tour[0])  # Exclude depot
        return total_demand <= capacity

    def _quick_local_optimization(self, tours, distance_matrix, depots, demand, constraint, start_time):
        """Quick local optimization with time limit"""
        max_iterations = 3  # Very limited for speed
        improved = True
        
        for iteration in range(max_iterations):
            if not improved or time.time() - start_time > 25:  # Leave 5s buffer
                break
            improved = False
            
            # Quick 2-opt within each tour
            for i, tour in enumerate(tours):
                if len(tour) > 4:
                    new_tour = self._quick_two_opt(tour, distance_matrix, demand, constraint[i])
                    if new_tour != tour:
                        tours[i] = new_tour
                        improved = True
            
            # Quick cross-exchange between tours (limited attempts)
            for i in range(len(tours)):
                for j in range(i + 1, min(i + 3, len(tours))):  # Limited pairs
                    if self._quick_cross_exchange(tours, i, j, distance_matrix, demand, constraint):
                        improved = True
                        break
                if improved:
                    break
        
        return tours

    def _quick_two_opt(self, tour, distance_matrix, demand, capacity):
        """Quick 2-opt optimization"""
        if len(tour) <= 4:
            return tour
            
        best_tour = tour.copy()
        best_cost = self._calculate_cost(tour, distance_matrix)
        
        # Limited 2-opt attempts
        for i in range(1, min(len(tour) - 2, 10)):  # Limit iterations
            for j in range(i + 1, min(len(tour) - 1, i + 10)):  # Limit range
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                
                if self._validate_capacity(new_tour, demand, capacity):
                    new_cost = self._calculate_cost(new_tour, distance_matrix)
                    
                    if new_cost < best_cost:
                        best_tour = new_tour.copy()
                        best_cost = new_cost
                        break
            if best_tour != tour:
                break
                
        return best_tour

    def _quick_cross_exchange(self, tours, i, j, distance_matrix, demand, constraint):
        """Quick cross-exchange between two tours"""
        if len(tours[i]) <= 2 or len(tours[j]) <= 2:
            return False
            
        original_cost = (self._calculate_cost(tours[i], distance_matrix) + 
                        self._calculate_cost(tours[j], distance_matrix))
        
        # Try exchanging one node from each tour (limited attempts)
        nodes_i = tours[i][1:-1][:3]  # Limit to first 3 nodes
        nodes_j = tours[j][1:-1][:3]  # Limit to first 3 nodes
        
        for node_i in nodes_i:
            for node_j in nodes_j:
                new_tour_i = [x if x != node_i else node_j for x in tours[i]]
                new_tour_j = [x if x != node_j else node_i for x in tours[j]]
                
                if (self._validate_capacity(new_tour_i, demand, constraint[i]) and
                    self._validate_capacity(new_tour_j, demand, constraint[j])):
                    
                    new_cost = (self._calculate_cost(new_tour_i, distance_matrix) + 
                               self._calculate_cost(new_tour_j, distance_matrix))
                    
                    if new_cost < original_cost:
                        tours[i] = new_tour_i
                        tours[j] = new_tour_j
                        return True
        
        return False

    def _calculate_cost(self, tour, distance_matrix):
        """Calculate tour cost"""
        if len(tour) < 2:
            return 0
        cost = 0
        for i in range(len(tour) - 1):
            cost += distance_matrix[tour[i]][tour[i+1]]
        return cost

    def problem_transformations(self, problem: Union[GraphV2ProblemMultiConstrained]):
        return problem 