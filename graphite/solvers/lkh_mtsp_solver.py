from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV2ProblemMulti
import elkai
import numpy as np
import random
from sklearn.cluster import KMeans
import math

class LKHmTSPSolver(BaseSolver):
    def __init__(self, problem_types:List[GraphV2ProblemMulti]=[GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int) -> List[List[int]]:
        problem = formatted_problem
        n = problem.n_nodes
        m = problem.n_salesmen
        depots = problem.depots
        distance_matrix = np.array(problem.edges)
        nodes = list(range(n))
        
        # Remove depot nodes from the list of nodes to assign
        non_depot_nodes = [node for node in nodes if node not in depots]
        
        # Phase 1: Fast clustering using K-means only
        if problem.nodes is not None and len(problem.nodes) > 0:
            coords = np.array([problem.nodes[i] for i in non_depot_nodes])
        else:
            # Fallback: use random coordinates
            np.random.seed(42)
            coords = np.random.rand(len(non_depot_nodes), 2)
        
        # Fast K-means clustering
        kmeans = KMeans(n_clusters=m, n_init=3, random_state=42)  # Reduced n_init for speed
        cluster_assignments = kmeans.fit_predict(coords)

        # Phase 2: Assign nodes to salesmen
        salesman_nodes = [[] for _ in range(m)]
        for idx, label in enumerate(cluster_assignments):
            salesman_nodes[label].append(non_depot_nodes[idx])

        # Phase 3: Construct tours using fastest method only
        tours = []
        for i in range(m):
            if not salesman_nodes[i]:
                tours.append([depots[i], depots[i]])
                continue
                
            # Use only LKH for tour construction (fastest and best quality)
            tour = self._fast_lkh_tour(salesman_nodes[i], depots[i], distance_matrix)
            tours.append(tour)
        
        # Phase 4: Quick local search optimization (limited iterations)
        tours = self._quick_local_search(tours, distance_matrix, depots)
        
        return tours
    
    def _fast_lkh_tour(self, nodes, depot, distance_matrix):
        """Fast LKH tour construction"""
        if not nodes:
            return [depot, depot]
            
        sub_nodes = [depot] + nodes
        sub_matrix = distance_matrix[np.ix_(sub_nodes, sub_nodes)]
        
        # Convert to integer matrix for LKH
        int_matrix = np.rint(sub_matrix).astype(int).tolist()
        
        try:
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
        except Exception:
            # Fallback to nearest neighbor if LKH fails
            return self._nearest_neighbor_tour(nodes, depot, distance_matrix)
    
    def _nearest_neighbor_tour(self, nodes, depot, distance_matrix):
        """Fast nearest neighbor tour construction"""
        if not nodes:
            return [depot, depot]
            
        tour = [depot]
        unvisited = nodes.copy()
        
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        tour.append(depot)
        return tour
    
    def _calculate_tour_cost(self, tour, distance_matrix):
        """Calculate total cost of a tour"""
        if len(tour) < 2:
            return 0
        cost = 0
        for i in range(len(tour) - 1):
            cost += distance_matrix[tour[i]][tour[i+1]]
        return cost
    
    def _quick_local_search(self, tours, distance_matrix, depots):
        """Quick local search with limited iterations"""
        improved = True
        max_iterations = 3  # Reduced from 10 to 3
        
        for iteration in range(max_iterations):
            if not improved:
                break
            improved = False
            
            # Quick 2-opt moves within each tour (limited attempts)
            for i, tour in enumerate(tours):
                if len(tour) > 4:  # Only optimize tours with more than 4 nodes
                    new_tour = self._quick_two_opt(tour, distance_matrix)
                    if new_tour != tour:
                        tours[i] = new_tour
                        improved = True
            
            # Skip cross-exchange for speed (most expensive operation)
        
        return tours
    
    def _quick_two_opt(self, tour, distance_matrix):
        """Quick 2-opt optimization with limited attempts"""
        if len(tour) <= 4:
            return tour
            
        best_tour = tour.copy()
        best_cost = self._calculate_tour_cost(tour, distance_matrix)
        attempts = 0
        max_attempts = 50  # Limit attempts for speed
        
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                if attempts >= max_attempts:
                    return best_tour
                    
                # Try 2-opt move
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                new_cost = self._calculate_tour_cost(new_tour, distance_matrix)
                
                if new_cost < best_cost:
                    best_tour = new_tour.copy()
                    best_cost = new_cost
                    tour = new_tour.copy()
                    break
                attempts += 1
            if attempts >= max_attempts:
                break
                
        return best_tour

    def problem_transformations(self, problem):
        return problem 