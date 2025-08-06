from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import numpy as np
import random

class FastMultiStartSolver(BaseSolver):
    """
    Fast multi-start solver for large TSP instances.
    Tries multiple starting points with quick local search.
    """
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV2Problem()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        
        best_tour = None
        best_distance = float('inf')
        
        # Try multiple starting points
        num_starts = min(10, max(3, n // 500))  # Adaptive number of starts
        
        for start in range(num_starts):
            if self.future_tracker.get(future_id):
                break
                
            # Generate random starting tour
            tour = self._random_tour(n)
            
            # Apply quick local search
            tour = self._quick_local_search(tour, distance_matrix)
            
            # Evaluate
            distance = self._calculate_tour_distance(tour, distance_matrix)
            
            if distance < best_distance:
                best_distance = distance
                best_tour = tour.copy()
        
        return best_tour
    
    def _random_tour(self, n):
        """Generate a random tour starting at 0"""
        tour = [0] + list(range(1, n))
        random.shuffle(tour[1:])  # Shuffle all except start
        tour.append(0)  # Return to start
        return tour
    
    def _quick_local_search(self, tour, distance_matrix, max_iterations=100):
        """Quick local search with limited iterations"""
        n = len(tour) - 1
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try simple swaps
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Calculate improvement
                    old_dist = (distance_matrix[tour[i-1]][tour[i]] + 
                               distance_matrix[tour[i]][tour[i+1]] +
                               distance_matrix[tour[j-1]][tour[j]] + 
                               distance_matrix[tour[j]][tour[j+1]])
                    
                    # Swap i and j
                    tour[i], tour[j] = tour[j], tour[i]
                    
                    new_dist = (distance_matrix[tour[i-1]][tour[i]] + 
                               distance_matrix[tour[i]][tour[i+1]] +
                               distance_matrix[tour[j-1]][tour[j]] + 
                               distance_matrix[tour[j]][tour[j+1]])
                    
                    if new_dist >= old_dist:
                        # Revert swap
                        tour[i], tour[j] = tour[j], tour[i]
                    else:
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return tour
    
    def _calculate_tour_distance(self, tour, distance_matrix):
        """Calculate total tour distance"""
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i+1]]
        return total_distance

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges 