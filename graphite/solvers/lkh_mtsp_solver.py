from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV2ProblemMulti
import elkai
import numpy as np
import random

class LKHmTSPSolver(BaseSolver):
    def __init__(self, problem_types:List[GraphV2ProblemMulti]=[GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int) -> List[List[int]]:
        # formatted_problem is expected to be the problem object
        problem = formatted_problem
        n = problem.n_nodes
        m = problem.n_salesmen
        depots = problem.depots
        distance_matrix = np.array(problem.edges)
        nodes = list(range(n))
        
        # Remove depot nodes from the list of nodes to assign
        non_depot_nodes = [node for node in nodes if node not in depots]
        
        # Assign nodes to salesmen (round-robin assignment)
        salesman_nodes = [[] for _ in range(m)]
        for i, node in enumerate(non_depot_nodes):
            salesman_nodes[i % m].append(node)
        
        tours = []
        for i in range(m):
            if not salesman_nodes[i]:
                # If no nodes assigned to this salesman, just return depot
                tours.append([depots[i], depots[i]])
                continue
                
            # Add depot to the beginning and end of the tour
            tour_nodes = [depots[i]] + salesman_nodes[i] + [depots[i]]
            
            # Create sub-matrix for this salesman's nodes
            sub_nodes = [depots[i]] + salesman_nodes[i]
            sub_matrix = distance_matrix[np.ix_(sub_nodes, sub_nodes)]
            
            # Solve TSP for this salesman using LKH
            try:
                sub_tour = elkai.solve_float_matrix(sub_matrix.tolist())
                # Map back to original node indices
                tour = [sub_nodes[idx] for idx in sub_tour]
                tours.append(tour)
            except Exception as e:
                # Fallback: use greedy approach
                tour = self._greedy_tour(sub_nodes, distance_matrix)
                tours.append(tour)
        
        return tours
    
    def _greedy_tour(self, nodes, distance_matrix):
        """Fallback greedy tour construction"""
        if len(nodes) <= 2:
            return nodes
            
        tour = [nodes[0]]  # Start with depot
        unvisited = set(nodes[1:])
        
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        tour.append(nodes[0])  # Return to depot
        return tour

    def problem_transformations(self, problem):
        return problem 