from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV2ProblemMultiConstrained
import elkai
import numpy as np
import random

class LKHcmTSPSolver(BaseSolver):
    def __init__(self, problem_types:List[GraphV2ProblemMultiConstrained]=[GraphV2ProblemMultiConstrained()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int) -> List[List[int]]:
        problem = formatted_problem
        n = problem.n_nodes
        m = problem.n_salesmen
        depots = problem.depots
        demand = problem.demand
        constraint = problem.constraint
        distance_matrix = np.array(problem.edges)
        nodes = list(range(n))
        non_depot_nodes = [node for node in nodes if node not in depots]

        # --- Greedy demand-aware assignment ---
        groups = [[] for _ in range(m)]
        group_demand = [0 for _ in range(m)]
        # Sort nodes by descending demand (to fit big ones first)
        nodes_by_demand = sorted(non_depot_nodes, key=lambda x: -demand[x])
        for node in nodes_by_demand:
            for i in np.argsort(group_demand):
                if group_demand[i] + demand[node] <= constraint[i]:
                    groups[i].append(node)
                    group_demand[i] += demand[node]
                    break
        
        tours = []
        for i in range(m):
            group = groups[i]
            depot = depots[i]
            if not group:
                tours.append([depot, depot])
                continue
            sub_nodes = [depot] + group
            submatrix = distance_matrix[np.ix_(sub_nodes, sub_nodes)]
            
            # Solve TSP for this salesman using LKH
            try:
                tour = elkai.solve_float_matrix(submatrix.tolist())
                mapped_tour = [sub_nodes[idx] for idx in tour]
                # Ensure start/end at depot
                if mapped_tour[0] != depot:
                    idx0 = mapped_tour.index(depot)
                    mapped_tour = mapped_tour[idx0:] + mapped_tour[1:idx0+1]
                if mapped_tour[-1] != depot:
                    mapped_tour.append(depot)
                tours.append(mapped_tour)
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

    def problem_transformations(self, problem: Union[GraphV2ProblemMultiConstrained]):
        return problem 