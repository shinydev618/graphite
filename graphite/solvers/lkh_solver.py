from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
import elkai

class LKHSolver(BaseSolver):
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV2Problem(n_nodes=2)]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int) -> List[int]:
        distance_matrix = formatted_problem
        tour = elkai.solve_float_matrix(distance_matrix)
        # Ensure the tour is a valid cycle starting and ending at 0
        if tour[0] != 0:
            # Rotate so that 0 is at the start
            idx = tour.index(0)
            tour = tour[idx:] + tour[1:idx+1]
        # Ensure the tour ends at 0
        if tour[-1] != 0:
            tour.append(0)
        # Remove duplicates except for start/end
        seen = set()
        cleaned_tour = []
        for i, node in enumerate(tour):
            if i == 0 or i == len(tour) - 1:
                cleaned_tour.append(node)
            elif node not in seen:
                cleaned_tour.append(node)
                seen.add(node)
        # Final check: length should be n+1, all nodes visited once, start/end at 0
        n = len(distance_matrix)
        if len(cleaned_tour) != n + 1 or cleaned_tour[0] != 0 or cleaned_tour[-1] != 0 or set(cleaned_tour[:-1]) != set(range(n)):
            # fallback: return False to indicate invalid solution
            return False
        return cleaned_tour

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        # Return the distance matrix (edges) as expected by elkai
        return problem.edges 