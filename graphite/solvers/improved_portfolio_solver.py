from typing import List, Union, Tuple
import numpy as np
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1PortfolioProblem, GraphV1PortfolioSynapse
from graphite.utils.graph_utils import get_portfolio_distribution_similarity
from graphite.base.subnetPool import SubnetPool
from copy import deepcopy
import random

class ImprovedPortfolioSolver(BaseSolver):
    '''
    Improved portfolio solver that combines greedy approach with local search optimization.
    Minimizes swap count while meeting all constraints.
    '''
    def __init__(self, problem_types:List[GraphV1PortfolioProblem]=[GraphV1PortfolioProblem()]):
        super().__init__(problem_types=problem_types)
    
    async def solve(self, formatted_problem:GraphV1PortfolioProblem, future_id:int):
        """
        Improved portfolio reallocation solver with local search optimization.
        
        Args:
            formatted_problem: GraphV1PortfolioProblem with initial portfolios and constraints
            
        Returns:
            List of swaps: [ [portfolio_idx, from_subnet_idx, to_subnet_idx, from_num_alpha_tokens], ... ]
        """
        
        # Phase 1: Generate initial solution using greedy approach
        initial_swaps = self._greedy_phase(formatted_problem)
        
        # Phase 2: Optimize using local search
        optimized_swaps = self._local_search_optimization(formatted_problem, initial_swaps)
        
        return optimized_swaps
    
    def _greedy_phase(self, problem: GraphV1PortfolioProblem) -> List[List[int]]:
        """Phase 1: Generate initial solution using greedy approach"""
        
        def instantiate_pools(pools):
            current_pools: List[SubnetPool] = []
            for netuid, pool in enumerate(pools):
                current_pools.append(SubnetPool(pool[0], pool[1], netuid))
            return current_pools

        start_pools = instantiate_pools(problem.pools)
        initialPortfolios = deepcopy(problem.initialPortfolios)
        
        # Convert all alpha tokens to TAO
        total_tao = 0
        portfolio_tao = [0] * problem.n_portfolio
        portfolio_swaps = []
        
        for idx, portfolio in enumerate(initialPortfolios):
            for netuid, alpha_token in enumerate(portfolio):
                if alpha_token > 0:
                    emitted_tao = start_pools[netuid].swap_alpha_to_tao(alpha_token)
                    portfolio_swaps.append([idx, netuid, 0, int(alpha_token)])
                    total_tao += emitted_tao
                    portfolio_tao[idx] += emitted_tao

        # Redistribute TAO to meet constraints
        for netuid, constraint_type in enumerate(problem.constraintTypes):
            if netuid != 0:  # Skip subnet 0 (TAO)
                constraint_value = problem.constraintValues[netuid]
                tao_required = constraint_value / 100 * total_tao
                
                if constraint_type == "eq" or constraint_type == "ge":
                    for idx in range(len(portfolio_tao)):
                        tao_to_swap = min(portfolio_tao[idx], tao_required)
                        if tao_to_swap > 0:
                            alpha_emitted = start_pools[netuid].swap_tao_to_alpha(tao_to_swap)
                            portfolio_swaps.append([idx, 0, netuid, int(tao_to_swap)])
                            tao_required -= tao_to_swap
                            portfolio_tao[idx] -= tao_to_swap
                            if tao_required <= 0:
                                break

        return portfolio_swaps
    
    def _local_search_optimization(self, problem: GraphV1PortfolioProblem, initial_swaps: List[List[int]]) -> List[List[int]]:
        """Phase 2: Optimize swap sequence using local search"""
        
        current_swaps = deepcopy(initial_swaps)
        best_swaps = deepcopy(current_swaps)
        best_score = self._evaluate_solution(problem, best_swaps)
        
        max_iterations = 100
        no_improvement_limit = 20
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try different local search moves
            for move_type in ['merge', 'eliminate', 'reorder']:
                new_swaps = self._apply_local_move(current_swaps, move_type)
                new_score = self._evaluate_solution(problem, new_swaps)
                
                if new_score < best_score:  # Lower score is better
                    best_swaps = deepcopy(new_swaps)
                    best_score = new_score
                    current_swaps = deepcopy(new_swaps)
                    improved = True
                    break
            
            # If no improvement for several iterations, try random perturbation
            if not improved and iteration > no_improvement_limit:
                current_swaps = self._random_perturbation(current_swaps)
        
        return best_swaps
    
    def _apply_local_move(self, swaps: List[List[int]], move_type: str) -> List[List[int]]:
        """Apply a specific local search move"""
        
        if move_type == 'merge':
            return self._merge_consecutive_swaps(swaps)
        elif move_type == 'eliminate':
            return self._eliminate_redundant_swaps(swaps)
        elif move_type == 'reorder':
            return self._reorder_swaps(swaps)
        else:
            return swaps
    
    def _merge_consecutive_swaps(self, swaps: List[List[int]]) -> List[List[int]]:
        """Merge consecutive swaps between same portfolios"""
        if len(swaps) < 2:
            return swaps
        
        merged = []
        i = 0
        
        while i < len(swaps):
            if i + 1 < len(swaps):
                swap1, swap2 = swaps[i], swaps[i + 1]
                
                # Check if we can merge: same portfolio, same direction
                if (swap1[0] == swap2[0] and 
                    swap1[1] == swap2[1] and 
                    swap1[2] == swap2[2]):
                    # Merge the amounts
                    merged_swap = [swap1[0], swap1[1], swap1[2], swap1[3] + swap2[3]]
                    merged.append(merged_swap)
                    i += 2
                else:
                    merged.append(swap1)
                    i += 1
            else:
                merged.append(swaps[i])
                i += 1
        
        return merged
    
    def _eliminate_redundant_swaps(self, swaps: List[List[int]]) -> List[List[int]]:
        """Eliminate swaps that cancel each other out"""
        if len(swaps) < 2:
            return swaps
        
        # Group swaps by portfolio
        portfolio_swaps = {}
        for swap in swaps:
            portfolio = swap[0]
            if portfolio not in portfolio_swaps:
                portfolio_swaps[portfolio] = []
            portfolio_swaps[portfolio].append(swap)
        
        # Eliminate redundant swaps within each portfolio
        cleaned_swaps = []
        for portfolio, p_swaps in portfolio_swaps.items():
            # Create net flow for each subnet pair
            net_flows = {}
            for swap in p_swaps:
                key = (swap[1], swap[2])  # (from, to)
                if key not in net_flows:
                    net_flows[key] = 0
                net_flows[key] += swap[3]
            
            # Create final swaps from net flows
            for (from_subnet, to_subnet), amount in net_flows.items():
                if amount > 0:
                    cleaned_swaps.append([portfolio, from_subnet, to_subnet, amount])
        
        return cleaned_swaps
    
    def _reorder_swaps(self, swaps: List[List[int]]) -> List[List[int]]:
        """Reorder swaps to optimize execution order"""
        if len(swaps) < 2:
            return swaps
        
        # Group swaps by portfolio and sort by amount (largest first)
        portfolio_swaps = {}
        for swap in swaps:
            portfolio = swap[0]
            if portfolio not in portfolio_swaps:
                portfolio_swaps[portfolio] = []
            portfolio_swaps[portfolio].append(swap)
        
        # Sort swaps within each portfolio by amount (descending)
        for portfolio in portfolio_swaps:
            portfolio_swaps[portfolio].sort(key=lambda x: x[3], reverse=True)
        
        # Reconstruct with optimized order
        reordered = []
        for portfolio in sorted(portfolio_swaps.keys()):
            reordered.extend(portfolio_swaps[portfolio])
        
        return reordered
    
    def _random_perturbation(self, swaps: List[List[int]]) -> List[List[int]]:
        """Apply random perturbation to escape local optima"""
        if len(swaps) < 2:
            return swaps
        
        perturbed = deepcopy(swaps)
        
        # Randomly swap two adjacent swaps
        if len(perturbed) >= 2:
            i = random.randint(0, len(perturbed) - 2)
            perturbed[i], perturbed[i + 1] = perturbed[i + 1], perturbed[i]
        
        return perturbed
    
    def _evaluate_solution(self, problem: GraphV1PortfolioProblem, swaps: List[List[int]]) -> float:
        """Evaluate solution quality (lower is better)"""
        if not swaps:
            return float('inf')
        
        # Create a temporary synapse to evaluate
        temp_synapse = GraphV1PortfolioSynapse(problem=problem, solution=swaps)
        
        try:
            num_swaps, objective_score = get_portfolio_distribution_similarity(temp_synapse)
            
            # If solution is invalid, return high penalty
            if num_swaps == 1000000:
                return float('inf')
            
            # Combine swap count and objective score
            # Weight swap count more heavily as it's the primary optimization goal
            return num_swaps * 10 + (100 - objective_score)
            
        except Exception:
            return float('inf')
    
    def problem_transformations(self, problem: GraphV1PortfolioProblem):
        return problem 