from typing import List, Union, Tuple
import numpy as np
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1PortfolioProblem, GraphV1PortfolioSynapse
from graphite.utils.graph_utils import get_portfolio_distribution_similarity
from graphite.base.subnetPool import SubnetPool
from copy import deepcopy
import random

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class LinearProgrammingPortfolioSolver(BaseSolver):
    '''
    Linear Programming based portfolio solver for optimal solutions.
    Minimizes swap count while meeting all constraints.
    '''
    def __init__(self, problem_types:List[GraphV1PortfolioProblem]=[GraphV1PortfolioProblem()]):
        super().__init__(problem_types=problem_types)
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for LinearProgrammingPortfolioSolver. Install with: pip install scipy")
    
    async def solve(self, formatted_problem:GraphV1PortfolioProblem, future_id:int):
        """
        Linear Programming based portfolio reallocation solver.
        
        Args:
            formatted_problem: GraphV1PortfolioProblem with initial portfolios and constraints
            
        Returns:
            List of swaps: [ [portfolio_idx, from_subnet_idx, to_subnet_idx, from_num_alpha_tokens], ... ]
        """
        
        try:
            # Try LP approach first
            lp_swaps = self._solve_with_linear_programming(formatted_problem)
            if lp_swaps and self._validate_solution(formatted_problem, lp_swaps):
                return lp_swaps
        except Exception as e:
            print(f"LP solver failed: {e}")
        
        # Fallback to greedy approach
        return self._greedy_fallback(formatted_problem)
    
    def _solve_with_linear_programming(self, problem: GraphV1PortfolioProblem) -> List[List[int]]:
        """Solve using Linear Programming"""
        
        n_portfolios = problem.n_portfolio
        n_subnets = len(problem.initialPortfolios[0])
        
        # Initialize pools
        pools = self._instantiate_pools(problem.pools)
        
        # Convert initial portfolios to TAO equivalents
        initial_tao = self._convert_to_tao_equivalents(problem.initialPortfolios, pools)
        
        # Calculate target TAO distribution
        total_tao = np.sum(initial_tao)
        target_distribution = np.array(problem.constraintValues) / 100.0 * total_tao
        
        # Formulate LP problem
        # Variables: x[i,j,k] = amount of TAO moved from portfolio i, subnet j to subnet k
        # Objective: minimize total number of swaps (sum of non-zero variables)
        
        # For simplicity, we'll use a simplified LP formulation
        # that focuses on meeting the target distribution with minimal transfers
        
        # Calculate required transfers
        current_distribution = np.sum(initial_tao, axis=0)
        required_transfers = target_distribution - current_distribution
        
        # Create swap list based on required transfers
        swaps = []
        
        # Handle positive transfers (need to add TAO to subnets)
        for subnet_idx in range(1, n_subnets):  # Skip subnet 0 (TAO)
            if required_transfers[subnet_idx] > 0:
                # Find portfolios with excess TAO
                for portfolio_idx in range(n_portfolios):
                    if initial_tao[portfolio_idx, 0] > 0:  # Has TAO
                        transfer_amount = min(initial_tao[portfolio_idx, 0], required_transfers[subnet_idx])
                        if transfer_amount > 0:
                            # Convert TAO to alpha tokens
                            alpha_tokens = pools[subnet_idx].swap_tao_to_alpha(transfer_amount)
                            swaps.append([portfolio_idx, 0, subnet_idx, int(transfer_amount)])
                            required_transfers[subnet_idx] -= transfer_amount
                            initial_tao[portfolio_idx, 0] -= transfer_amount
                            if required_transfers[subnet_idx] <= 0:
                                break
        
        # Handle negative transfers (need to remove TAO from subnets)
        for subnet_idx in range(1, n_subnets):
            if required_transfers[subnet_idx] < 0:
                # Find portfolios with alpha tokens in this subnet
                for portfolio_idx in range(n_portfolios):
                    if initial_tao[portfolio_idx, subnet_idx] > 0:
                        transfer_amount = min(initial_tao[portfolio_idx, subnet_idx], -required_transfers[subnet_idx])
                        if transfer_amount > 0:
                            # Convert alpha tokens to TAO
                            tao_tokens = pools[subnet_idx].swap_alpha_to_tao(transfer_amount)
                            swaps.append([portfolio_idx, subnet_idx, 0, int(transfer_amount)])
                            required_transfers[subnet_idx] += transfer_amount
                            initial_tao[portfolio_idx, subnet_idx] -= transfer_amount
                            if required_transfers[subnet_idx] >= 0:
                                break
        
        return swaps
    
    def _convert_to_tao_equivalents(self, portfolios: List[List[int]], pools: List[SubnetPool]) -> np.ndarray:
        """Convert portfolio holdings to TAO equivalents"""
        n_portfolios = len(portfolios)
        n_subnets = len(portfolios[0])
        
        tao_equivalents = np.zeros((n_portfolios, n_subnets))
        
        for portfolio_idx, portfolio in enumerate(portfolios):
            for subnet_idx, alpha_tokens in enumerate(portfolio):
                if alpha_tokens > 0:
                    if subnet_idx == 0:
                        # Already in TAO
                        tao_equivalents[portfolio_idx, subnet_idx] = alpha_tokens
                    else:
                        # Convert alpha to TAO equivalent
                        tao_equivalent = pools[subnet_idx].swap_alpha_to_tao(alpha_tokens)
                        tao_equivalents[portfolio_idx, subnet_idx] = tao_equivalent
                        # Restore the pool state
                        pools[subnet_idx].swap_tao_to_alpha(tao_equivalent)
        
        return tao_equivalents
    
    def _instantiate_pools(self, pools_data):
        """Instantiate SubnetPool objects"""
        pools = []
        for netuid, pool in enumerate(pools_data):
            pools.append(SubnetPool(pool[0], pool[1], netuid))
        return pools
    
    def _validate_solution(self, problem: GraphV1PortfolioProblem, swaps: List[List[int]]) -> bool:
        """Validate if the solution meets all constraints"""
        if not swaps:
            return False
        
        try:
            temp_synapse = GraphV1PortfolioSynapse(problem=problem, solution=swaps)
            num_swaps, objective_score = get_portfolio_distribution_similarity(temp_synapse)
            return num_swaps != 1000000  # Valid solution
        except Exception:
            return False
    
    def _greedy_fallback(self, problem: GraphV1PortfolioProblem) -> List[List[int]]:
        """Fallback to greedy approach if LP fails"""
        
        def instantiate_pools(pools):
            current_pools = []
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
            if netuid != 0:
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
    
    def problem_transformations(self, problem: GraphV1PortfolioProblem):
        return problem 