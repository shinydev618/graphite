from typing import List
import numpy as np
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1PortfolioProblem, GraphV1PortfolioSynapse
from graphite.base.subnetPool import SubnetPool
from copy import deepcopy
import logging

class TargetedPortfolioSolver(BaseSolver):
    """
    Direct constraint-driven portfolio solver that minimizes swaps by targeting
    surplus-to-deficit movements within each portfolio.
    """
    def __init__(self, problem_types:List[GraphV1PortfolioProblem]=[GraphV1PortfolioProblem()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem:GraphV1PortfolioProblem, future_id:int):
        problem = formatted_problem
        n_portfolios = problem.n_portfolio
        n_subnets = len(problem.initialPortfolios[0])
        constraint_values = np.array(problem.constraintValues)
        constraint_types = problem.constraintTypes
        pools = self._instantiate_pools(problem.pools)
        initial_portfolios = deepcopy(problem.initialPortfolios)

        logging.info(f"Solving portfolio problem: {n_portfolios} portfolios, {n_subnets} subnets")
        logging.info(f"Constraints: {constraint_values} ({constraint_types})")

        # Step 1: Calculate current TAO-equivalent distribution
        current_tao_distribution = self._calculate_current_tao_distribution(
            initial_portfolios, pools, n_portfolios, n_subnets
        )
        
        logging.info(f"Current TAO distribution: {current_tao_distribution}")
        logging.info(f"Target TAO distribution: {constraint_values}")

        # Step 2: Calculate required changes to meet constraints
        required_changes = self._calculate_required_changes(
            current_tao_distribution, constraint_values, constraint_types
        )
        
        logging.info(f"Required changes: {required_changes}")

        # Step 3: Generate swaps to achieve required changes
        swaps = self._generate_swaps(
            initial_portfolios, pools, required_changes, n_portfolios, n_subnets
        )

        logging.info(f"Generated {len(swaps)} swaps")
        return swaps

    def _calculate_current_tao_distribution(self, portfolios, pools, n_portfolios, n_subnets):
        """Calculate current TAO-equivalent distribution across all portfolios"""
        total_tao_per_subnet = np.zeros(n_subnets)
        
        for portfolio in portfolios:
            for subnet_idx, alpha_amount in enumerate(portfolio):
                if alpha_amount > 0:
                    if subnet_idx == 0:  # TAO subnet
                        total_tao_per_subnet[subnet_idx] += alpha_amount
                    else:  # Alpha subnet - convert to TAO
                        tao_equiv = pools[subnet_idx].swap_alpha_to_tao(alpha_amount)
                        total_tao_per_subnet[subnet_idx] += tao_equiv
                        # Restore pool state
                        pools[subnet_idx].swap_tao_to_alpha(tao_equiv)
        
        total_tao = np.sum(total_tao_per_subnet)
        if total_tao > 0:
            return (total_tao_per_subnet / total_tao) * 100
        else:
            return np.zeros(n_subnets)

    def _calculate_required_changes(self, current_dist, target_dist, constraint_types):
        """Calculate how much each subnet needs to change to meet constraints"""
        changes = np.zeros(len(current_dist))
        
        for i, (current, target, constraint_type) in enumerate(zip(current_dist, target_dist, constraint_types)):
            if constraint_type == "eq":
                changes[i] = target - current
            elif constraint_type == "ge":
                if current < target:
                    changes[i] = target - current
                else:
                    changes[i] = 0
            elif constraint_type == "le":
                if current > target:
                    changes[i] = target - current
                else:
                    changes[i] = 0
        
        return changes

    def _generate_swaps(self, portfolios, pools, required_changes, n_portfolios, n_subnets):
        """Generate swaps to achieve required changes"""
        swaps = []
        
        # Identify subnets that need to increase (deficit) and decrease (surplus)
        deficit_subnets = np.where(required_changes > 1e-6)[0]
        surplus_subnets = np.where(required_changes < -1e-6)[0]
        
        logging.info(f"Deficit subnets: {deficit_subnets}")
        logging.info(f"Surplus subnets: {surplus_subnets}")
        
        # For each portfolio, try to move tokens from surplus to deficit subnets
        for portfolio_idx in range(n_portfolios):
            portfolio = portfolios[portfolio_idx]
            
            # Find which subnets in this portfolio have surplus
            portfolio_surplus = []
            for subnet_idx in surplus_subnets:
                if portfolio[subnet_idx] > 0:
                    portfolio_surplus.append(subnet_idx)
            
            # Find which subnets in this portfolio need more tokens
            portfolio_deficit = []
            for subnet_idx in deficit_subnets:
                portfolio_deficit.append(subnet_idx)
            
            # Create swaps from surplus to deficit
            for from_subnet in portfolio_surplus:
                available_alpha = portfolio[from_subnet]
                if available_alpha <= 0:
                    continue
                
                for to_subnet in portfolio_deficit:
                    if from_subnet == to_subnet:
                        continue
                    
                    # Calculate how much we can transfer
                    if from_subnet == 0:  # From TAO
                        transfer_amount = min(available_alpha, abs(required_changes[to_subnet]))
                        if transfer_amount > 1e-6:
                            swaps.append([int(portfolio_idx), int(0), int(to_subnet), int(transfer_amount)])
                            logging.info(f"Swap: portfolio {portfolio_idx}, TAO({transfer_amount}) -> subnet {to_subnet}")
                            available_alpha -= transfer_amount
                            required_changes[to_subnet] -= transfer_amount
                            required_changes[0] += transfer_amount
                    
                    else:  # From Alpha subnet
                        # Convert available alpha to TAO
                        tao_equiv = pools[from_subnet].swap_alpha_to_tao(available_alpha)
                        pools[from_subnet].swap_tao_to_alpha(tao_equiv)  # Restore
                        
                        transfer_tao = min(tao_equiv, abs(required_changes[to_subnet]))
                        if transfer_tao > 1e-6:
                            # Calculate how much alpha this represents
                            alpha_amount = available_alpha
                            swaps.append([int(portfolio_idx), int(from_subnet), int(to_subnet), int(alpha_amount)])
                            logging.info(f"Swap: portfolio {portfolio_idx}, subnet {from_subnet}({alpha_amount}) -> subnet {to_subnet}")
                            available_alpha = 0
                            required_changes[to_subnet] -= transfer_tao
                            required_changes[from_subnet] += transfer_tao
                            break  # Move to next from_subnet
                    
                    if available_alpha <= 1e-6:
                        break  # No more tokens to transfer from this subnet
        
        return swaps

    def _instantiate_pools(self, pools_data):
        pools = []
        for netuid, pool in enumerate(pools_data):
            pools.append(SubnetPool(pool[0], pool[1], netuid))
        return pools

    def problem_transformations(self, problem: GraphV1PortfolioProblem):
        return problem 