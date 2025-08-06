from typing import List, Union, Tuple, Optional
import numpy as np
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1PortfolioProblem, GraphV1PortfolioSynapse
from graphite.utils.graph_utils import get_portfolio_distribution_similarity
from graphite.base.subnetPool import SubnetPool
from copy import deepcopy
import random
import time
import asyncio
import logging

# Set up logging
logger = logging.getLogger(__name__)

class AdaptivePortfolioSolver(BaseSolver):
    '''
    Robust adaptive portfolio solver with comprehensive error handling.
    Features:
    - Timeout protection at multiple levels
    - Graceful degradation with fallback strategies
    - Comprehensive error handling and logging
    - Input validation and sanitization
    - Performance monitoring
    '''
    
    def __init__(self, problem_types:List[GraphV1PortfolioProblem]=[GraphV1PortfolioProblem()]):
        super().__init__(problem_types=problem_types)
        self.max_total_time = 8.0  # Maximum 8 seconds total
        self.max_analysis_time = 1.0  # Maximum 1 second for analysis
        self.max_optimization_time = 5.0  # Maximum 5 seconds for optimization
        self.max_fine_tuning_time = 2.0  # Maximum 2 seconds for fine-tuning
        
    async def solve(self, formatted_problem:GraphV1PortfolioProblem, future_id:int):
        """
        Robust adaptive portfolio reallocation solver with comprehensive error handling.
        
        Args:
            formatted_problem: GraphV1PortfolioProblem with initial portfolios and constraints
            future_id: Future identifier for tracking
            
        Returns:
            List of swaps: [ [portfolio_idx, from_subnet_idx, to_subnet_idx, from_num_alpha_tokens], ... ]
        """
        
        start_time = time.time()
        logger.info(f"Starting adaptive portfolio solver (future_id: {future_id})")
        
        try:
            # Input validation
            if not self._validate_input(formatted_problem):
                logger.error("Input validation failed")
                return []
            
            # Phase 1: Quick analysis with timeout protection
            analysis_start = time.time()
            try:
                analysis = await asyncio.wait_for(
                    self._quick_analysis(formatted_problem),
                    timeout=self.max_analysis_time
                )
                analysis_time = time.time() - analysis_start
                logger.info(f"Analysis completed in {analysis_time:.3f}s")
                
            except asyncio.TimeoutError:
                logger.warning(f"Analysis timed out after {self.max_analysis_time}s, proceeding with basic optimization")
                analysis = None
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                analysis = None
            
            # Check if we have time for optimization
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_total_time * 0.8:
                logger.warning("Too much time spent on analysis, using emergency fallback")
                return self._emergency_fallback(formatted_problem)
            
            # Phase 2: Optimization with timeout protection
            optimization_start = time.time()
            try:
                swaps = await asyncio.wait_for(
                    self._robust_optimization(formatted_problem, analysis),
                    timeout=self.max_optimization_time
                )
                optimization_time = time.time() - optimization_start
                logger.info(f"Optimization completed in {optimization_time:.3f}s with {len(swaps)} swaps")
                
            except asyncio.TimeoutError:
                logger.warning(f"Optimization timed out after {self.max_optimization_time}s, using basic greedy")
                swaps = self._basic_greedy_fallback(formatted_problem)
            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                swaps = self._basic_greedy_fallback(formatted_problem)
            
            # Validate swaps before fine-tuning
            swaps = self._validate_and_clean_swaps(swaps)
            if not swaps:
                logger.warning("No valid swaps generated, using emergency fallback")
                return self._emergency_fallback(formatted_problem)
            
            # Check if we have time for fine-tuning
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_total_time * 0.9:
                logger.info("Skipping fine-tuning due to time constraints")
                return swaps
            
            # Phase 3: Fine-tuning with timeout protection
            try:
                final_swaps = await asyncio.wait_for(
                    self._robust_fine_tuning(formatted_problem, swaps),
                    timeout=self.max_fine_tuning_time
                )
                final_swaps = self._validate_and_clean_swaps(final_swaps)
                logger.info(f"Fine-tuning completed, final solution has {len(final_swaps)} swaps")
                return final_swaps
                
            except asyncio.TimeoutError:
                logger.warning(f"Fine-tuning timed out after {self.max_fine_tuning_time}s")
                return swaps
            except Exception as e:
                logger.error(f"Fine-tuning failed: {e}")
                return swaps
                
        except Exception as e:
            logger.error(f"Critical solver error: {e}")
            return self._emergency_fallback(formatted_problem)
        finally:
            total_time = time.time() - start_time
            logger.info(f"Solver completed in {total_time:.3f}s (future_id: {future_id})")
    
    def _validate_input(self, problem: GraphV1PortfolioProblem) -> bool:
        """Validate input problem"""
        try:
            # Check basic structure
            if not hasattr(problem, 'n_portfolio') or problem.n_portfolio <= 0:
                logger.error("Invalid n_portfolio")
                return False
                
            if not hasattr(problem, 'initialPortfolios') or not problem.initialPortfolios:
                logger.error("Missing or empty initialPortfolios")
                return False
                
            if not hasattr(problem, 'constraintTypes') or not problem.constraintTypes:
                logger.error("Missing or empty constraintTypes")
                return False
                
            if not hasattr(problem, 'constraintValues') or not problem.constraintValues:
                logger.error("Missing or empty constraintValues")
                return False
                
            if not hasattr(problem, 'pools') or not problem.pools:
                logger.error("Missing or empty pools")
                return False
            
            # Check dimensions
            n_subnets = len(problem.initialPortfolios[0])
            if len(problem.constraintTypes) != n_subnets:
                logger.error(f"Constraint types length {len(problem.constraintTypes)} != subnets {n_subnets}")
                return False
                
            if len(problem.constraintValues) != n_subnets:
                logger.error(f"Constraint values length {len(problem.constraintValues)} != subnets {n_subnets}")
                return False
                
            if len(problem.pools) != n_subnets:
                logger.error(f"Pools length {len(problem.pools)} != subnets {n_subnets}")
                return False
            
            # Check portfolio consistency
            for i, portfolio in enumerate(problem.initialPortfolios):
                if len(portfolio) != n_subnets:
                    logger.error(f"Portfolio {i} length {len(portfolio)} != subnets {n_subnets}")
                    return False
                if any(amount < 0 for amount in portfolio):
                    logger.error(f"Portfolio {i} has negative amounts")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    async def _quick_analysis(self, problem: GraphV1PortfolioProblem) -> Optional[dict]:
        """Quick analysis of portfolio state with error handling"""
        try:
            # Initialize pools safely
            pools = self._instantiate_pools(problem.pools)
            
            # Calculate current distribution safely
            current_distribution = np.sum(problem.initialPortfolios, axis=0)
            total_tokens = np.sum(current_distribution)
            
            if total_tokens <= 0:
                logger.warning("Total tokens is zero or negative")
                return None
            
            # Calculate target distribution
            target_distribution = np.array(problem.constraintValues) / 100.0 * total_tokens
            
            # Calculate required changes
            required_changes = target_distribution - current_distribution
            
            return {
                'current_distribution': current_distribution,
                'target_distribution': target_distribution,
                'required_changes': required_changes,
                'total_tokens': total_tokens,
                'pools': pools
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return None
    
    async def _robust_optimization(self, problem: GraphV1PortfolioProblem, analysis: Optional[dict]) -> List[List[int]]:
        """Robust optimization with multiple fallback strategies"""
        try:
            # Try advanced optimization first
            if analysis and analysis.get('total_tokens', 0) > 0:
                swaps = self._advanced_optimization(problem, analysis)
                if swaps:
                    return swaps
            
            # Fallback to simple greedy
            swaps = self._simple_greedy_optimization(problem)
            if swaps:
                return swaps
            
            # Emergency fallback
            return self._emergency_fallback(problem)
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return self._emergency_fallback(problem)
    
    def _advanced_optimization(self, problem: GraphV1PortfolioProblem, analysis: dict) -> List[List[int]]:
        """Advanced optimization using analysis results"""
        try:
            swaps = []
            portfolios = deepcopy(problem.initialPortfolios)
            required_changes = analysis['required_changes']
            
            # Find deficits and surpluses
            deficits = []
            surpluses = []
            
            for subnet_idx, change in enumerate(required_changes):
                if change > 0:
                    deficits.append((subnet_idx, change))
                elif change < 0:
                    surpluses.append((subnet_idx, abs(change)))
            
            # Sort by magnitude
            deficits.sort(key=lambda x: x[1], reverse=True)
            surpluses.sort(key=lambda x: x[1], reverse=True)
            
            # Execute transfers
            for deficit_subnet, deficit_amount in deficits:
                if deficit_amount <= 0:
                    continue
                    
                for surplus_subnet, surplus_amount in surpluses:
                    if surplus_amount <= 0:
                        continue
                        
                    # Find best portfolio for transfer
                    best_portfolio = self._find_best_transfer_portfolio(
                        portfolios, surplus_subnet, deficit_subnet
                    )
                    
                    if best_portfolio >= 0:
                        transfer_amount = self._calculate_optimal_transfer(
                            portfolios[best_portfolio], surplus_subnet, deficit_subnet,
                            deficit_amount, surplus_amount
                        )
                        
                        if transfer_amount > 0:
                            swaps.append([
                                int(best_portfolio),
                                surplus_subnet,
                                deficit_subnet,
                                int(transfer_amount)
                            ])
                            
                            # Update portfolios
                            portfolios[best_portfolio][surplus_subnet] -= transfer_amount
                            portfolios[best_portfolio][deficit_subnet] += transfer_amount
                            
                            # Update amounts
                            deficit_amount -= transfer_amount
                            surplus_amount -= transfer_amount
                            
                            if deficit_amount <= 0:
                                break
                
                if deficit_amount <= 0:
                    break
            
            return swaps
            
        except Exception as e:
            logger.error(f"Advanced optimization error: {e}")
            return []
    
    def _find_best_transfer_portfolio(self, portfolios: List[List[int]], from_subnet: int, to_subnet: int) -> int:
        """Find the best portfolio for a transfer"""
        best_portfolio = -1
        best_score = -1
        
        for portfolio_idx, portfolio in enumerate(portfolios):
            if portfolio[from_subnet] <= 0:
                continue
                
            # Score based on available amount and current balance
            available = portfolio[from_subnet]
            current_balance = portfolio[to_subnet]
            score = available * (1 + current_balance / max(sum(portfolio), 1))
            
            if score > best_score:
                best_score = score
                best_portfolio = portfolio_idx
        
        return best_portfolio
    
    def _calculate_optimal_transfer(self, portfolio: List[int], from_subnet: int, to_subnet: int,
                                  deficit_amount: float, surplus_amount: float) -> int:
        """Calculate optimal transfer amount"""
        available = portfolio[from_subnet]
        
        # Don't transfer more than 80% of available to avoid empty portfolios
        max_transfer = min(
            available,
            deficit_amount,
            surplus_amount,
            int(available * 0.8)
        )
        
        return int(max_transfer)
    
    def _simple_greedy_optimization(self, problem: GraphV1PortfolioProblem) -> List[List[int]]:
        """Simple greedy optimization using direct token transfers"""
        try:
            swaps = []
            portfolios = deepcopy(problem.initialPortfolios)
            
            # Calculate target distribution
            total_tokens = sum(sum(portfolio) for portfolio in portfolios)
            if total_tokens <= 0:
                return []
                
            target_distribution = [val / 100.0 * total_tokens for val in problem.constraintValues]
            
            # Find deficits and surpluses
            current_distribution = np.sum(portfolios, axis=0)
            deficits = []
            surpluses = []
            
            for subnet_idx, (current, target) in enumerate(zip(current_distribution, target_distribution)):
                if current < target * 0.95:  # 5% tolerance
                    deficits.append((subnet_idx, target - current))
                elif current > target * 1.05:  # 5% tolerance
                    surpluses.append((subnet_idx, current - target))
            
            # Sort by magnitude
            deficits.sort(key=lambda x: x[1], reverse=True)
            surpluses.sort(key=lambda x: x[1], reverse=True)
            
            # Execute transfers
            for deficit_subnet, deficit_amount in deficits:
                if deficit_amount <= 0:
                    continue
                    
                for surplus_subnet, surplus_amount in surpluses:
                    if surplus_amount <= 0:
                        continue
                        
                    # Find portfolio with most tokens in surplus subnet
                    best_portfolio = -1
                    max_tokens = 0
                    
                    for portfolio_idx in range(problem.n_portfolio):
                        if portfolios[portfolio_idx][surplus_subnet] > max_tokens:
                            max_tokens = portfolios[portfolio_idx][surplus_subnet]
                            best_portfolio = portfolio_idx
                    
                    if best_portfolio >= 0 and max_tokens > 0:
                        # Calculate transfer amount - ensure it's an integer
                        transfer_amount = min(
                            max_tokens,
                            surplus_amount,
                            deficit_amount,
                            int(max_tokens * 0.8)  # Don't transfer more than 80% to avoid empty portfolios
                        )
                        
                        # Convert to integer to avoid floating point issues
                        transfer_amount = int(transfer_amount)
                        
                        if transfer_amount > 0:
                            # Execute transfer
                            swaps.append([
                                int(best_portfolio),
                                surplus_subnet,
                                deficit_subnet,
                                int(transfer_amount)
                            ])
                            
                            # Update portfolios
                            portfolios[best_portfolio][surplus_subnet] -= transfer_amount
                            portfolios[best_portfolio][deficit_subnet] += transfer_amount
                            
                            # Update deficits and surpluses
                            deficit_amount -= transfer_amount
                            surplus_amount -= transfer_amount
                            
                            if deficit_amount <= 0:
                                break
                    
                    if deficit_amount <= 0:
                        break
            
            return swaps
            
        except Exception as e:
            logger.error(f"Simple greedy optimization error: {e}")
            return []
    
    def _basic_greedy_fallback(self, problem: GraphV1PortfolioProblem) -> List[List[int]]:
        """Basic greedy fallback strategy"""
        try:
            swaps = []
            portfolios = deepcopy(problem.initialPortfolios)
            
            # Simple strategy: move tokens from first portfolio to meet constraints
            for portfolio_idx in range(min(1, problem.n_portfolio)):  # Only use first portfolio
                portfolio = portfolios[portfolio_idx]
                portfolio_total = sum(portfolio)
                
                if portfolio_total <= 0:
                    continue
                
                for subnet_idx in range(1, len(portfolio)):  # Skip TAO (subnet 0)
                    constraint_type = problem.constraintTypes[subnet_idx]
                    constraint_value = problem.constraintValues[subnet_idx]
                    current_percentage = (portfolio[subnet_idx] / portfolio_total) * 100
                    
                    if constraint_type == "ge" and current_percentage < constraint_value:
                        needed_amount = int((constraint_value / 100) * portfolio_total - portfolio[subnet_idx])
                        if needed_amount > 0 and portfolio[0] >= needed_amount:
                            swaps.append([portfolio_idx, 0, subnet_idx, int(needed_amount)])
                    
                    elif constraint_type == "le" and current_percentage > constraint_value:
                        excess_amount = int(portfolio[subnet_idx] - (constraint_value / 100) * portfolio_total)
                        if excess_amount > 0:
                            swaps.append([portfolio_idx, subnet_idx, 0, int(excess_amount)])
            
            return swaps
            
        except Exception as e:
            logger.error(f"Basic greedy fallback error: {e}")
            return []
    
    def _emergency_fallback(self, problem: GraphV1PortfolioProblem) -> List[List[int]]:
        """Emergency fallback - return empty solution"""
        logger.warning("Using emergency fallback - returning empty solution")
        return []
    
    async def _robust_fine_tuning(self, problem: GraphV1PortfolioProblem, current_swaps: List[List[int]]) -> List[List[int]]:
        """Robust fine-tuning with error handling"""
        try:
            swaps = current_swaps.copy()
            portfolios = deepcopy(problem.initialPortfolios)
            
            # Apply current swaps safely
            for swap in current_swaps:
                if len(swap) == 4:
                    portfolio_idx, from_subnet, to_subnet, amount = swap
                    if (0 <= portfolio_idx < len(portfolios) and 
                        0 <= from_subnet < len(portfolios[0]) and 
                        0 <= to_subnet < len(portfolios[0]) and
                        amount > 0 and
                        portfolios[portfolio_idx][from_subnet] >= amount):
                        
                        portfolios[portfolio_idx][from_subnet] -= amount
                        portfolios[portfolio_idx][to_subnet] += amount
            
            # Quick constraint checking and fixing
            max_fixes = 3  # Reduced limit for speed
            fixes_applied = 0
            
            for portfolio_idx in range(min(problem.n_portfolio, 2)):  # Only check first 2 portfolios
                if fixes_applied >= max_fixes:
                    break
                    
                portfolio = portfolios[portfolio_idx]
                portfolio_total = sum(portfolio)
                
                if portfolio_total == 0:
                    continue
                
                for subnet_idx in range(1, min(len(portfolio), len(problem.constraintTypes))):
                    if fixes_applied >= max_fixes:
                        break
                        
                    constraint_type = problem.constraintTypes[subnet_idx]
                    constraint_value = problem.constraintValues[subnet_idx]
                    current_percentage = (portfolio[subnet_idx] / portfolio_total) * 100
                    
                    # Only fix if constraint violation is significant (>5%)
                    if constraint_type == "eq" and abs(current_percentage - constraint_value) > 5.0:
                        target_amount = (constraint_value / 100) * portfolio_total
                        needed_amount = target_amount - portfolio[subnet_idx]
                        
                        if needed_amount > 0 and portfolio[0] >= needed_amount:
                            swaps.append([portfolio_idx, 0, subnet_idx, int(needed_amount)])
                            fixes_applied += 1
                        elif needed_amount < 0:
                            swaps.append([portfolio_idx, subnet_idx, 0, int(abs(needed_amount))])
                            fixes_applied += 1
                            
                    elif constraint_type == "ge" and current_percentage < constraint_value - 5.0:
                        target_amount = (constraint_value / 100) * portfolio_total
                        needed_amount = target_amount - portfolio[subnet_idx]
                        
                        if needed_amount > 0 and portfolio[0] >= needed_amount:
                            swaps.append([portfolio_idx, 0, subnet_idx, int(needed_amount)])
                            fixes_applied += 1
                            
                    elif constraint_type == "le" and current_percentage > constraint_value + 5.0:
                        target_amount = (constraint_value / 100) * portfolio_total
                        excess_amount = portfolio[subnet_idx] - target_amount
                        
                        if excess_amount > 0:
                            swaps.append([portfolio_idx, subnet_idx, 0, int(excess_amount)])
                            fixes_applied += 1
            
            return swaps
            
        except Exception as e:
            logger.error(f"Fine-tuning error: {e}")
            return current_swaps
    
    def _validate_and_clean_swaps(self, swaps: List[List[int]]) -> List[List[int]]:
        """Validate and clean swaps to ensure proper format"""
        cleaned_swaps = []
        
        try:
            for swap in swaps:
                if len(swap) == 4:
                    # Ensure all values are integers
                    portfolio_idx = int(swap[0])
                    from_subnet = int(swap[1])
                    to_subnet = int(swap[2])
                    amount = int(swap[3])
                    
                    # Validate ranges
                    if (0 <= portfolio_idx < 100 and  # Reasonable portfolio range
                        0 <= from_subnet < 100 and   # Reasonable subnet range
                        0 <= to_subnet < 100 and     # Reasonable subnet range
                        amount > 0):                 # Positive amount
                        cleaned_swaps.append([portfolio_idx, from_subnet, to_subnet, amount])
                    else:
                        logger.warning(f"Invalid swap range: {swap}")
                else:
                    logger.warning(f"Invalid swap length: {swap}")
            
            return cleaned_swaps
            
        except Exception as e:
            logger.error(f"Swap validation error: {e}")
            return []
    
    def _instantiate_pools(self, pools_data):
        """Instantiate SubnetPool objects with error handling"""
        try:
            pools = []
            for netuid, pool in enumerate(pools_data):
                if len(pool) == 2:
                    pools.append(SubnetPool(pool[0], pool[1], netuid))
                else:
                    logger.warning(f"Invalid pool data at index {netuid}: {pool}")
                    pools.append(SubnetPool(1000, 1000, netuid))  # Default pool
            return pools
        except Exception as e:
            logger.error(f"Pool instantiation error: {e}")
            return []
    
    def problem_transformations(self, problem: GraphV1PortfolioProblem):
        return problem
    
    async def solve_problem(self, problem: GraphV1PortfolioProblem):
        """Wrapper method to match miner's expected interface"""
        return await self.solve(problem, future_id=1)