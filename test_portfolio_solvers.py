#!/usr/bin/env python3
"""
Test script to compare different portfolio solvers.
"""

import asyncio
import time
import random
import numpy as np
from graphite.protocol import GraphV1PortfolioProblem, GraphV1PortfolioSynapse
from graphite.solvers import GreedyPortfolioSolver, ImprovedPortfolioSolver, LinearProgrammingPortfolioSolver
from graphite.utils.graph_utils import get_portfolio_distribution_similarity

def create_test_problem(n_portfolios=10, n_subnets=20):
    """Create a test portfolio problem"""
    
    # Create initial portfolios with random allocations
    initialPortfolios = []
    for _ in range(n_portfolios):
        portfolio = [random.randint(0, 1000) for _ in range(n_subnets)]
        initialPortfolios.append(portfolio)
    
    # Create constraint types (mix of eq, ge, le)
    constraintTypes = []
    for _ in range(n_subnets):
        constraintTypes.append(random.choice(["eq", "ge", "le"]))
    
    # Create constraint values
    constraintValues = []
    for ctype in constraintTypes:
        if ctype == "eq":
            constraintValues.append(random.uniform(0.5, 3.0))
        elif ctype == "ge":
            constraintValues.append(random.uniform(0.0, 5.0))
        elif ctype == "le":
            constraintValues.append(random.uniform(10.0, 100.0))
    
    # Create pools
    pools = [[random.randint(1000, 10000), random.randint(1000, 10000)] for _ in range(n_subnets)]
    
    return GraphV1PortfolioProblem(
        problem_type="PortfolioReallocation",
        n_portfolio=n_portfolios,
        initialPortfolios=initialPortfolios,
        constraintValues=constraintValues,
        constraintTypes=constraintTypes,
        pools=pools
    )

async def test_solver(solver, problem, solver_name):
    """Test a specific solver"""
    print(f"\nTesting {solver_name}...")
    
    start_time = time.time()
    try:
        solution = await solver.solve_problem(problem)
        solve_time = time.time() - start_time
        
        # Evaluate solution
        synapse = GraphV1PortfolioSynapse(problem=problem, solution=solution)
        num_swaps, objective_score = get_portfolio_distribution_similarity(synapse)
        
        print(f"  ✅ {solver_name}:")
        print(f"     Time: {solve_time:.3f}s")
        print(f"     Swaps: {num_swaps}")
        print(f"     Score: {objective_score:.2f}")
        
        return {
            'solver': solver_name,
            'time': solve_time,
            'swaps': num_swaps,
            'score': objective_score,
            'valid': num_swaps != 1000000
        }
        
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"  ❌ {solver_name} failed: {e}")
        return {
            'solver': solver_name,
            'time': solve_time,
            'swaps': 1000000,
            'score': 0,
            'valid': False,
            'error': str(e)
        }

async def main():
    """Main test function"""
    print("🚀 Portfolio Solver Comparison Test")
    print("=" * 50)
    
    # Create test problem
    problem = create_test_problem(n_portfolios=15, n_subnets=25)
    print(f"Test problem: {problem.n_portfolio} portfolios, {len(problem.initialPortfolios[0])} subnets")
    
    # Initialize solvers
    solvers = [
        (GreedyPortfolioSolver(), "GreedyPortfolioSolver"),
        (ImprovedPortfolioSolver(), "ImprovedPortfolioSolver"),
    ]
    
    # Try to add LP solver if scipy is available
    try:
        solvers.append((LinearProgrammingPortfolioSolver(), "LinearProgrammingPortfolioSolver"))
    except ImportError:
        print("⚠️  LinearProgrammingPortfolioSolver not available (scipy not installed)")
    
    # Test all solvers
    results = []
    for solver, name in solvers:
        result = await test_solver(solver, problem, name)
        results.append(result)
    
    # Print comparison
    print("\n" + "=" * 50)
    print("📊 RESULTS COMPARISON")
    print("=" * 50)
    
    valid_results = [r for r in results if r['valid']]
    if valid_results:
        best_swaps = min(valid_results, key=lambda x: x['swaps'])
        best_score = max(valid_results, key=lambda x: x['score'])
        fastest = min(valid_results, key=lambda x: x['time'])
        
        print(f"🏆 Best swap count: {best_swaps['solver']} ({best_swaps['swaps']} swaps)")
        print(f"🏆 Best score: {best_score['solver']} ({best_score['score']:.2f})")
        print(f"🏆 Fastest: {fastest['solver']} ({fastest['time']:.3f}s)")
    
    print("\nDetailed Results:")
    for result in results:
        status = "✅" if result['valid'] else "❌"
        print(f"{status} {result['solver']:30} | Swaps: {result['swaps']:6} | Score: {result['score']:6.2f} | Time: {result['time']:6.3f}s")

if __name__ == "__main__":
    asyncio.run(main()) 