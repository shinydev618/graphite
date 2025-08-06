# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV2ProblemMulti
import numpy as np
import time
import asyncio
import random

import bittensor as bt

class GeneticMTSPsolver(BaseSolver):
    """
    Genetic Algorithm for mTSP with evolutionary approach.
    Time complexity: O(generations × population × n)
    Quality: Can find very good solutions
    """
    
    def __init__(self, problem_types: List[GraphV2ProblemMulti] = [GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)
        
        # Genetic algorithm parameters
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.elite_size = 5
        self.tournament_size = 3

    def create_individual(self, n_nodes: int, n_salesmen: int, depots: List[int]) -> List[List[int]]:
        """Create a random individual (solution) for mTSP"""
        # Create list of all cities (excluding depots)
        cities = [i for i in range(n_nodes) if i not in depots]
        random.shuffle(cities)
        
        # Distribute cities among salesmen
        individual = []
        cities_per_salesman = len(cities) // n_salesmen
        remainder = len(cities) % n_salesmen
        
        start_idx = 0
        for i in range(n_salesmen):
            # Add depot at the beginning
            route = [depots[i]]
            
            # Add cities for this salesman
            if i < remainder:
                num_cities = cities_per_salesman + 1
            else:
                num_cities = cities_per_salesman
            
            end_idx = start_idx + num_cities
            route.extend(cities[start_idx:end_idx])
            
            # Add depot at the end
            route.append(depots[i])
            
            individual.append(route)
            start_idx = end_idx
        
        return individual

    def calculate_fitness(self, individual: List[List[int]], distance_matrix: np.ndarray) -> float:
        """Calculate fitness (inverse of total distance) for an individual"""
        total_distance = 0
        
        for route in individual:
            route_distance = 0
            for i in range(len(route) - 1):
                route_distance += distance_matrix[route[i]][route[i + 1]]
            total_distance += route_distance
        
        # Return inverse of distance (higher is better)
        return 1.0 / (total_distance + 1e-10)

    def tournament_selection(self, population: List[List[List[int]]], fitnesses: List[float]) -> List[List[int]]:
        """Select parent using tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_idx]

    def crossover(self, parent1: List[List[int]], parent2: List[List[int]], n_salesmen: int) -> List[List[int]]:
        """Perform crossover between two parents"""
        # Flatten parents to single lists
        flat_parent1 = []
        for route in parent1:
            flat_parent1.extend(route[1:-1])  # Exclude depots
        
        flat_parent2 = []
        for route in parent2:
            flat_parent2.extend(route[1:-1])  # Exclude depots
        
        # Order crossover (OX)
        size = len(flat_parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create child
        child = [-1] * size
        child[start:end] = flat_parent1[start:end]
        
        # Fill remaining positions from parent2
        remaining = [x for x in flat_parent2 if x not in child[start:end]]
        j = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
        
        # Reconstruct routes
        result = []
        cities_per_salesman = size // n_salesmen
        remainder = size % n_salesmen
        
        start_idx = 0
        for i in range(n_salesmen):
            route = [parent1[i][0]]  # Use depot from parent1
            
            if i < remainder:
                num_cities = cities_per_salesman + 1
            else:
                num_cities = cities_per_salesman
            
            end_idx = start_idx + num_cities
            route.extend(child[start_idx:end_idx])
            route.append(parent1[i][-1])  # Use depot from parent1
            
            result.append(route)
            start_idx = end_idx
        
        return result

    def mutate(self, individual: List[List[int]], n_nodes: int, depots: List[int]) -> List[List[int]]:
        """Perform mutation on an individual"""
        if random.random() > self.mutation_rate:
            return individual
        
        # Choose mutation type
        mutation_type = random.choice(['swap', 'reverse', 'insert'])
        
        if mutation_type == 'swap':
            # Swap two cities between routes
            if len(individual) >= 2:
                route1_idx = random.randint(0, len(individual) - 1)
                route2_idx = random.randint(0, len(individual) - 1)
                
                if route1_idx != route2_idx:
                    route1 = individual[route1_idx]
                    route2 = individual[route2_idx]
                    
                    if len(route1) > 2 and len(route2) > 2:
                        # Find non-depot positions
                        pos1 = random.randint(1, len(route1) - 2)
                        pos2 = random.randint(1, len(route2) - 2)
                        
                        # Swap cities
                        route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
        
        elif mutation_type == 'reverse':
            # Reverse a segment in a random route
            route_idx = random.randint(0, len(individual) - 1)
            route = individual[route_idx]
            
            if len(route) > 3:
                start, end = sorted(random.sample(range(1, len(route) - 1), 2))
                route[start:end+1] = route[start:end+1][::-1]
        
        elif mutation_type == 'insert':
            # Move a city from one route to another
            if len(individual) >= 2:
                route1_idx = random.randint(0, len(individual) - 1)
                route2_idx = random.randint(0, len(individual) - 1)
                
                if route1_idx != route2_idx:
                    route1 = individual[route1_idx]
                    route2 = individual[route2_idx]
                    
                    if len(route1) > 2:
                        # Remove city from route1
                        pos1 = random.randint(1, len(route1) - 2)
                        city = route1.pop(pos1)
                        
                        # Insert into route2
                        pos2 = random.randint(1, len(route2) - 1)
                        route2.insert(pos2, city)
        
        return individual

    def optimize_routes(self, individual: List[List[int]], distance_matrix: np.ndarray) -> List[List[int]]:
        """Apply 2-OPT optimization to each route"""
        optimized = []
        
        for route in individual:
            if len(route) <= 3:
                optimized.append(route)
                continue
            
            # Apply 2-OPT to the route (excluding depots)
            inner_route = route[1:-1]
            improved = True
            max_iterations = 10
            iteration = 0
            
            while improved and iteration < max_iterations:
                improved = False
                for i in range(len(inner_route) - 1):
                    for j in range(i + 2, len(inner_route)):
                        if j - i == 1:
                            continue
                        
                        # Calculate current distance
                        old_dist = (distance_matrix[inner_route[i]][inner_route[i+1]] + 
                                  distance_matrix[inner_route[j-1]][inner_route[j]])
                        
                        # Calculate new distance
                        new_dist = (distance_matrix[inner_route[i]][inner_route[j-1]] + 
                                  distance_matrix[inner_route[i+1]][inner_route[j]])
                        
                        if new_dist < old_dist:
                            # Reverse the segment
                            inner_route[i+1:j] = inner_route[j-1:i:-1]
                            improved = True
                            break
                    if improved:
                        break
                iteration += 1
            
            # Reconstruct route with depots
            optimized_route = [route[0]] + inner_route + [route[-1]]
            optimized.append(optimized_route)
        
        return optimized

    async def solve(self, formatted_problem, future_id: int) -> List[List[int]]:
        """
        Solve mTSP using Genetic Algorithm:
        1. Initialize population
        2. Evaluate fitness
        3. Selection, crossover, mutation
        4. Repeat for generations
        5. Return best solution
        """
        problem = formatted_problem
        distance_matrix = np.array(problem.edges)
        n_nodes = problem.n_nodes
        n_salesmen = problem.n_salesmen
        depots = problem.depots
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual(n_nodes, n_salesmen, depots)
            population.append(individual)
        
        best_fitness = 0
        best_individual = None
        
        # Main genetic algorithm loop
        for generation in range(self.generations):
            if self.future_tracker.get(future_id):
                return None
            
            # Calculate fitness for all individuals
            fitnesses = []
            for individual in population:
                fitness = self.calculate_fitness(individual, distance_matrix)
                fitnesses.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = [route[:] for route in individual]
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:self.elite_size]
            for idx in elite_indices:
                new_population.append([route[:] for route in population[idx]])
            
            # Generate rest of population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                child = self.crossover(parent1, parent2, n_salesmen)
                
                # Mutation
                child = self.mutate(child, n_nodes, depots)
                
                new_population.append(child)
            
            population = new_population
            
            # Apply local optimization to best individual occasionally
            if generation % 20 == 0 and best_individual:
                best_individual = self.optimize_routes(best_individual, distance_matrix)
        
        # Return best solution found
        if best_individual:
            return best_individual
        else:
            # Fallback: return first individual
            return population[0] if population else []

    def problem_transformations(self, problem: GraphV2ProblemMulti):
        """Transform problem to required format"""
        return problem

if __name__ == "__main__":
    # Test the Genetic mTSP solver
    import random
    
    # Create a test problem
    n_nodes = 100
    n_salesmen = 5
    depots = [0, 1, 2, 3, 4]
    
    # Create random distance matrix
    distance_matrix = np.random.rand(n_nodes, n_nodes) * 100
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Create test problem
    test_problem = GraphV2ProblemMulti(
        n_nodes=n_nodes,
        n_salesmen=n_salesmen,
        depots=depots,
        single_depot=False
    )
    test_problem.edges = distance_matrix
    
    solver = GeneticMTSPsolver(problem_types=[test_problem])
    
    start_time = time.time()
    routes = asyncio.run(solver.solve_problem(test_problem))
    end_time = time.time()
    
    print(f"Genetic mTSP Solution: {routes}")
    print(f"Time taken for {n_nodes} nodes, {n_salesmen} salesmen: {end_time - start_time:.3f} seconds")
    
    # Calculate total distance
    if routes:
        total_distance = 0
        for route in routes:
            route_distance = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            total_distance += route_distance
        print(f"Total distance: {total_distance:.2f}")
        print(f"Route lengths: {[len(route) for route in routes]}") 