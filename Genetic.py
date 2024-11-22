import random
import numpy as np

# Function to calculate the total distance of a given route
def calculate_distance(route, dist_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i]][route[i + 1]]
    distance += dist_matrix[route[-1]][route[0]]  # Return to the starting point
    return distance

# Function to create an initial population of random routes
def create_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        route = random.sample(range(num_cities), num_cities)  # Generate a random route
        population.append(route)
    return population

# Fitness function (inverse of the total distance)
def fitness(route, dist_matrix):
    return 1 / calculate_distance(route, dist_matrix)

# Selection function: Tournament selection
def tournament_selection(population, dist_matrix, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    tournament_fitness = [fitness(route, dist_matrix) for route in tournament]
    winner_idx = tournament_fitness.index(max(tournament_fitness))
    return tournament[winner_idx]

# Crossover function: Order crossover (OX)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))  # Randomly choose crossover points
    child = [-1] * size
    
    # Copy a subsequence from parent1
    child[start:end+1] = parent1[start:end+1]
    
    # Fill the remaining positions with the genes from parent2 in order
    current_pos = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[current_pos] in child:
                current_pos += 1
            child[i] = parent2[current_pos]
    
    return child

# Mutation function: Swap mutation
def mutate(route, mutation_rate=0.01):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)  # Randomly choose two indices
        route[idx1], route[idx2] = route[idx2], route[idx1]  # Swap their positions
    return route

# Main Genetic Algorithm function
def genetic_algorithm(dist_matrix, pop_size=100, generations=500, mutation_rate=0.01, tournament_size=3):
    num_cities = len(dist_matrix)
    population = create_population(pop_size, num_cities)
    
    best_route = None
    best_distance = float('inf')

    for generation in range(generations):
        # Evaluate fitness
        population_fitness = [fitness(route, dist_matrix) for route in population]
        
        # Select the best solution in the population
        current_best_idx = population_fitness.index(max(population_fitness))
        current_best_route = population[current_best_idx]
        current_best_distance = calculate_distance(current_best_route, dist_matrix)
        
        if current_best_distance < best_distance:
            best_route = current_best_route
            best_distance = current_best_distance
        
        # Create the next generation
        new_population = []
        
        # Elitism: Carry the best individual to the next generation
        new_population.append(best_route)
        
        # Create offspring using selection, crossover, and mutation
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, dist_matrix, tournament_size)
            parent2 = tournament_selection(population, dist_matrix, tournament_size)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population  # Update the population for the next generation
        
        # Optionally print the progress
        print(f"Generation {generation+1}: Best Distance = {best_distance}")
    
    return best_route, best_distance

# Example usage
if __name__ == "__main__":
    # Example distance matrix (symmetric, non-negative distances)
    dist_matrix = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 5],
        [20, 25, 30, 0, 15],
        [25, 30, 5, 15, 0]
    ])
    
    best_route, best_distance = genetic_algorithm(dist_matrix, pop_size=50, generations=500)
    print(f"Best route: {best_route}")
    print(f"Best distance: {best_distance}")
