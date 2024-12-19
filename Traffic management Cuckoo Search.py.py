import numpy as np

# Define parameters for the problem
N = 20  # Population size (number of nests)
D = 5   # Number of traffic parameters to optimize (e.g., signal timings)
MAX_ITER = 1000  # Maximum number of iterations
pa = 0.25  # Probability of host nest abandonment
alpha = 1.0  # Step size for Levy flights

# Objective function: minimize total waiting time (as an example)
def fitness(solution):
    # For simplicity, let's assume the objective is to minimize the sum of signal timings
    return np.sum(solution)

# Levy flight step size
def levy_flight(Lambda):
    sigma_u = np.power(np.random.rand(), -1.0 / Lambda)
    sigma_v = np.random.rand()
    step = sigma_u * np.sin(np.pi * sigma_v) / np.power(np.abs(np.cos(np.pi * sigma_v)), 1.0 / Lambda)
    return step

# Generate initial population of solutions (random traffic signal timings)
def initialize_population():
    return np.random.rand(N, D) * 10  # Random timings between 0 and 10 minutes for each signal

# Cuckoo Search Algorithm
def cuckoo_search():
    population = initialize_population()
    fitness_values = np.array([fitness(ind) for ind in population])
    
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx]
    best_fitness = fitness_values[best_idx]
    
    for iteration in range(MAX_ITER):
        # Generate new solutions by Levy flight
        for i in range(N):
            # Apply Levy flight to explore new solutions
            step_size = levy_flight(1.5)  # Lambda parameter
            new_solution = population[i] + alpha * step_size * np.random.randn(D)
            
            # Ensure new solution is within bounds (0 to 10 minutes for each signal)
            new_solution = np.clip(new_solution, 0, 10)
            
            new_fitness = fitness(new_solution)
            
            # If the new solution is better, replace the old one
            if new_fitness < fitness_values[i]:
                population[i] = new_solution
                fitness_values[i] = new_fitness
        
        # Find the best solution so far
        best_idx = np.argmin(fitness_values)
        current_best_solution = population[best_idx]
        current_best_fitness = fitness_values[best_idx]
        
        # Update the global best solution
        if current_best_fitness < best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness
        
        # Abandon some nests (set some solutions to random values)
        for i in range(N):
            if np.random.rand() < pa:
                population[i] = np.random.rand(D) * 10  # Reset to random solution
                fitness_values[i] = fitness(population[i])
        
        print(f"Iteration {iteration+1}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness

# Running the Cuckoo Search Algorithm
best_solution, best_fitness = cuckoo_search()

# Output the optimal solution found
print("Optimal Traffic Signal Timings (in minutes):")
print(best_solution)
print(f"Minimal Waiting Time: {best_fitness}")
