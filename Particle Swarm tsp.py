import numpy as np
import random

# Function to calculate the total distance of a given route
def calculate_distance(route, dist_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i]][route[i + 1]]
    distance += dist_matrix[route[-1]][route[0]]  # Return to the starting point
    return distance

# Function to initialize particles (each particle represents a random permutation)
def initialize_particles(num_particles, num_cities):
    particles = []
    for _ in range(num_particles):
        route = random.sample(range(num_cities), num_cities)  # Generate a random route
        velocity = [0] * num_cities  # Initial velocity is zero
        particles.append({
            'position': route,
            'velocity': velocity,
            'pbest': route,
            'best_fitness': float('inf')  # Initialize with a large value
        })
    return particles

# Function to update velocity
def update_velocity(particle, gbest, w=0.5, c1=1.5, c2=1.5):
    velocity = []
    for i in range(len(particle['position'])):
        # Update velocity based on personal best and global best
        v = (w * particle['velocity'][i] +
             c1 * random.random() * (particle['pbest'][i] - particle['position'][i]) +
             c2 * random.random() * (gbest['position'][i] - particle['position'][i]))
        velocity.append(v)
    return velocity

# Function to update position
def update_position(particle):
    new_position = particle['position'][:]
    # Convert velocity into new positions (ensure valid permutations)
    for i in range(len(particle['position'])):
        new_position[i] = int(new_position[i] + particle['velocity'][i])
        new_position[i] = max(0, min(len(particle['position']) - 1, new_position[i]))
    return new_position

# Function to evaluate fitness of a particle (total distance)
def evaluate_fitness(particle, dist_matrix):
    return calculate_distance(particle['position'], dist_matrix)

# Repair function to ensure valid permutations
def repair_position(particle):
    seen = set()
    for i, city in enumerate(particle['position']):
        if city in seen:
            # Replace the duplicate city with the smallest missing city
            missing = set(range(len(particle['position']))) - seen
            particle['position'][i] = min(missing)
        seen.add(particle['position'][i])

# Main PSO algorithm for TSP
def pso_tsp(dist_matrix, num_particles=50, max_iterations=100):
    num_cities = len(dist_matrix)
    particles = initialize_particles(num_particles, num_cities)
    gbest = {'position': None, 'best_fitness': float('inf')}
    
    # Main loop
    for iteration in range(max_iterations):
        for particle in particles:
            # Evaluate fitness
            fitness = evaluate_fitness(particle, dist_matrix)
            
            # Update personal best if we found a better solution
            if fitness < particle['best_fitness']:
                particle['pbest'] = particle['position']
                particle['best_fitness'] = fitness
                
            # Update global best if we found a better solution
            if fitness < gbest['best_fitness']:
                gbest['position'] = particle['position']
                gbest['best_fitness'] = fitness
                
        # Update velocity and position for each particle
        for particle in particles:
            particle['velocity'] = update_velocity(particle, gbest)
            particle['position'] = update_position(particle)
            repair_position(particle)  # Ensure valid permutation
        
        # Print the best solution found so far
        print(f"Iteration {iteration + 1}: Best Distance = {gbest['best_fitness']}")
    
    return gbest

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
    
    # Run PSO to solve TSP
    gbest = pso_tsp(dist_matrix, num_particles=50, max_iterations=100)
    print(f"Best route: {gbest['position']}")
    print(f"Best distance: {gbest['best_fitness']}")
