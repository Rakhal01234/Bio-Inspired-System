import numpy as np
import random
import matplotlib.pyplot as plt

# Target function: f(x) = x^2 + 2x + 1
def target_function(x):
    return x**2 + 2*x + 1

# Generate noisy data
def generate_noisy_data(n_points=100):
    x_vals = np.linspace(-10, 10, n_points)
    y_vals = target_function(x_vals) + np.random.normal(0, 5, size=n_points)  # Add noise
    return x_vals, y_vals

class Expression:
    def __init__(self, expression=None):
        self.expression = expression if expression else self.random_expression()

    def random_expression(self):
        operators = ['+', '-', '*', '/']
        variables = ['x', '2', '3', '4']
        length = random.randint(3, 5)
        
        # Start with a variable or number
        expression = [random.choice(variables)]
        
        # Build the expression
        for _ in range(length - 1):
            expression.append(random.choice(operators))
            expression.append(random.choice(variables))
        
        return ''.join(expression)

    def evaluate(self, x):
        expr = self.expression.replace('x', str(x))
        try:
            return eval(expr)
        except ZeroDivisionError:
            return float('inf')  # Handle division by zero gracefully
        except Exception as e:
            print(f"Error evaluating expression '{self.expression}': {e}")
            return float('inf')  # Return a large error value for invalid expressions

def fitness(individual, x_vals, y_vals):
    error = 0
    for x, y_true in zip(x_vals, y_vals):
        y_pred = individual.evaluate(x)
        error += (y_pred - y_true) ** 2
    return error / len(x_vals)

def tournament_selection(population, fitness_vals, tournament_size=3):
    selected_parents = []
    for _ in range(2):
        tournament = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_vals[i] for i in tournament]
        winner = tournament[tournament_fitness.index(min(tournament_fitness))]
        selected_parents.append(population[winner])
    return selected_parents

def crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1.expression), len(parent2.expression)) - 1)
    child1_expr = parent1.expression[:crossover_point] + parent2.expression[crossover_point:]
    child2_expr = parent2.expression[:crossover_point] + parent1.expression[crossover_point:]
    
    child1 = Expression(child1_expr)
    child2 = Expression(child2_expr)
    
    return child1, child2

def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutation_point = random.randint(0, len(individual.expression) - 1)
        mutated_char = random.choice(['+', '-', '*', '/', 'x', '2', '3', '4'])
        individual.expression = individual.expression[:mutation_point] + mutated_char + individual.expression[mutation_point + 1:]
    return individual

def symbolic_regression(pop_size, generations, mutation_rate=0.1, tournament_size=3):
    x_vals, y_vals = generate_noisy_data()  # Generate noisy data
    
    # Initialize population
    population = [Expression() for _ in range(pop_size)]
    
    # Evolution loop
    for generation in range(generations):
        fitness_vals = [fitness(individual, x_vals, y_vals) for individual in population]
        
        # Select parents, apply crossover and mutation
        next_generation = []
        
        while len(next_generation) < pop_size:
            parents = tournament_selection(population, fitness_vals, tournament_size)
            child1, child2 = crossover(parents[0], parents[1])
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])
        
        population = next_generation[:pop_size]
        
        # Output the best individual of this generation
        best_fitness = min(fitness_vals)
        best_individual = population[fitness_vals.index(best_fitness)]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Expression = {best_individual.expression}")
    
    # Return the best individual
    return best_individual

# Parameters
pop_size = 50
generations = 50
mutation_rate = 0
