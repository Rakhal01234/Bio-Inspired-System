import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the objective function for feature selection
def fitness(solution, X, y):
    if np.sum(solution) == 0:  # Prevent selecting no features
        return 0
    selected_features = X[:, solution == 1]
    X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return accuracy_score(y_test, predictions)

# Lévy flight function
def levy_flight(Lambda):
    sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v) ** (1 / Lambda)
    return step[0]

# Cuckoo Search Algorithm
def cuckoo_search(X, y, n=20, max_gen=100, pa=0.25):
    n_features = X.shape[1]
    nests = np.random.randint(0, 2, (n, n_features))  # Initialize nests (solutions)
    fitness_values = np.array([fitness(nest, X, y) for nest in nests])
    
    for gen in range(max_gen):
        for i in range(n):
            # Perform Lévy flight
            new_nest = nests[i] + levy_flight(1.5) * (np.random.randint(0, 2, n_features) - nests[i])
            new_nest = np.clip(new_nest, 0, 1)  # Clip values to binary
            new_nest = np.random.choice([0, 1], size=n_features, p=[1 - new_nest.mean(), new_nest.mean()])
            new_fitness = fitness(new_nest, X, y)
            
            # Replace with better solution
            if new_fitness > fitness_values[i]:
                nests[i] = new_nest
                fitness_values[i] = new_fitness

        # Abandon some nests (probability pa)
        abandon_indices = np.random.rand(n) < pa
        nests[abandon_indices] = np.random.randint(0, 2, (np.sum(abandon_indices), n_features))
        fitness_values[abandon_indices] = [fitness(nest, X, y) for nest in nests[abandon_indices]]

        # Keep the best solution
        best_idx = np.argmax(fitness_values)
        print(f"Generation {gen + 1}, Best Fitness: {fitness_values[best_idx]}")

    # Return the best solution
    best_idx = np.argmax(fitness_values)
    return nests[best_idx], fitness_values[best_idx]

# Load dataset (example: breast cancer dataset from sklearn)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# Apply Cuckoo Search
best_solution, best_fitness = cuckoo_search(X, y)
print("Best Feature Subset:", best_solution)
print("Best Fitness (Accuracy):", best_fitness)
