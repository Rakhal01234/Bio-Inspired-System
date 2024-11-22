import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Grey Wolf Optimization (GWO)
class GreyWolfOptimizer:
    def __init__(self, obj_function, dim, lb, ub, num_agents=10, max_iter=50):
        self.obj_function = obj_function
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_agents, self.dim))
        self.alpha_pos = np.zeros(self.dim)
        self.beta_pos = np.zeros(self.dim)
        self.delta_pos = np.zeros(self.dim)
        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")

    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.num_agents):
                # Calculate fitness for each agent
                fitness = self.obj_function(self.positions[i, :])
                # Update Alpha, Beta, Delta positions
                if fitness < self.alpha_score:
                    self.alpha_score, self.alpha_pos = fitness, self.positions[i, :].copy()
                elif fitness < self.beta_score:
                    self.beta_score, self.beta_pos = fitness, self.positions[i, :].copy()
                elif fitness < self.delta_score:
                    self.delta_score, self.delta_pos = fitness, self.positions[i, :].copy()

            # Update positions
            a = 2 - iter * (2 / self.max_iter)  # Linear reduction of 'a'
            for i in range(self.num_agents):
                for j in range(self.dim):
                    r1, r2 = np.random.random(), np.random.random()
                    A1, C1 = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.random(), np.random.random()
                    A2, C2 = 2 * a * r1 - a, 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.random(), np.random.random()
                    A3, C3 = 2 * a * r1 - a, 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i, j] = np.clip((X1 + X2 + X3) / 3, self.lb[j], self.ub[j])

        return self.alpha_score, self.alpha_pos

# Objective Function (Accuracy Maximization)
def objective_function(params):
    k = int(params[0])
    p = int(params[1])
    if k <= 0:  # Ensure k is valid
        return 1e6
    knn = KNeighborsClassifier(n_neighbors=k, p=p)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    return -accuracy_score(y_val, y_pred)  # Negative accuracy (we minimize)

# Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split Dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Define GWO parameters
dim = 2  # [k (n_neighbors), p (distance metric)]
lb = [1, 1]  # Lower bounds
ub = [20, 2]  # Upper bounds (Manhattan or Euclidean distance)
num_agents = 10
max_iter = 50

# Optimize using GWO
gwo = GreyWolfOptimizer(objective_function, dim, lb, ub, num_agents, max_iter)
best_score, best_params = gwo.optimize()

# Evaluate the best solution
best_k = int(best_params[0])
best_p = int(best_params[1])
knn = KNeighborsClassifier(n_neighbors=best_k, p=best_p)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)

print("Best Parameters Found by GWO:")
print(f"k: {best_k}, p: {best_p}")
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
