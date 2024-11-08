import numpy as np

class CuckooSearch:
    def __init__(self, num_nests=25, num_iterations=100, pa=0.25, dim=2):
        self.num_nests = num_nests  # Number of nests
        self.num_iterations = num_iterations  # Number of iterations
        self.pa = pa  # Probability of abandoning a nest
        self.dim = dim  # Dimension of the problem
        self.nests = np.random.rand(self.num_nests, self.dim) * 10 - 5  # Initialize nests randomly
        self.fitness = np.array([self.objective_function(nest) for nest in self.nests])  # Evaluate fitness

    def objective_function(self, x):
        return np.sum(x ** 2)  # Sphere function (minimization)

    def get_best_nest(self):
        best_index = np.argmin(self.fitness)
        return self.nests[best_index], self.fitness[best_index]

    def random_nest(self):
        return np.random.rand(self.dim) * 10 - 5  # Random nest within bounds

    def step(self):
        new_nests = np.copy(self.nests)

        # Generate new solutions
        for i in range(self.num_nests):
            # Lévy flight
            step_size = np.random.normal(0, 1, self.dim) * (np.random.rand() ** (1/3))
            new_nest = new_nests[i] + step_size * (new_nests[np.random.randint(self.num_nests)] - new_nests[i])
            new_nests[i] = np.clip(new_nest, -5, 5)  # Keep within bounds

            # Evaluate new solution
            new_fitness = self.objective_function(new_nests[i])
            if new_fitness < self.fitness[i]:
                self.fitness[i] = new_fitness
                self.nests[i] = new_nests[i]

        # Abandon some nests
        for i in range(self.num_nests):
            if np.random.rand() < self.pa:
                new_nests[i] = self.random_nest()
                new_fitness = self.objective_function(new_nests[i])
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    self.nests[i] = new_nests[i]

    def optimize(self):
        for iteration in range(self.num_iterations):
            self.step()
            best_nest, best_fitness = self.get_best_nest()
            print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}, Best Nest = {best_nest}")

if __name__ == "__main__":
    cuckoo_search = CuckooSearch(num_nests=25, num_iterations=100, pa=0.25, dim=2)
    cuckoo_search.optimize()
