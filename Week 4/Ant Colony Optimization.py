import numpy as np
import random
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, cities, n_ants, n_iterations, decay):
        self.cities = cities
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.pheromone = np.ones((self.n_cities, self.n_cities)) / self.n_cities

    def distance(self, city1, city2):
        return np.linalg.norm(np.array(city1) - np.array(city2))

    def total_distance(self, path):
        return sum(self.distance(self.cities[path[i]], self.cities[path[(i + 1) % self.n_cities]]) for i in range(self.n_cities))

    def get_path(self):
        path = [0]
        visited = set(path)
        for _ in range(1, self.n_cities):
            probabilities = self.calculate_probabilities(path[-1], visited)
            next_city = np.random.choice(range(self.n_cities), p=probabilities)
            path.append(next_city)
            visited.add(next_city)
        return path

    def calculate_probabilities(self, current_city, visited):
        pheromone = self.pheromone[current_city]
        visibility = 1 / (np.array([self.distance(self.cities[current_city], self.cities[i]) for i in range(self.n_cities)]) + 1e-10)
        probabilities = pheromone ** visibility
        probabilities[list(visited)] = 0
        return probabilities / np.sum(probabilities)

    def update_pheromone(self, all_paths):
        self.pheromone *= (1 - self.decay)
        for path in all_paths:
            distance = self.total_distance(path)
            for i in range(len(path)):
                self.pheromone[path[i], path[(i + 1) % self.n_cities]] += 1 / distance

    def optimize(self):
        best_path = None
        best_distance = float('inf')
        
        for iteration in range(self.n_iterations):
            all_paths = [self.get_path() for _ in range(self.n_ants)]
            self.update_pheromone(all_paths)

            for path in all_paths:
                distance = self.total_distance(path)
                if distance < best_distance:
                    best_distance = distance
                    best_path = path
            
            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best distance: {best_distance}")

        return best_path, best_distance

    def plot_path(self, best_path):
        best_path_cities = [self.cities[i] for i in best_path] + [self.cities[best_path[0]]] 
        x, y = zip(*best_path_cities)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker='o')
        plt.title("Best Path Found by Ant Colony Optimization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid()
        plt.scatter(*zip(*self.cities), color='red', zorder=5)
        plt.show()

cities = [(0, 0), (1, 1), (2, 0), (1, 2), (3, 3), (4, 1), (2, 4), (0, 3), (3, 0), (1, 3)]
aco = AntColony(cities, n_ants=20, n_iterations=100, decay=0.01)
best_path, best_distance = aco.optimize()

print("Best path:", best_path)
print("Best distance:", best_distance)

aco.plot_path(best_path)
