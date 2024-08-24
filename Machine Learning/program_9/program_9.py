import numpy as np
import matplotlib.pyplot as plt

# Define the objective function
def objective_function(x):
    return x * np.sin(10 * np.pi * x) + 1

# Generate initial population
def generate_population(size, x_min, x_max):
    return np.random.uniform(x_min, x_max, size)

# Calculate fitness of each individual
def calculate_fitness(population):
    return objective_function(population)

# Select individuals based on fitness (roulette wheel selection)
def selection(population, fitness, num_parents):
    fitness_sum = np.sum(fitness)
    probabilities = fitness / fitness_sum
    parents = np.random.choice(population, size=num_parents, p=probabilities)
    return parents

# Crossover (single point)
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)
    
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring.flatten()

# Mutation
def mutation(offspring, mutation_rate, x_min, x_max):
    for idx in range(offspring.shape[0]):
        if np.random.rand() < mutation_rate:
            random_value = np.random.uniform(-0.1, 0.1)
            offspring[idx] += random_value
            offspring[idx] = np.clip(offspring[idx], x_min, x_max)
    return offspring

# Genetic Algorithm
def genetic_algorithm(objective_function, generations, population_size, x_min, x_max, num_parents, mutation_rate):
    population = generate_population(population_size, x_min, x_max)
    
    for generation in range(generations):
        fitness = calculate_fitness(population)
        parents = selection(population, fitness, num_parents)
        offspring_size = (population_size - num_parents,)
        offspring = crossover(parents.reshape(parents.shape[0], 1), (offspring_size[0], 1))
        offspring = mutation(offspring, mutation_rate, x_min, x_max)
        population[:num_parents] = parents
        population[num_parents:] = offspring
        
        best_fitness = np.max(calculate_fitness(population))
        print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")
    
    best_solution_idx = np.argmax(calculate_fitness(population))
    best_solution = population[best_solution_idx]
    
    return best_solution

# Parameters
generations = 100
population_size = 20
x_min = 0
x_max = 1
num_parents = 10
mutation_rate = 0.1

# Run Genetic Algorithm
best_solution = genetic_algorithm(objective_function, generations, population_size, x_min, x_max, num_parents, mutation_rate)

print(f"Best solution: x = {best_solution:.4f}, f(x) = {objective_function(best_solution):.4f}")

# Plot the objective function
x = np.linspace(x_min, x_max, 1000)
y = objective_function(x)
plt.plot(x, y, label="Objective Function")
plt.plot(best_solution, objective_function(best_solution), 'ro', label="Best Solution")
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Genetic Algorithm Optimization")
plt.show()