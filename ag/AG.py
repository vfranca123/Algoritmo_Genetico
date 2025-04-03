import numpy as np
import random

def read_tsp(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    nodes = []
    reading_nodes = False
    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            reading_nodes = True
            continue
        if line.startswith("EOF"):
            break
        if reading_nodes:
            parts = line.split()
            nodes.append((int(parts[1]), int(parts[2])))
    return np.array(nodes)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def compute_distance_matrix(nodes):
    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = euclidean_distance(nodes[i], nodes[j])
    return dist_matrix

def fitness(solution, dist_matrix):
    return sum(dist_matrix[solution[i], solution[i + 1]] for i in range(len(solution) - 1)) + dist_matrix[solution[-1], solution[0]]

def tournament_selection(population, scores, k=3):
    selected = random.sample(list(zip(population, scores)), k)
    return min(selected, key=lambda x: x[1])[0]

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    remaining = [item for item in parent2 if item not in child]
    index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[index]
            index += 1
    return child

def swap_mutation(solution):
    a, b = random.sample(range(len(solution)), 2)
    solution[a], solution[b] = solution[b], solution[a]
    return solution

def genetic_algorithm(filename, generations=50, population_size=50, pc=0.8, pm=0.05):
    nodes = read_tsp(filename)
    dist_matrix = compute_distance_matrix(nodes)
    
    population = [random.sample(range(len(nodes)), len(nodes)) for _ in range(population_size)]
    
    for generation in range(generations):
        scores = [fitness(ind, dist_matrix) for ind in population]
        new_population = []
        
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            
            if random.random() < pc:
                child1, child2 = ordered_crossover(parent1, parent2), ordered_crossover(parent2, parent1)
            else:
                child1, child2 = parent1[:], parent2[:]
            
            if random.random() < pm:
                child1 = swap_mutation(child1)
            if random.random() < pm:
                child2 = swap_mutation(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population
    
    best_solution = min(population, key=lambda ind: fitness(ind, dist_matrix))
    best_fitness = fitness(best_solution, dist_matrix)
    return best_solution, best_fitness

if __name__ == "__main__":
    best_path, best_distance = genetic_algorithm("data/data2.tsp")
    print("Melhor Fitness:", best_distance)
   