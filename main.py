from itertools import permutations
import matplotlib.pyplot as plt
import random
from CitiesGraph import CitiesGraph
from GeneticAlgorithm import GeneticAlgorithm

# Travelling Salesman Problem - Genetic Algorithms
# Author Agostinho Junior

cities_controller = CitiesGraph(size=10)

genetic_algorithm = GeneticAlgorithm(
    population_size=500,
    mutation_chance=0.05,
    max_generations=10000,
    max_no_improvement=30,
    number_parents=4,
    cities_controller=cities_controller
)

genetic_algorithm.init_population()

genetic_algorithm.run()

