from itertools import permutations
import matplotlib.pyplot as plt

from CitiesGraph import CitiesGraph
from GeneticAlgorithm import GeneticAlgorithm

# Travelling Salesman Problem - Genetic Algorithms
# Author Agostinho Junior

cities_controller = CitiesGraph(size=10)

genetic_algorithm = GeneticAlgorithm(
    population_size=10,
    mutation_chance=0.05,
    number_parents=4,
    cities_controller=cities_controller
)

genetic_algorithm.init_population()

genetic_algorithm.run()

