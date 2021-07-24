import random
import numpy as np
from itertools import permutations
import sys

from CitiesGraph import CitiesGraph


class GeneticAlgorithm:

    def __init__(self, population_size=10, mutation_chance=0.05, number_parents=2, cities_controller=CitiesGraph()):

        # Cities config

        self.cities_controller = cities_controller
        self.cities_graph = self.cities_controller.create()
        self.start_city = self.cities_controller.getStart()
        self.cities = self.cities_controller.cities
        self.available_cities = self.cities_controller.cities.copy()

        self.available_cities.remove(self.start_city)

        self.cities_permutations = list(permutations(self.available_cities))

        # Population config

        self.population_size = population_size
        self.mutation_chance = mutation_chance

        if self.population_size > len(self.cities_permutations):
            print("The population size must be less than", len(self.cities_permutations))
            sys.exit()

        self.population = []

        self.population_eval = []
        self.number_parents = number_parents
        self.best_solution = None
        self.history = []

        # Fitting config

        selection_modes = {
            0: "tournament"
        }

        self.parent_selection_mode = selection_modes[0]

        self.max_no_improvement = 15
        self.max_generations = 1000

        self.no_improvement_generations = 0
        self.generation_counter = 1

    def init_population(self):
        print("Creating first population")
        print("Start city:", self.start_city)
        print("Cities Graph\n")
        print(self.cities_graph, "\n")

        self.population = random.sample(self.cities_permutations, self.population_size)

    def run(self):

        while self.generation_counter < self.max_generations and self.no_improvement_generations < self.max_no_improvement:
            self.evaluate()
            self.get_log()
            self.update_best_solution()
            parents = self.get_parents()
            self.crossover(parents, self.number_parents)
            # crossover
            # apply mutation

            break

    def get_parents(self):
        parents = []

        if self.parent_selection_mode == "tournament":
            while len(parents) < self.number_parents:
                t = random.sample(self.population_eval, int(self.population_size*0.6))
                p = min(t, key=lambda x: x[1])
                if p not in parents:
                    parents.append(p)

        print(f"\tParents selected by {self.parent_selection_mode.upper()}:")
        for index, pa in enumerate(parents):
            p_id = index + 1
            print(f"\tP{p_id} [Solution: {pa[0]}] [Travel Distance: {pa[1]}]")

        return parents

    #Implementaion of Multi-Parent Order Crossover

    def crossover(self, parents, number_children):

        if number_children > len(parents):
            print("Number of children must be less than", len(parents))
            sys.exit()

        slice_size = int(0.4*len(self.available_cities))
        start_index = random.choice(range(len(self.available_cities) - slice_size - 1))
        end_index = start_index + slice_size

        children = []

        print("\t Starting Crossover")

        for child_id in range(number_children):
            temp_child = list("-" * len(self.available_cities))
            temp_child[start_index:end_index] = parents[child_id][0][start_index:end_index]

            order_cycle = self.parents_order_cycle(child_id)

            child = self.create_child(temp_child, parents, order_cycle, end_index)
            print("Slices", temp_child)
            print("Created Child", child)
            if self.valid_solution(child):
                children.append(child)

    def create_child(self, child, parents, order_cycle, end_index):

        temp_child = child.copy()

        last_id = order_cycle[-1]
        current_gene = end_index

        for parent_id in order_cycle:
            while True:

                if "-" not in temp_child:
                    break

                if current_gene >= len(self.available_cities):
                    current_gene = 0

                current_parent = parents[parent_id][0]

                gene = current_parent[current_gene]

                if gene not in temp_child:

                    temp_child[current_gene] = gene

                else:
                    if parent_id != last_id:
                        break

                current_gene += 1



        return temp_child





    def evaluate(self):

        eval_results = []

        for individual in self.population:
            ind_eval = self.individual_eval(individual)
            eval_results.append((individual, ind_eval))


        self.population_eval = sorted(eval_results, key=lambda x: x[1])

    def update_best_solution(self):

        if self.best_solution == None:

            self.best_solution = self.population_eval[0]
            print("\tFound a better solution {} - Travel Distance {}".format(self.best_solution[0],
                                                                         self.best_solution[1]))

        else:
            if self.population_eval[0][1] < self.best_solution[1]:

                self.best_solution = self.population_eval[0]
                self.no_improvement_generations = 0
                print("\tFound a better solution {} - Travel Distance {}".format(self.best_solution[0],
                                                                             self.best_solution[1]))

            else:
                self.no_improvement_generations += 1
                print("\tNo improvement was found in the solution.")
                print("\tBest solution in the generation {} - Travel Distance {}".format(self.population_eval[0][0],
                                                                                     self.population_eval[0][1]))
                print("\tBest solution found {} - Travel Distance {}".format(self.best_solution[0],
                                                                         self.best_solution[1]))

        print()
        self.generation_counter += 1

    def individual_eval(self, individual):

        travel = [self.start_city] + list(individual) + [self.start_city]
        travel_value = 0
        prev_city = None

        for city in travel:

            if prev_city != None:

                dist = self.cities_graph.loc[prev_city, city]
                travel_value += dist

            prev_city = city

        return travel_value

    def valid_solution(self, solution):

        temp_solution = list(solution)

        for ava_city in self.available_cities:
            if temp_solution.count(ava_city) != 1:
                return False

        return True

    def parents_order_cycle(self, index):

        order = list(range(self.number_parents))
        order = order[index:] + order[:index]
        order = order[1:]

        return order

    def get_log(self):

        print("[INFO] {}° Generation Results\n".format(self.generation_counter))

        for index, (ind, ev) in enumerate(self.population_eval):
            line = f"\t[{index}°] [Solution: {ind}] [Travel Distance: {ev}]"
            print(line)

        print()





