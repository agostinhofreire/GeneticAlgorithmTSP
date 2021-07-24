import random
import numpy as np
from itertools import permutations
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from CitiesGraph import CitiesGraph


class GeneticAlgorithm:

    def __init__(self, population_size=10, mutation_chance=0.05, max_generations=1000, max_no_improvement=25, number_parents=2,selection_mode_id=0, cities_controller=CitiesGraph(), fit_mode="create", start="A"):

        # Cities config

        self.cities_controller = cities_controller

        if fit_mode == "create":
            self.cities_graph = self.cities_controller.create()
            self.start_city = self.cities_controller.getStart()
        else:
            self.cities_graph = self.cities_controller.load()
            self.start_city = start.upper()

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

        self.to_remove = []
        self.history = {
            "Generation": [],
            "Fitness": []
        }

        # Fitting config

        selection_modes = {
            0: "tournament",
            1: "elitism"
        }

        self.parent_selection_mode = selection_modes[selection_mode_id]

        self.max_no_improvement = max_no_improvement
        self.max_generations = max_generations

        self.no_improvement_generations = 0
        self.generation_counter = 1
        self.individual_id = self.population_size - 1

    def init_population(self):
        print("Creating first population")
        print("Start city:", self.start_city)
        print("Cities Graph\n")
        print(self.cities_graph, "\n")

        self.population = list(enumerate(random.sample(self.cities_permutations, self.population_size)))

    def run(self):

        while self.generation_counter < self.max_generations and self.no_improvement_generations < self.max_no_improvement:
            print("\n[INFO] {}° Generation Results\n".format(self.generation_counter))
            self.evaluate()
            self.get_log()
            self.update_best_solution()
            parents = self.get_parents()
            children = self.crossover(parents, self.number_parents)
            children = self.mutation(children)
            self.population += children

            self.history["Generation"].append(self.generation_counter)
            self.history["Fitness"].append(self.population_eval[0][2])

            sns.set_style("whitegrid")
            plt.title("Genetic Algorithm - TSP")
            sns.lineplot(data=self.history, y="Fitness", x="Generation")
            plt.plot()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()

        print("\n\n Stopping fitting")
        print("\tBest solution [IND-{}] {} - Travel Distance {}".format(self.best_solution[0],
                                                                                  self.best_solution[1],
                                                                                  self.best_solution[2]))


        sns.set_style("whitegrid")
        plt.title("Genetic Algorithm - TSP")
        sns.lineplot(data=self.history, y="Fitness", x="Generation")
        plt.plot()
        plt.show()



    def get_parents(self):
        parents = []

        if self.parent_selection_mode == "tournament":
            while len(parents) < self.number_parents:
                t = random.sample(self.population_eval, int(self.population_size*0.4))
                p = min(t, key=lambda x: x[2])
                if p not in parents:
                    parents.append(p)

        elif self.parent_selection_mode == "elitism":
            parents = self.population_eval[:self.number_parents]

        print(f"\tParents selected by {self.parent_selection_mode.upper()}:")
        temp_to_remove = []
        for index, pa in enumerate(parents):
            p_id = index + 1
            print(f"\tP{p_id} [Solution: [IND-{pa[0]}] {pa[1]}] [Travel Distance: {pa[2]}]")
            temp_to_remove.append(pa[0])

        self.to_remove = temp_to_remove
        return parents

    def mutation(self, children):

        print("\n\tStarting Mutation")

        new_children = []

        for child_id, child in children:
            new_child = child.copy()

            if random.random() <= self.mutation_chance:

                number_of_mutations = random.randint(1, int(0.6*len(self.available_cities)))
                available_index = range(len(self.available_cities))

                for i in range(number_of_mutations):

                    a, b = random.sample(available_index, 2)

                    new_child[a] = child[b]
                    new_child[b] = child[a]

                if self.valid_solution(new_child):
                    print(f"\tMutating [IND-{child_id}] {child}] -> Result [IND-{child_id}] {new_child}]")
                else:
                    print(f"\tMutating [IND-{child_id}] {child}] -> Result [IND-{child_id}] {new_child}] [INVALID]")
                    new_child =child.copy()


            new_children.append((child_id, new_child))


        return new_children

    def crossover(self, parents, number_children):

        # Implementaion inspired by
        # A novel multi-parent order crossover in genetic algorithm for combinatorial optimization problems
        # https://doi.org/10.1016/j.cie.2019.05.012


        if number_children > len(parents):
            print("Number of children must be less than", len(parents))
            sys.exit()

        slice_size = int(0.5*len(self.available_cities))
        start_index = random.choice(range(len(self.available_cities) - slice_size - 1))
        end_index = start_index + slice_size

        children = []

        print("\n\tStarting Crossover")

        child_id = 0

        while len(children) < number_children:

            temp_child = list("-" * len(self.available_cities))
            temp_child[start_index:end_index] = parents[child_id][1][start_index:end_index]

            order_cycle = self.parents_order_cycle(child_id)

            child = self.create_child(temp_child, parents, order_cycle, end_index)

            if self.valid_solution(child):
                self.individual_id += 1
                print(f"\tSlice {temp_child} -> Created Child [IND-{self.individual_id}] {child}")
                children.append((self.individual_id, child))
                child_id += 1
            else:
                print(child)

        return children

    def create_child(self, child, parents, order_cycle, end_index):

        temp_child = child.copy()

        last_id = order_cycle[-1]
        current_gene = end_index
        current_child_gene = end_index

        for parent_id in order_cycle:
            while True:

                if "-" not in temp_child:
                    return temp_child

                if current_gene >= len(self.available_cities):
                    current_gene = 0

                if current_child_gene >= len(self.available_cities):
                    current_child_gene = 0

                current_parent = parents[parent_id][1]

                gene = current_parent[current_gene]


                if gene not in temp_child:
                    temp_child[current_child_gene] = gene
                    current_child_gene += 1

                else:
                    if parent_id != last_id:
                        break

                current_gene += 1

        return temp_child


    def evaluate(self):

        eval_results = []

        new_population = []

        print("\n\tStarting individuals selection")

        for ind_id, individual in self.population:
            ind_eval = self.individual_eval(individual)

            if ind_id in self.to_remove:
                print(f"\t\tRemoving individual [IND-{ind_id}] {individual} - Travel Distance {ind_eval}")
            else:
                eval_results.append((ind_id, individual, ind_eval))
                new_population.append((ind_id, individual))


        self.population = new_population

        self.population_eval = sorted(eval_results, key=lambda x: x[2])

        print()


    def update_best_solution(self):

        if self.best_solution == None:

            self.best_solution = self.population_eval[0]
            print("\tFound a better solution [IND-{}] {} - Travel Distance {}".format(self.best_solution[0],
                                                                         self.best_solution[1],
                                                                         self.best_solution[2]))

        else:
            if self.population_eval[0][2] < self.best_solution[2]:

                self.best_solution = self.population_eval[0]
                self.no_improvement_generations = 0
                print("\tFound a better solution [IND-{}] {} - Travel Distance {}".format(self.best_solution[0],
                                                                             self.best_solution[1],
                                                                             self.best_solution[2]))

            else:
                self.no_improvement_generations += 1
                print(f"\tNo improvement was found - {self.no_improvement_generations} generations")
                print("\tBest solution in the generation [IND-{}] {} - Travel Distance {}".format(self.population_eval[0][0],
                                                                                     self.population_eval[0][1],
                                                                                     self.population_eval[0][2]))
                print("\tCurrent best solution found [IND-{}] {} - Travel Distance {}".format(self.best_solution[0],
                                                                             self.best_solution[1],
                                                                             self.best_solution[2]))

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


        for index, (ind_id, ind, ev) in enumerate(self.population_eval):

            if index < 9:
                spaces = " "
            else:
                spaces = ""

            line = f"\t[{index+1}°{spaces}] [Solution: [IND-{ind_id}] {ind}] [Travel Distance: {ev}]"
            print(line)

        print()





