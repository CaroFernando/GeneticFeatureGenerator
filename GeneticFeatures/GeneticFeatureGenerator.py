import numpy as np
# from same folder import all from Node
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Node import *

from minepy import MINE

class GeneticFeatureGenerator:
    def __init__(self, data, target, operations, operation_names = None, popsize = 100, maxiter = 200, clone_prob = 0.3, mutation_rate = 0.05):
        self.data = data
        self.target = target
        self.operations = operations
        self.operation_names = operation_names
        self.popsize = popsize
        self.maxiter = maxiter
        self.clone_prob = clone_prob
        self.mutation_rate = mutation_rate
        self.mine = MINE(alpha=0.6, c=15)

    def objective_function(self, t):
        res = t(self.data)

        # if res is instance of float
        if isinstance(res, float):
            return 0

        self.mine.compute_score(self.target, res)
        return self.mine.mic()

    def crossover(self, t1, t2):
        random_prob = np.random.rand()
        child = t1.copy() if random_prob < 0.5 else t2.copy()
        if random_prob < 0.5:
            child.random_paste_node(t2.get_random_node())
        else:
            child.random_paste_node(t1.get_random_node())

        return child

    def mutate(self, t):
        t.random_paste_node(t.create_random_node())

    def init_generation(self):
        return [Tree(self.operations, self.data.shape[1], self.operation_names) for _ in range(self.popsize)]

    def optimize(self):
        generation = self.init_generation()
        print("Generation initialized")
        fitness = [self.objective_function(t) for t in generation]
        fitness = np.array(fitness)

        best = generation[np.argmax(fitness)]
        best_fitness = np.max(fitness)

        for i in range(self.maxiter):
            print("Generation: ", i)
            print("Best fitness: ", np.max(fitness))
            print("Worst fitness: ", np.min(fitness))
            print("Mean fitness: ", np.mean(fitness))
            print("Median fitness: ", np.median(fitness))
            print("Std fitness: ", np.std(fitness))
            print("")

            for i in generation:
                # selecto two parents
                p1 = np.random.choice(generation, p = fitness / np.sum(fitness))
                p2 = np.random.choice(generation, p = fitness / np.sum(fitness))

                child = self.crossover(p1, p2)
                
                if np.random.rand() < self.mutation_rate:
                    self.mutate(child)

                child_fitness = self.objective_function(child)

                # choose a random individual index 
                reverse_fitness = 1 - fitness
                sel = np.random.choice(range(self.popsize), p = reverse_fitness / np.sum(reverse_fitness))

                generation[sel].erase()
                generation[sel] = child
                fitness[sel] = child_fitness

                if child_fitness > best_fitness:
                    best = child.copy()
                    best_fitness = child_fitness

        return best
    