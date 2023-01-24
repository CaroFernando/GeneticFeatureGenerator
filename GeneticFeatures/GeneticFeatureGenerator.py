import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Node import *
from scipy.stats import rankdata

class GeneticFeatureGenerator:

    def __init__(self, operations, operation_names = None, popsize = 100, maxiter = 200, clone_prob = 0.3, mutation_rate = 0.05, max_tree_depth = 7):
        self.operations = operations
        self.operation_names = operation_names
        self.popsize = popsize
        self.maxiter = maxiter
        self.clone_prob = clone_prob
        self.mutation_rate = mutation_rate
        self.max_tree_depth = max_tree_depth
        self.y_rank = None

    def corr(self, X, Y):
        assert len(X) == len(Y), "X and Y must have the same length in order to calculate correlation"
        n = len(X)
        if self.y_rank is None:
            self.y_rank = rankdata(Y, method = 'ordinal')
        y_rank = self.y_rank[X.argsort()].copy()
        sum_y_rank = np.abs(np.diff(y_rank)).sum()
        return 1 - (3 * sum_y_rank) / (n*n - 1)

    def objective_function(self, t, data, target):
        res = t(data)
        if isinstance(res, float):
            return 0
        
        c = self.corr(res, target)
        if c < 0 or np.isnan(c):
            return 0
        return c

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

    def init_generation(self, no_cols):
        return [Tree(self.operations, no_cols, self.operation_names, max_init_depth=self.max_tree_depth) for _ in range(self.popsize)]

    def optimize(self, data, target, verbose = False):
        generation = self.init_generation(data.shape[1])
        fitness = [self.objective_function(t, data, target) for t in generation]
        fitness = np.array(fitness)

        best = generation[np.argmax(fitness)]
        best_fitness = np.max(fitness)

        for it in range(self.maxiter):
            for i in generation:
                # selecto two parents
                p1 = np.random.choice(generation, p = fitness / np.sum(fitness))
                p2 = np.random.choice(generation, p = fitness / np.sum(fitness))

                child = self.crossover(p1, p2)
                
                if np.random.rand() < self.mutation_rate:
                    self.mutate(child)

                child_fitness = self.objective_function(child, data, target)

                # choose a random individual index 
                reverse_fitness = 1 - fitness
                sel = np.random.choice(range(self.popsize), p = reverse_fitness / np.sum(reverse_fitness))

                generation[sel].erase()
                generation[sel] = child
                fitness[sel] = child_fitness

                if child_fitness > best_fitness:
                    best = child.copy()
                    best_fitness = child_fitness

            if verbose:
                print("Iteration: ", it, "Best fitness: ", best_fitness, "Generation best", np.max(fitness), end = '\r')
        if verbose:
            print()
        return best

class MultiFeatureGenerator:
    def __init__(self, data, target, feature_generator, no_features, no_splits, max_split_size, verbose = False):
        self.data = data
        self.target = target
        self.no_features = no_features
        self.no_splits = no_splits
        self.max_split_size = max_split_size
        self.splitsize = int(np.floor(self.data.shape[0] / self.no_splits))
        self.indexes = np.arange(len(self.data), dtype=np.int32)
        # set numpy random seed
        np.random.seed(np.random.randint(0, 1000000))
        np.random.shuffle(self.indexes)
        self.current_split = 0
        self.feature_generator = feature_generator
        self.verbose = verbose
        self.cont = 0

    def get_next_split(self):
        if self.cont >= self.no_features:
            raise StopIteration
        else:
            fixed_size = int(min(self.splitsize, self.max_split_size))
            current_indexes = self.indexes[self.current_split * self.splitsize: self.current_split * self.splitsize + fixed_size]
            
            print("Split: ", self.current_split, "Feature: ", self.cont)
            self.current_split += 1
            self.current_split %= self.no_splits
            self.cont += 1
            return self.feature_generator.optimize(self.data[current_indexes], self.target[current_indexes], self.verbose)

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next_split()

    def __len__(self):
        return self.no_splits
    