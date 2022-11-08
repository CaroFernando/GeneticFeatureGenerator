# generic programming
import numpy as np
import copy

class Node:
    left = None
    right = None
    have_children = False

    def __init__(self):
        pass

    def __call__(self, data):
        return None

    def __str__(self):
        return "Node"

    def copy(self):
        return copy.deepcopy(self)

class OperationNode(Node):
    def __init__(self, op, name = "OperationNode"):
        self.op = op
        self.have_children = True
        self.name = name

    def __call__(self, data):
        return self.op(self.left(data), self.right(data))

    def __str__(self):
        return self.name

class DataNode(Node):
    def __init__(self, ind, name = "DataNode"):
        self.name = name
        self.ind = ind

    def __call__(self, data):
        return data[:, self.ind]

    def __str__(self):
        return self.name

class ScalarNode(Node):
    def __init__(self, scalar):
        self.scalar = scalar

    def __call__(self, data):
        return self.scalar

    def __str__(self):
        return "ScalarNode " + str(self.scalar)

class Tree:
    def create_random_node(self, curr_depth = 0, max_depth = 5):
        # choose random type of node between operation, data and scalar 
        if curr_depth == self.max_init_depth:
            # choose only data or scalar
            if np.random.rand() < 0.5:
                dataind = np.random.randint(0, self.nocols)
                return DataNode(dataind, "DataNode " + str(dataind))
            else:
                return ScalarNode(np.random.uniform(self.scalar_range[0], self.scalar_range[1]))
        else:
            # 80 % chance to create operation node
            p = np.array([0.6, 0.2, 0.2])
            node_type = np.random.choice([0, 1, 2], p=p.ravel())

            if node_type == 0:
                # operation
                opind = np.random.randint(0, len(self.operations))
                node = OperationNode(self.operations[opind], self.operation_names[opind] if self.operation_names is not None else "OperationNode " + str(opind))
                node.left = self.create_random_node(curr_depth + 1, max_depth)
                node.right = self.create_random_node(curr_depth + 1, max_depth)

                return node

            elif node_type == 1:
                # data
                dataind = np.random.randint(0, self.nocols)
                return DataNode(dataind, "DataNode " + str(dataind))
            else:
                # scalar
                return ScalarNode(np.random.uniform(self.scalar_range[0], self.scalar_range[1]))

    def init_random(self):
        self.root = self.create_random_node()

    def __init__(self, operations, nocols, operation_names = None, scalar_range = [0, 1], max_init_depth = 5, random_paste_prob=0.25):
        self.operations = operations
        self.nocols = nocols
        self.operation_names = operation_names
        self.scalar_range = scalar_range
        self.max_init_depth = max_init_depth
        self.random_paste_prob = random_paste_prob

        self.root = None
        self.init_random()

    def __get_random_node(self, curr, p = 0.15):
        if np.random.rand() < p:
            return curr

        if curr.have_children:
            if np.random.rand() < 0.5:
                return self.__get_random_node(curr.left, min(p * 1.01, 0.7))
            else:
                return self.__get_random_node(curr.right, min(p * 1.01, 0.7))

        return curr

    def get_random_node(self):
        return self.__get_random_node(self.root).copy()

    def __random_replace(self, node, curr, p = 0.15):
        if curr is self.root:
            if curr.have_children:
                if np.random.rand() < 0.5:
                    if curr.left.have_children and np.random.rand() > p:
                        self.__random_replace(node, curr.left, min(p * 1.01, 0.7))
                    else:
                        curr.left = node.copy()
                else:
                    if curr.right.have_children and np.random.rand() > p:
                        self.__random_replace(node, curr.right, min(p * 1.01, 0.7))
                    else:
                        curr.right = node.copy()

            else:
                self.root = node.copy()
        
            return 

        if curr.have_children:
            if np.random.rand() < 0.5:
                if curr.left.have_children and np.random.rand() > self.random_paste_prob:
                    self.__random_replace(node, curr.left)
                else:
                    curr.left = node.copy()
            else:
                if curr.right.have_children and np.random.rand() > self.random_paste_prob:
                    self.__random_replace(node, curr.right)
                else:
                    curr.right = node.copy()

        return


    def random_paste_node(self, node):
        self.__random_replace(node, self.root)

    def toString(self, curr, indent = ""):
        if curr is None:
            return ""

        if curr.have_children:
            left = ""
            right = ""

            if curr.left is not None:
                left = self.toString(curr.left, indent + "\t")

            if curr.right is not None:
                right = self.toString(curr.right, indent + "\t")

            #print(str(curr), "(", left, " ", right, ")")

            return indent + str(curr) + "\n" + left + "\n" + right
        
        return indent + str(curr)

    def __str__(self):
        return self.toString(self.root)

    def __call__(self, data):
        return self.root(data)

    def copy(self):
        return copy.deepcopy(self)


    def __erase(self, curr):
        if curr.have_children:
            self.__erase(curr.left)
            self.__erase(curr.right)

        del curr
    def erase(self):
        self.__erase(self.root)
        self.root = None

