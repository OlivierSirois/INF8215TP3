import numpy as np

from Graph import Graph
from Solution import Solution

SOURCE = 0


class ACO(object):
    def __init__(self, q0, beta, rho, phi, K, data):
        self.parameter_q0 = q0
        self.parameter_beta = beta
        self.parameter_rho = rho
        self.parameter_phi = phi
        self.parameter_K = K

        self.graph = Graph(data)
        self.best = Solution(self.graph)
        self.best.cost = 99999999999999
        self.pheromone_init = np.ones((self.graph.N, self.graph.N))
        f = open(data + '_init', 'r')
        self.pheromone_init *= float(f.readline())
        self.pheromone = np.ones((self.graph.N, self.graph.N))

    def get_next_city(self, sol):
        q = np.random.rand()
        beta = self.parameter_beta
        if (len(sol.visited) == 0):
            source = 0
        else:
            source = sol.visited(len(sol.visited))
        if (len(sol.not_visited) == 1):
            nextcity = 0
        else:
            if q < self.parameter_q0:
                not_visited = np.array(sol.not_visited)  
                not_visited = np.delete(not_visited, np.where(not_visited==0))
                nextcity = np.argmax(self.pheromone[source][not_visited] / np.power(self.graph.costs[source][not_visited], beta))
                return not_visited[nextcity]
            elif q >= self.parameter_q0:
                not_visited = np.array(sol.not_visited)
                prob = np.zeros(len(not_visited))
                not_visited = np.delete(not_visited, np.where(not_visited==0))
                termsum = np.sum(self.pheromone[source][not_visited] / np.power(self.graph.costs[source][not_visited], beta))
                for j in not_visited:
                    prob[j] = (self.pheromone[source][j] / np.power(self.graph.costs[source][j], beta))/termsum
                return np.random.choice(not_visited, 1, p = prob[not_visited])

    def heuristic2opt(self, sol):
        raise NotImplementedError()

    def global_update(self, sol):
        raise NotImplementedError()

    def local_update(self, sol):
        raise NotImplementedError()

    def runACO(self, maxiteration):
        raise NotImplementedError()

# if __name__ == '__main__':
#     aco = ACO(0.9, 2, 0.1, 0.1, 10, 'N12.data')
#     aco.runACO(50)
