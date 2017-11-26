import numpy as np

from Graph import Graph
from Solution import Solution
import copy

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
        # generate random q distributed in [0:1]
        q = np.random.rand() 
        b = self.parameter_beta
        if len(sol.visited) > 0:
            source = sol.visited[-1]

        else : 
            source = 0
        
        if len(sol.not_visited) <= 1:
            return 0
        else:
            if q < self.parameter_q0:
                nv = np.array(sol.not_visited)
                nv = np.delete(nv, np.where(nv == 0))
                t = self.pheromone[source][nv]            # pheromone associated to each edge
                c = self.graph.costs[source][nv] 
                return nv[np.argmax(t/np.power(c, b))]
            else:
                nv = np.array(sol.not_visited)       
                nv = np.delete(nv, np.where(nv==0))
                t = self.pheromone[source][nv]                 # pheromone
                c = np.power(self.graph.costs[source][nv], b)  # cost 
                sm = np.sum(t/c) 
                prob = np.divide(np.divide(t, c), sm)
                return np.random.choice(nv, 1, p=prob)[0]

    # Choose vertex i and an non-consecutive vertex j, invert them and check if
    # the cost of the solution improved. If it did, then that's the new best 
    # solution. Otherwise, we keep the current solution.  
    # Repeat this process until there's no more possible exchanges. 
    def heuristic2opt(self, sol):
        new_sol = Solution(sol)
        for i in range(-1, len(sol.visited)-1):
            for j in range(i+2, len(sol.visited)-1): 
                # print "\nold solution"
                # print new_sol.visited
                new_sol.inverser_ville(i, j)
                # print "new solution" 
                # print new_sol.visited
                new_sol.cost = new_sol.get_cost(0)
                # print new_sol.cost
                if new_sol.cost < sol.cost:
                    sol.visited = new_sol.visited
                    sol.cost = new_sol.cost
        return sol

    def global_update(self, sol):
        L_gb = sol.cost
        best_sol = sol.visited

        
        
        for i in range(0,self.pheromone.shape[0]):
            for j in range(0, self.pheromone.shape[1]):
                self.pheromone[i][j] = (1 -self.parameter_rho)*self.pheromone[i][j]
        for j in range(0, len(best_sol)):
            if (j == 0):
                i = 0
            i = j-1
            self.pheromone[best_sol[i]][best_sol[j]] = self.pheromone[best_sol[i]][best_sol[j]] + self.parameter_rho * 1/ L_gb
    def local_update(self, sol):
        s = sol.visited 
        phi = self.parameter_phi 
        for j in range(0, len(s)):
            self.pheromone[s[j-1]][s[j]] = (1-phi)*self.pheromone[s[j-1]][s[j]] + phi*self.pheromone_init[s[j-1]][s[j]]
            self.pheromone[s[j]][s[j-1]] = (1-phi)*self.pheromone[s[j]][s[j-1]] + phi*self.pheromone_init[s[j]][s[j-1]]
    def runACO(self, maxiteration):

        best_solution_of_iteration = Solution(self.graph)
        solutions = Solution(self.graph)
        crange = len(solutions.not_visited)
        
        for m in range(0,maxiteration):
            for k in range(0, self.parameter_K):
                solutions = Solution(self.graph)
                for c_ in range(0, crange):
                    if (c_ == 0):
                        source = 0
                    else:
                        source = solutions.visited[-1]
                    nc = self.get_next_city(solutions)
                    solutions.add_edge(source, nc)
                self.local_update(solutions)
                if k == 0:
                    best_solution_of_iteration = Solution(solutions)
                else :
                    if solutions.cost < best_solution_of_iteration.cost: #do the heuristic only on the best solution of the iteration
                        best_solution_of_iteration = Solution(solutions)
                        
                        
            best_solution_of_iteration = Solution(self.heuristic2opt(best_solution_of_iteration))       
            if best_solution_of_iteration.cost < self.best.cost:
                self.best = Solution(best_solution_of_iteration)
                self.global_update(self.best)
                print("best cost till now:", self.best.cost)
            
            if m%10 == 0:
                
                print(m*100/maxiteration, "%")

        self.best.printsol()
        print("the cost", self.best.cost)            
            

# if __name__ == '__main__':
#     aco = ACO(0.9, 2, 0.1, 0.1, 10, 'N12.data')
#     aco.runACO(50)
