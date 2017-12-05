import numpy as np

from Graph import Graph
from Solution import Solution
import copy
from time import time 
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
    #on choisit la prochaine villes de notre soluiton en fonction de la pheromone avec une chance d'avoir une villes choisi au hasard.
    def get_next_city(self, sol):
        #On specifie une variable aleatoire q entre 0 et 1
        q = np.random.rand() 
        b = self.parameter_beta

        #dans le cas ou on a visiter des villes, on prend la derniere comme etant la source pour notre prochaine aretes
        if len(sol.visited) > 0:
            source = sol.visited[-1]
        #sinon, on place la source a 0
        else : 
            source = 0
        #si on a visiter toutes les villes, on donne comme notre prochaine ville la destination de 0
        if len(sol.not_visited) <= 1:
            return 0
        #si il nous reste des villes
        else:
            #nous avons une chance qu'on choississe une ville en fonction l'enoncer 1 du TP
            if q < self.parameter_q0:
                nv = np.array(sol.not_visited)
                nv = np.delete(nv, np.where(nv == 0))
                t = self.pheromone[source][nv]            # pheromone associated to each edge
                c = self.graph.costs[source][nv] 
                return nv[np.argmax(t/np.power(c, b))]
            #nous avons aussi une chance qu'on choississe une ville aleatoirement en fonction de la pheromone, l'enoncer 2 du TP
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
        minimum_local = 0
        iterateur = 0
        while (minimum_local == 0 ):
            iterateur+=1
            minimum_local = 1
            for i in range(-1, len(sol.visited)-2):
                for j in range(i+2, len(sol.visited)-1):
                    #i_p1 = sol.visited[i+1]
                    cost_avant = cost_apres = 0 
                    cost_avant = self.graph.costs[sol.visited[i],sol.visited[i+1]] + self.graph.costs[sol.visited[j], sol.visited[j+1]]
                    
                    cost_apres = self.graph.costs[sol.visited[i],sol.visited[j]] + self.graph.costs[sol.visited[i+1], sol.visited[j+1]]
                    delta = cost_apres - cost_avant


                    if delta < 0:
                        minimum_local = 0
                        sol.inverser_ville(i, j)
                        sol.cost = sol.cost -cost_avant+ cost_apres  #delta should be negatif 

        #print( iterateur)
        return sol


    def global_update(self, sol):
        self.best = Solution(sol)
        s = sol.visited 
        self.pheromone = (1-self.parameter_rho)*self.pheromone
        for j in range(0, len(s)): 
            self.pheromone[s[j-1]][s[j]] += self.parameter_rho / self.best.cost
            self.pheromone[s[j]][s[j-1]] += self.parameter_rho / self.best.cost
        
    def local_update(self, sol):
        s = sol.visited 
        phi = self.parameter_phi 
        for j in range(0, len(s)):
            self.pheromone[s[j-1]][s[j]] = (1-phi)*self.pheromone[s[j-1]][s[j]] + phi*self.pheromone_init[s[j-1]][s[j]]
            self.pheromone[s[j]][s[j-1]] = (1-phi)*self.pheromone[s[j]][s[j-1]] + phi*self.pheromone_init[s[j]][s[j-1]]
    def runACO(self, maxiteration):
        optimal = 1290319
        
        best_solution_of_iteration = Solution(self.graph)
        solutions = Solution(self.graph)
        crange = len(solutions.not_visited)
        id_since_improvement = 0
        id_best = 0
        t1 = time()
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
                self.global_update(best_solution_of_iteration)
                id_best = m
            td = time()-t1
            if m%100 == 0:
                id_since_improvement = m - id_best
                gap_best = 100*(self.best.cost - optimal)/optimal
                print ("iteration, :%i , gap_best, %r, id_without_improvement, %i , CPU , %r" %(m, gap_best,id_since_improvement, td))
                #print(m*100/maxiteration, "%")

        #self.best.printsol()
        td = time()-t1
        
        print("parameters q0 %r,beta %r, rho %r, phi %r, k %r, time %f , cost %r " % (self.parameter_q0 , self.parameter_beta , self.parameter_rho,   self.parameter_phi , self.parameter_K , td,  self.best.cost ))
        #print("the cost", self.best.cost) 
           
            

# if __name__ == '__main__':
#     aco = ACO(0.9, 2, 0.1, 0.1, 10, 'N12.data')
#     aco.runACO(50)
