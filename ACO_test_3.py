from ACO import ACO
from Solution import Solution
import copy
from time import time
import numpy as np
import multiprocessing  as mp 
def test_heuristic2opt():
    print("testing heuristic 2opt...")
    aco = ACO(0, 0, 0, 0, 0, 'test')
    s = Solution(aco.graph)
    s.add_edge(0, 2)
    s.add_edge(2, 3)
    s.add_edge(3, 1)
    s.add_edge(1, 0)
    aco.heuristic2opt(s)
    assert s.cost == 6
    print('ok')


def test_local_update():
    print("testing local update...")
    phi = 0.5
    aco = ACO(0, 0, 0, phi, 0, 'test')
    s = Solution(aco.graph)
    aco.pheromone[:, :] = 1
    c = copy.copy(aco.pheromone)
    s.add_edge(0, 2)
    #aco.local_update(s)
    s.add_edge(2, 3)
    #aco.local_update(s)
    s.add_edge(3, 1)
    #aco.local_update(s)
    s.add_edge(1, 0)
    aco.local_update(s)
    print(s.visited)
    print("c is",c)
    print("pheromone is",aco.pheromone)
    #assert (c != aco.pheromone).any()
    #assert (aco.pheromone.sum() == 8)
    assert abs(aco.pheromone[0, 1] - 0.833) < 1e-3
    assert abs(aco.pheromone[0, 2] - 0.833) < 1e-3
    assert abs(aco.pheromone[3, 1] - 0.833) < 1e-3
    assert abs(aco.pheromone[3, 2] - 0.833) < 1e-3
    print('ok')


def test_global_update():
    print('testing global update...')
    rho = 0.1
    aco = ACO(0, 0, rho, 0, 0, 'test')
    s = Solution(aco.graph)
    s.add_edge(0, 2)
    s.add_edge(2, 3)
    s.add_edge(3, 1)
    s.add_edge(1, 0)
    aco.pheromone[:, :] = 1
    aco.global_update(s)
    #aco.global_update(s)
    #print(aco.pheromone)
    assert abs(aco.pheromone[0, 3] - 0.9) < 1e-3
    assert abs(aco.pheromone[1, 2] - 0.9) < 1e-3
    assert abs(aco.pheromone[0, 2] - 0.91) < 1e-3
    assert abs(aco.pheromone[2, 3] - 0.91) < 1e-3
    assert abs(aco.pheromone[3, 1] - 0.91) < 1e-3
    assert abs(aco.pheromone[1, 0] - 0.91) < 1e-3
    print('ok')


def test_next_city():
    
    print('testing get next city...')
    aco = ACO(0.5, 2, 0, 0, 0, 'testa')
    s = Solution(aco.graph)
    #s.add_edge(0,3)
    c1 = 0
    c3 = 0
    c2 = 0
    for i in range(1000):
        c = aco.get_next_city(s)
        if c == 3:
            c3 += 1
        elif c == 1:
            c1 += 1
        elif c == 2:
            c2 += 1
                
    
    assert (abs(c3 - 870) < 30)
    assert (abs(c1 - 90) < 30)
    print('ok')
    

def runACO_test(q0,beta, rho, phi, K):
    
    aco = ACO(q0, beta, rho, phi, K, './tsp/canada')
    aco.runACO(1000)
    
    

if __name__ == '__main__':
    q0 = 0.8 # 0.75
    beta = 4 #2.4
    rho = 0.85 # 0.15
    phi = 1 #0.12
    K = 15
     
    
    #test_heuristic2opt()
    #test_global_update()
    #test_local_update()
    #test_next_city()
    
    beta_list = np.array([0.01,1.2,1.6,1.8,2,2.2,2.8,3,4,5,10])
    q0_list = np.array([0,0.1,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1]) #,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95])
    rho_list= np.array([0.01,0.05, 0.1 ,0.12, 0.15, 0.2, 0.5,0.8, 0.9, 1])
    phi_list =np.array([0.01,0.05, 0.1 ,0.12, 0.15, 0.2, 0.5,0.8, 0.9, 1])
    K_list = np.array([1,5,10,15,25,50,75,100,200])
    
    for n in range(0,1):
        np.random.seed(n)
        #runACO_test(0.7,1.9,0.9,1, 12)
        #runACO_test(q0,beta,rho,phi, K)
        processes = []
        for n in range(0,5):
            np.random.seed(n)
            process = [mp.Process(target=runACO_test, args=(q0, beta, rho, phi, K))]
            processes.extend(process)      
        
            # Run processes
        i = 1
        for p in processes:
                np.random.seed(i) # to make sure we have different results 
                p.start()
                i+=1
    
            # Exit the completed processes
        for p in processes:
                p.join()        
        
        
        """
        processes = []
        for beta in beta_list:
            process = [mp.Process(target=runACO_test, args=(q0, beta, rho, phi, K))]
            processes.extend(process)
        leng = len(processes)
        print('beta')
            # Run processes
        for p in processes:
                p.start()
    
            # Exit the completed processes
        for p in processes:
                p.join()
        
        beta = 2 
        
        np.random.seed(n)
        processes = []
        for q0 in q0_list:
            process = [mp.Process(target=runACO_test, args=(q0, beta, rho, phi, K))]
            processes.extend(process)
        leng = len(processes)
        print('q0')
            # Run processes
        for p in processes:
                p.start()
    
            # Exit the completed processes
        for p in processes:
                p.join()        
        
        q0 = 0.9
        processes = []
        for rho in rho_list:
            process = [mp.Process(target=runACO_test, args=(q0, beta, rho, phi, K))]
            processes.extend(process)
        leng = len(processes)
        print('rho')
            # Run processes
        for p in processes:
                p.start()
    
            # Exit the completed processes
        for p in processes:
                p.join()           
        
        rho = 0.1
        processes = []
        for phi in phi_list:
            process = [mp.Process(target=runACO_test, args=(q0, beta, rho, phi, K)) for x in range(1)]
            processes.extend(process)
        leng = len(processes)
        print('phi')
            # Run processes
        for p in processes:
                p.start()
    
            # Exit the completed processes
        for p in processes:
                p.join()  
                
        phi = 0.1       
        processes = []
        for K in K_list:
            process = [mp.Process(target=runACO_test, args=(q0, beta, rho, phi, K)) for x in range(1)]
            processes.extend(process)
        leng = len(processes)
        print('k')
            # Run processes
        for p in processes:
                p.start()
    
            # Exit the completed processes
        for p in processes:
            p.join()             
        K = 10
        """
    #print("prochaine iteration")
    #aco.runACO(1000)
    #print("prochaine iteration")
    #aco.runACO(1000)
    #print("prochaine iteration")
    
    #aco.runACO(1000)
    #print("prochaine iteration")
    
    #aco.runACO(1000)
    #print("prochaine iteration")