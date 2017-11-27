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
    aco = ACO(0.5, 2, 0, 0, 0, 'test')
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
    t1 = time()
    aco = ACO(q0, beta, rho, phi, K, './tsp/qatar')
    aco.runACO(250)
    t2 = time()
    td = t2-t1
    print('Time difference: , parameters ', td, q0,beta, rho, phi, K )
if __name__ == '__main__':
    q0 = 0.9
    beta = 2
    rho = 0.1
    phi = 0.1
    K = 10
    
    #test_heuristic2opt()
    #test_global_update()
    #test_local_update()
    #test_next_city()
    beta_list = np.array([1,2,3,4,5,6,7,8,9])
    q0_list = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95])
    processes = []
    
    for q0 in q0_list:
        processes = [mp.Process(target=runACO_test, args=(q0, beta, rho, phi, K)) for x in range(5)]
        
        leng = len(processes)
        print('lenght', leng)
        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        
            
        
    #print("prochaine iteration")
    #aco.runACO(1000)
    #print("prochaine iteration")
    #aco.runACO(1000)
    #print("prochaine iteration")
    
    #aco.runACO(1000)
    #print("prochaine iteration")
    
    #aco.runACO(1000)
    #print("prochaine iteration")