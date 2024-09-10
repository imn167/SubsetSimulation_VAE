import numpy as np 
import matplotlib.pyplot as plt 
import openturns as ot 

from IPython.display import clear_output


#Simulation of a truncated normal distribution 
def truncatedDistribution(n_samples,  mean, variance,  bound):
    sd = np.sqrt(variance)
    L = list()
    i = 0 
    while i < n_samples:
        prop = np.random.normal(loc = mean, scale=sd)

        if np.abs(prop[0]) > bound and np.abs(prop[1]) > 2: 
            L.append(prop)
            i += 1
            if i %10 == 0:
                clear_output(wait=True)
                print("boucle %d terminée" %(i))

    return np.array(L)
#dimension
d = 50
dist = truncatedDistribution(10000,  np.zeros(d), np.ones(d), 2)
dist.shape

np.save('2_component_truncated50.npy', dist)