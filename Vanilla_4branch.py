from Exemple_Test.four_branch import four_branch
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt 
from statsmodels.graphics import tsaplots
from function.Vanillla_SS import subset_simulation
from collections import deque


def autocorr(x, h):
    n = x.shape[0]
    x1 = x[(h+1):]
    x2 = x[1: (n-h)]
    mu = x.mean()
    sigma2 = x.var()
    gamma_h = ((n-h)/n)*((x1-mu) * (x2-mu)).mean()
    rho_h = gamma_h / sigma2
    return rho_h 

## Vanilla Subset Simulation with a  normal proposal kernel with mean the precedent points and variance  
# d = [2, 10, 20,50,70, 100]
t = 5
Pf = 1.5e-6
sd = .1
d= 100
length_chain = 6
estimation = deque()
acceptance = deque()
cost = deque()
# sample = np.random.normal(size = (1000, 2))
# print(np.mean(four_branch(sample, 5) > t)
# )
for elt in range(100):
    
    sample = np.random.normal(size= (10000, d))
    
    chain, k, failure, quantile, accep_rate, Ntot = subset_simulation(sample, t, four_branch, sd, length_chain, .25, False)
    estimation.append(failure)
    print('-------------------------------------------------------------------------')
    print(f'Pf_SS = {np.round(failure, 8)} pour la dimension {d} et une perturbation {sd}')
    print(f'quantile : {np.round(quantile, 3)}')
    accept = np.array(accep_rate)
    accep_all_chain = accept.mean(axis=1)
    acceptance.append(accep_all_chain)
    cost.append(Ntot)
    print(f"taux d'acceptation : {accep_all_chain}")
    last = chain[-1]
    print(f"Autocorr à un lag 1 : {autocorr(last[:,0], 1)} et {autocorr(last[:,1], 1)}")
    print(f'Estimation {elt+1} finie')
    print('--------------------------------------------------------------')
    # a = tsaplots.plot_acf(last[:, 0])
    # a = tsaplots.plot_acf(last[:, 1])
    # plt.show(block = False)
    # plt.pause(2)
    # plt.close()
    # plt.plot(last[ :, 0], label = r'$X_1$')
    # plt.legend()
    # plt.show(block = False)
    # plt.pause(2)
    # plt.close()
    # plt.plot(last[:,  1], label = r'$X_2$')
    # plt.legend()
    # plt.show(block = False)
    # plt.pause(2)
    # plt.close()

# acceptance = np.array(acceptance)
cost = np.array(cost)
estimation = np.array(estimation)
cov = estimation.std() / estimation.mean()
Ntot = int(cost.mean())
Nreq = int((1-Pf)/ (Pf*cov**2)) +1


file = 'vanilla_SS/SS_d' + str(d) +'.txt'

with open(file, 'a') as f :

    print('---------------------------------------------------------------', file= f)
    print('---------------------------------------------------------------', file = f)


    print(f'SS avec noyau de proposition gaussien | sd = {sd}', file = f)
    print('---------------------------------------------------------------', file=f)
    print(f'dimension d = {d} | Pf_SS = {np.round(estimation.mean(), 8)} | cov(Pf_SS) = {np.round(cov, 8)} | nu = {Nreq /Ntot} | l = {length_chain}', file=f)

    print('---------------------------------------------------------------', file=f)
    print(f'taux acceptation moyen : {accep_all_chain}', file=f)
    print('---------------------------------------------------------------', file=f)

