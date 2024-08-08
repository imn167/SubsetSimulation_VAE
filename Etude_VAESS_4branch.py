import numpy as np 
import matplotlib.pyplot as plt 


t = 5
pt = 1.15e-6

def load_values(path):
    tables = np.load(path)
    estimation = tables['arr_0']
    cost = tables['arr_1']
    return estimation, cost 

def get_estimation(est1, est_2, cost1, cost2):
    estimation = np.concatenate([est1, est_2])
    cost = np.concatenate([cost1, cost2])
    return estimation, cost, estimation.shape


path1 = 'Resultats/Estimation_d100/batch1.npz'    #batch2 dernier avec les continue si nan dans les variances
path2 = 'Resultats/Estimation_d100/batch2.npz'

estimation_1, cost_1 = load_values(path1)
estimation_2, cost_2 = load_values(path2)

estimation, cost, n = get_estimation(estimation_1, estimation_2, cost_1, cost_2)

Pf_ss = estimation.mean()
Ntot = np.floor(cost.mean())
cov = estimation.std()/estimation.mean()

print(f'cov = {cov}')

Nreq = np.floor((1-pt) / (pt * cov**2))

print(f'Pf_SS = {Pf_ss} | nu = {Nreq / Ntot} | avec {n} estimations')

plt.boxplot(estimation)
plt.show()

plt.hist(estimation, density=True)
# plt.vlines(1.15e-6, 0, 7e-6)
plt.show()

print(f'{np.unique(estimation, return_counts=True)}')




