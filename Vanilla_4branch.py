from Exemple_Test.four_branch import four_branch
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt 
from statsmodels.graphics import tsaplots
from function.Vanillla_SS import subset_simulation


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
t = 3.5
Pf = 9.3e-4
sd = .4
estimation = list()
sample = np.random.normal(size = (100, 2))
print(np.mean(four_branch(sample, 5) > t)
)
# for elt in range(100):
#     # np.random.seed(123)
#     sample = np.random.normal(size= (10000, 2))
    
#     chain, k, failure, quantile, accep_rate = subset_simulation(sample, t, four_branch, sd, .25)
#     estimation.append(failure)
#     print(f'Pf_SS = {np.round(failure, 5)} pour la dimension {elt} et une perturbation {sd}')
#     print(f'quantile : {np.round(quantile, 3)}')
#     acceptance = np.array(accep_rate)
#     print(f"taux d'acceptation : {acceptance.mean(axis=1)}")
#     last = chain[-1]
#     print(f"Autocorr à un lag 1 : {autocorr(last[:,0], 1)} et {autocorr(last[:,1], 1)}")
#     # a = tsaplots.plot_acf(last[:, 0])
#     # a = tsaplots.plot_acf(last[:, 1])
#     # plt.show(block = False)
#     # plt.pause(2)
#     # plt.close()
#     # plt.plot(last[ :, 0], label = r'$X_1$')
#     # plt.legend()
#     # plt.show(block = False)
#     # plt.pause(2)
#     # plt.close()
#     # plt.plot(last[:,  1], label = r'$X_2$')
#     # plt.legend()
#     # plt.show(block = False)
#     # plt.pause(2)
#     # plt.close()

# print(f"Estimation par Vanilla SS est de {(np.array(estimation).mean())}")
# print(f"Ecart-type de l'estimateur {np.array(estimation).std()}")

