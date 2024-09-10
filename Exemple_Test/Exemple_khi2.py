import sys 
sys.path.append("..")
import numpy as np 
import matplotlib.pyplot as plt 
from function.Vanillla_SS import subset_simulation

from scipy import stats

#####
def phi(X):
    return (np.power(X,2)).sum(axis = 1) #somme au carré des composantes du vecteur 

#threshold 
t = 10

X = np.random.normal(size= (1000, 2))
acceptance = np.zeros(3)
print(phi(X).shape)

sequence, n_event, failure, quantile, acceptance,  rate = subset_simulation(X, 10, phi, 0.3,level= .2)
print('-------------------------------------------------------------')
print(f'Estimation par SS {failure}')
print("La probabilité théorique (approx numérique de la cdf) est %f" %(1-stats.chi2.cdf(10, 2)))
print('-------------------------------------------------------------')


# print(acceptance)


fig, ax = plt.subplots()
last_simu = sequence[-1]
print(last_simu.shape)
second_simu = sequence[0]
fig, ax = plt.subplots()
ax.scatter(second_simu[:, 0], second_simu[:,  1], c = "green", s= 10)
ax.scatter(last_simu[:, 0], last_simu[:, 1], c = "red", s= 10)
plt.show()
