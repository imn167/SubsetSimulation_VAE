import sys 
sys.path.append("..")
import numpy as np 
import matplotlib.pyplot as plt 
from function.Vanillla_SS import subset_simulation

from scipy import stats

#Construction de la fonction Phi
def phi(X):
    return (np.power(X,2)).sum(axis = 1) #somme au carré des composantes du vecteur 

t = 10



X = np.random.normal(size= (1000, 2))
acceptance = np.zeros(3)
print(phi(X).shape)

sequence, n_event, failure, quantile, rate = subset_simulation(X, 10, phi, 0.3, .2)
print(failure)
print("La probabilité théorique (approx numérique de la cdf) est %f" %(1-stats.chi2.cdf(10, 2)))
acceptance = acceptance + rate.mean(axis = 1)
print(acceptance)
# X= np.meshgrid(np.linspace(-7,7, 100), np.linspace(-7,7, 100))

fig, ax = plt.subplots()
# pc = ax.pcolormesh(X[0], X[1], phi(X))
# fig.colorbar(pc)



# Z = np.random.normal(size=(10000, 2))
# prob_100 = np.mean(phi(Z) > 10)
# print("probabilité %f avec un echantillon de 10000 " %prob_100 )


# print(n_event,
# quantile)


last_simu = sequence[-1]
print(last_simu.shape)
second_simu = sequence[0]
fig, ax = plt.subplots()
# pc = ax.pcolormesh(X[0],X[1], phi(X))
ax.scatter(second_simu[:, 0], second_simu[:,  1], c = "green", s= 10)
ax.scatter(last_simu[:, 0], last_simu[:, 1], c = "red", s= 10)
# cs = ax.contour(X[0],X[1], phi(X), [3,7,10])
# fig.colorbar(pc)
# ax.clabel(cs, cs.levels, inline=True, fontsize=15)
# fig.savefig('figures_ss/Khi2_VanillaSS.png')
plt.show()

# print("La probabilité théorique  est %f" %(1-stats.chi2.cdf(10, 2)))
# print("La probabilité subset_sample  est %f" %failure)