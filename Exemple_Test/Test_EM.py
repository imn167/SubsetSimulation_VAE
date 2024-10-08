##### Testing the EM algorithm to initialize the parameters of the mixture MoG
import sys 
sys.path.append('..')

from function.VAE import *
from function.EM import *

prior = MoGPrior(2,4)
from sklearn.datasets import make_blobs
Z, y, centers = make_blobs(1000, centers = 4, return_centers=True)
print(y.shape)
plt.scatter(Z[:,0], Z[:,1], s=8, c= y.reshape(-1,1))
plt.show()
w_t, mu_t, sigma2_t, max_iter = EM(Z, prior, 500, 1e-3)

print(w_t, '\n', mu_t, '\n', sigma2_t)
print(centers, '\n', )

print(max_iter)

mixture_plot(Z, w_t, mu_t, sigma2_t, min(Z[:,0]), max(Z[:,0]), min(Z[:,1]), max(Z[:,1]))