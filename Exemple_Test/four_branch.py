import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as sp 


#definition de la fonction du problème 4-branches 
def four_branch(X, beta = 0):
   
    N, d = X.shape 
    quant1 =  np.expand_dims(np.sum(X, axis=1) / np.sqrt(d), axis=1) +beta
    quant2 = - np.expand_dims(np.sum(X, axis=1) / np.sqrt(d), axis= 1 ) +beta
    quant3 =  np.expand_dims((np.sum(X[:, : int(d/2)], axis= 1) - np.sum(X[:, int(d/2) :], axis= 1)) / np.sqrt(d), axis=1) +beta
    quant4 =  np.expand_dims((-np.sum(X[:, : int(d/2)], axis= 1) + np.sum(X[:, int(d/2) :], axis= 1)) / np.sqrt(d), axis=1) +beta

    tensor = np.concatenate([quant1, quant2, quant3, quant4], axis=1)
    minimum = np.min(tensor, axis = 1)
        

    return - minimum
