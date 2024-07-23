from Exemple_Test.four_branch import four_branch
from function.MMA import MMA
import numpy as np
import scipy.stats as sp 
import matplotlib.pyplot as plt 
from statsmodels.graphics import tsaplots

# dim = [2,10,20,50,70,100]
estimation = list()
CMC = list()
rv = sp.multivariate_normal()
for elt in range(100):
    # np.random.seed(123)
    samples = np.random.normal(size = (10000,20))
    N, d = samples.shape


    print('------------------------------------------')
    print(f"Taille de l'echantillon {samples.shape}")
    print(f"MC estimation {np.mean(four_branch(samples) > 3.5)}")
    quantile, chain, acceptance, failure, mean = MMA(samples, four_branch, 3.5, rv, .4, .25)
    estimation.append(failure)
    CMC.append(np.mean(four_branch(samples) > 3.5))
    print(f"Estimation SS de défaillance {np.round(failure, 5)} et en moyenne {np.round(mean, 5)} et quantile {np.round(quantile, 5)}")
    acceptance = np.array(acceptance)
    print(f"Taux d'acceptation en moyenne {np.round(acceptance.mean(axis=1), 5)}")

    # last = chain[-1]

    # a = tsaplots.plot_acf(last[:, 1])
    # plt.show(block = False)
    # plt.pause(2)
    # plt.close()

    # if d ==2 : 
    #     row = 2
    #     plt.figure()
    #     plt.scatter(chain[-1][:, 0], chain[-1][:, 1], s= 6)
    #     plt.show(block = False)
    #     plt.pause(2)
    #     plt.close()
    # else :
    #     row = elt // 5
    # plt.figure(figsize= (15,10))
    # for i in range(elt):
    #     plt.subplot(row, 5, i+1)
    #     plt.hist(last[:, i], bins = 100)
    # plt.show(block = False)
    # plt.pause(2)
    # plt.close()

print(f"Estimation par CMC {np.array(CMC).mean()}")
print(f"Ecart-type pour CMC {np.array(CMC).std()}")
print(f"Estimation par MMA est de {(np.array(estimation).mean())}")
print(f"Ecart-type de l'estimateur {np.array(estimation).std()}")