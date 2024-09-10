from Exemple_Test.four_branch import four_branch
from function.MMA import MMA
import numpy as np
import scipy.stats as sp 
import matplotlib.pyplot as plt 
from statsmodels.graphics import tsaplots
from collections import deque 
import time 


############################################# Estimation with MMA for the 4-branch exemple ####################################

estimation = deque()
CMC = deque()
cost = deque()

#############
rv = sp.multivariate_normal() #proposal Kernel 
###############
t = 5
pt = 1.5e-6
sd = .4
p0 = .25
d = 100
chain_length = 6



for elt in range(100):
    # np.random.seed(123)
    samples = np.random.normal(size = (10000,d))
    N, d = samples.shape


    print('------------------------------------------')
    print(f"Taille de l'echantillon {samples.shape}")
    print(f"MC estimation {np.mean(four_branch(samples) > 3.5)}")
    print('-----------------------------------------------------')
    start = time.time()
    quantile, chain, acceptance, failure, mean_phi, run = MMA(samples, four_branch, t, rv, sd, chain_length, p0)
    stop = time.time() - start
    
    cost.append((run))
    estimation.append(failure)
    CMC.append(np.mean(four_branch(samples) > t))
    
    print(f"Estimation SS de défaillance {np.round(failure, 5)} et en moyenne {np.round(mean_phi, 5)} et quantile {np.round(quantile, 5)}")
    acceptance = np.array(acceptance)
    rate = np.round(acceptance.mean(axis=1), 5)
    print(f"Taux d'acceptation en moyenne {rate}")


    #### PLOT 
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


plt.plot(chain[-1][:, 0])
plt.show()

estimation = np.array(estimation)
np.save('MMA_results/estimation_d' + str(d), estimation)

cost = np.array(cost)
cov = estimation .std() / estimation.mean()
Ntot = cost.mean()
Nreq = (1-pt) / (pt* cov**2)


#Writting the results in the file MMA_output_beta5 for the case where beta = 5
file = 'MMA_results/MMA_output_beta5.txt'
with open(file, 'a') as f : 
    print('------------------------------------', file=f)
    print(f'temps excution {stop}', file=f)
    print('------------------------------------------', file=f)
    print(f'MMA avec des noyaux de propositions gaussien | sd = {sd} et probabilité fixe {p0} | taille echantillon {10000}', file=f)
    print('-----------------------------------------------------------------------------------', file=f)
    print(f'dimension d = {d} | Pf_MMA = {np.round(estimation.mean(), 8)} | cov(Pf_MMA) = {cov} | nu = {Nreq / Ntot} | l = {chain_length}', file=f)
    print('-----------------------------------------------------------------------------------', file = f)
    print(f"taux d'acceptation moyen pour chaque evenement  {rate}", file = f)
    print('-----------------------------------------------------------------', file=f)