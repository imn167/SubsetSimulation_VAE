import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 
import sys 
import scipy.stats as sp 


############# SUBSET SIMULATION ALGORITHME ########################

def subset_simulation(sample, threshold,phi, sd,  level = .1):
    N, dim = np.shape(sample)
    #list of the intermediate threshold 
    quantile = list()
    sequence = list()
    accep_rate = list()
    #first threshold 
    PHI = phi(sample)
    quantile.append(np.quantile(PHI, 1-level))
    k = 0
    rv = stats.multivariate_normal(mean=np.zeros(dim)) #computy the probability density of a normal vector of dimension dim

    while quantile[k] < threshold:
        idx = np.where(PHI > quantile[k])[0] #index that statisfie.s the condition 
        # RAISING AN ERROR "vae approximation could be bad"
        try: 
            idx[0]
        except IndexError:
            print("Exception raised")
            break

        #BOOTSTRAP ON THE N*LEVEL SELECTED SAMPLES 
        random_seed = np.random.choice(a=idx, size =N)
        sample_threshold = sample[random_seed, :] #N*level samples that lie in Fk 
        phi_threshold = PHI[random_seed]
        ## simulation according to mcmc 
        chain = np.zeros((N, dim)) # mcmc chain 

        length_chain = 6
        #tensor to parallelize the process 
        L = np.zeros((length_chain, N, dim)) 
        accep_sequance = np.zeros(N)
        L[0] = sample_threshold
        #phi_accepted = phi_threshold
        for j in range(int(length_chain-1)):
            candidate = L[j] + np.random.normal(scale= sd, size=(N, dim)) 
            #stock for phi(candidate) in order to calculat it only once 
            phi_candidate = phi(candidate)
            ratio =  rv.pdf(candidate) / rv.pdf(L[j]) *(phi_candidate> quantile[k]) # Transposé pour candidate si no vae 
            #MAJ
            u = np.random.uniform(size= N)
            L[j+1] =  L[j] * np.expand_dims(u >= ratio, axis = 1) + candidate * np.expand_dims(u< ratio, axis=1)
            phi_threshold[(u< ratio)] = phi_candidate[(u< ratio)]
            accep_sequance += 1 * (u<ratio)
        #phi_value management 
        #phi_threshold = phi_accepted
        PHI = phi_threshold #new phi_values 
        
        #we only keep the last mcmc simulation
        chain = L[j+1]   
        ##### chain for the step k completed 
        sequence.append(chain) 
        quantile.append(np.quantile(PHI, 1-level )) #searching the next intermediate 
        accep_rate.append(accep_sequance / length_chain)
        sample = chain
        k += 1 
        
    failure = (level)**(k) * np.mean(PHI > threshold)
    return sequence, k, failure, quantile, np.array(accep_rate)
