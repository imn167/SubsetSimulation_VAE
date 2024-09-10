import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as sp 

############################ MMA Algorithm ###################################


def MMA(sample, phi, threshold, rv, sd,chain_length,  level):
    N, d = sample.shape 

    quantile = list()
    sequence = list()
    acceptance_rate = list()

    # first fixed quantile 
    PHI = phi(sample)
    quantile.append(np.quantile(PHI, (1-level) )) 
    run_count = N #counting calls to phi 
    k = 0
    while quantile[k] <= threshold :
        idx = np.where(PHI > quantile[k])[0] #N x level 
        #Bootstrap 
        random_seed = np.random.choice(idx, size=N)
        sample_threshold = sample[random_seed, :] #N x d in Fk
        phi_threshold = PHI[random_seed] #N
        
        #MCMC with the modified Metropolis 

        L = np.zeros((chain_length, N, d))
        L[0] = sample_threshold

        chain_acceptance = np.zeros(N)
        for i in range(chain_length-1):
            candidate = np.zeros((N, d))
            for j in range(d):
                candidate_j = L[i, :, j] + np.random.normal(  scale = sd, size=N)
                ratio = rv.pdf(candidate_j) / rv.pdf(L[i, :, j]) #N 
                u = np.random.uniform(size = N)
                candidate[:, j] = candidate_j * (u < ratio) + L[i, :, j] * (u >= ratio)
            
            #now we test if the candidates belongs to Fk
            phi_candidate = phi(candidate) #N
            run_count += N #optimizing the evaluation of phi is hard in this case 
            accepted_candidate = phi_candidate > quantile[k] #N 
            chain_acceptance = chain_acceptance + (1 * (accepted_candidate))
            phi_threshold[accepted_candidate] = phi_candidate[accepted_candidate] #N : MAJ of phi values 
            L[i+1] =  candidate * accepted_candidate.reshape(-1,1)+  L[i] * np.invert(accepted_candidate).reshape(-1,1)
        
        sample = L[i+1]
        #samples for each conditionnal distribution
        sequence.append(sample)
        acceptance_rate.append(chain_acceptance /  chain_length)
        PHI = phi_threshold

        #next quantile 
        quantile.append(np.quantile(PHI, (1-level) ))
        k += 1 
    
    failure = (level)**(k) * np.mean(PHI > threshold)
    print(np.mean(PHI))

    return quantile, sequence, acceptance_rate, failure, np.mean(PHI), run_count




