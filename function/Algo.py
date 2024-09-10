import sys 

from function.VAE import *
import time 
import matplotlib.pyplot as plt
from collections import deque
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

##memory space tracker
# from memory_profiler import profile

import gc


##################################################################################################
############################ Function for visualisation ##############################
import matplotlib
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Sequential", [ '#80ef80', 'green', "limegreen", 'yellow', 'orange'])


#######################################################################################
#plot prior if dim_latent == 2
def plot_prior( z_variationnel, mu, sigma2,  path, z_encoded, z_mcmc, scatter = True):
    #################################################################
    K = mu.shape[0]
    min_x = np.min(mu[:, 0])
    max_x = np.max(mu[:, 0])
    min_y = np.min(mu[:, 1])
    max_y = np.max(mu[:, 1])
    x1 = min_x - 0.3*(max_x - min_x)
    x2 = max_x + 0.3*(max_x - min_x)
    y1 = min_y - 0.3*(max_y - min_y)
    y2 = max_y + 0.3*(max_y - min_y)

    inf_x = np.min([x1, np.min(z_variationnel[:, 0]), np.min(z_encoded[:, 0]), np.min(z_mcmc[:, 0])])
    sup_x = np.max([x2, np.max(z_variationnel[:, 0]), np.max(z_encoded[:, 0]), np.max(z_mcmc[:, 0])])
    inf_y = np.min([y1, np.min(z_variationnel[:, 1]), np.min(z_encoded[:, 1]), np.min(z_mcmc[:, 1])])
    sup_y = np.max([y2, np.max(z_variationnel[:, 1]), np.max(z_encoded[:, 1]), np.max(z_mcmc[:, 1])])
    ############################################################################

    X, Y = np.meshgrid(np.linspace(inf_x,sup_x, 500), np.linspace(inf_y,sup_y, 500))
    pos = np.dstack((X,Y))
    rv =  [sp.multivariate_normal(mu, np.diag(sigma)) for mu, sigma in zip(mu, tf.exp(sigma2))]
    pdf = np.zeros((500, 500))
    for i in range(K):
        pdf = pdf + np.array( rv[i].pdf(pos))
    pdf = pdf/K
        # plt.contour(X, Y,np.array( rv[i].pdf(pos)), levels = [.1,.2,.5])
    plt.pcolormesh(X, Y, pdf, cmap = cmap)
    if scatter :
        plt.scatter(z_variationnel[:, 0], z_variationnel[:, 1], s=6, c = 'red', alpha = .1)
        plt.scatter(z_encoded[:, 0], z_encoded[:, 1], s=6, c = 'blue', alpha = .1)
        plt.scatter(z_mcmc[:, 0], z_mcmc[:, 1], s=6, c = 'purple', alpha = .1)
    
    plt.title('Gaussians of the Vamprior Mixture')
    plt.savefig(path)
    plt.show(block = False)
    plt.pause(3)
    plt.close()


####################################################################################
##################### VAE setting ########################

# @profile
def set_vae(input_dim, latent_dim, pseudo_inputs, samples):
    encoder = Encoder(input_dim, latent_dim, True)
    decoder = Decoder(input_dim, latent_dim, True) 
    prior = VP(input_dim, pseudo_inputs)
    vae = VAE(encoder, decoder, prior, name_prior = 'vamprior')
    

    # # #init latent space
    ae = AutoEncoder(encoder, decoder)
    ae.initialized_ae(samples, 1e-3, 100, 80)

    # #init of pseudo-inputs 
    prior.initialized_ps(samples, 0.001)


    # samples = tf.data.Dataset.from_generator(samples) 
    # #vae training 
    vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
    vae.fit(samples, epochs = 100, batch_size =100, shuffle = True, verbose = 0)
    tf.keras.backend.clear_session()
    return vae


#####################################################################################################################
############################## MCMC  #################################


############### Classic MCMC in the case of a SS algorithm
def MCMC(length_chain, sample_threshold, sample_threshold_centered, PHI, N,d, inf_mixture, rv, phi, sd, mean, gamma, run_count):
    L = np.zeros((length_chain, N, d)) #NON_centered chain MCMC
    U = np.zeros((length_chain, N, d)) #Centered chain MCMC 
    accep_sequance = np.zeros(N)
    #BOOTSTRAP 
    random_seed = np.random.choice(sample_threshold.shape[0], size = N) 
    L[0] = sample_threshold[random_seed, :]
    U[0] = sample_threshold_centered[random_seed, :]
    phi_threshold = PHI[random_seed] # N


    for i in range(length_chain-1):
        centered_candidat = np.array(inf_mixture.getSample(size = N))#N x d 
        candidat = sd  * centered_candidat + mean
        pdf_candidat = rv.logpdf(candidat) #(N,) 
        phi_candidate = phi(candidat) # (N,)
        run_count += N
        pdf_Li = rv.logpdf(L[i]) #(N,)
        
        
        ratio = np.squeeze(pdf_candidat.reshape(-1,1)  + np.array(inf_mixture.computeLogPDF(U[i])) - pdf_Li.reshape(-1,1) - np.array(inf_mixture.computeLogPDF(centered_candidat)))
        # ratio = np.squeeze((pdf_candidat).reshape(-1,1) *np.array( inf_mixture.computePDF(U[i])) / (pdf_Li.reshape(-1,1) * np.array(inf_mixture.computePDF(centered_candidat))) )
        near = np.exp(ratio) 
        ratio = np.exp(ratio) * (phi_candidate > gamma)
        
        u = np.random.uniform(size = N)
        L[i+1] = candidat * np.expand_dims(u < ratio, axis = 1) + L[i] * np.expand_dims(u >= ratio, axis = 1) 
        U[i+1] = centered_candidat * np.expand_dims(u < ratio, axis = 1) + U[i] * np.expand_dims(u >= ratio, axis = 1) 
        accep_sequance = accep_sequance + (u < ratio) * 1 #Acceptance for each Markov chain 
        phi_threshold[(u < ratio)] = phi_candidate[(u < ratio)] #MAJ phi values
    #N_chain of length 10 are completed 
    PHI = phi_threshold #MAJ phi value
    

    #lag by keeping the last MCMC values for each chain 
    sample = L[i+1]
    un, counts = np.unique(sample, return_counts= True, axis= 0 )
    print(un.shape)
    print(f"Nombre de doublons au total {(counts[counts > 1]).sum()}")

    return PHI, sample, accep_sequance, run_count


########## With adaptive chain depth (stopping at each accepted proposal )
# @profile
def MCMC_adaptive_acceptation(length_chain, sample_threshold, sample_threshold_centered, PHI, N,d, inf_mixture, rv, phi, sd, mean, gamma, run_count):
    L = np.zeros((length_chain, N, d)) #Non_centered chain MCMC
    U = np.zeros((length_chain, N, d)) #Centered chain MCMC 
    accep_sequance = np.zeros(N)
    #BOOTSTRAP 
    random_seed = np.random.choice(sample_threshold.shape[0], size = N) 
    L[0] = sample_threshold[random_seed, :]
    U[0] = sample_threshold_centered[random_seed, :]
    phi_threshold = PHI[random_seed] # N

    n = N
    idx = np.arange(N)
    for i in range(length_chain-1):
        centered_candidat = np.array(inf_mixture.getSample(size = n))#N x d 
        candidat = sd  * centered_candidat + mean
        pdf_candidat = rv.logpdf(candidat)
        pdf_Li = rv.logpdf(L[i, idx, :]) #(N,)
        ratio = np.squeeze(pdf_candidat.reshape(-1,1)  + \
                           np.array(inf_mixture.computeLogPDF(U[i, idx, :])) - pdf_Li.reshape(-1,1) - np.array(inf_mixture.computeLogPDF(centered_candidat)))
        log_u = np.log(np.random.uniform(size = n))

        #track for the indices of the acceptation / rejection Markov process at each iteration
        idx_ratio = idx[log_u < ratio]

        #track for the indices of the indices for the i-th iteration
        idx_candidat = np.arange(n)[log_u < ratio]

        phi_candidate = phi(candidat[log_u < ratio]) 

        failure_zone = phi_candidate > gamma
        idx_candidat = idx_candidat[failure_zone]
        idx_failure_zone = idx_ratio[failure_zone] 
        
        print(f"Shape des indices avec proposition {idx_failure_zone.shape}")

        #Number of call to code phi
        run_count += phi_candidate.shape[0] 

        
        #stopping the chain for those accepted 

        idx_not_done = np.delete(idx, [i for i,val in enumerate(idx) if val in idx_failure_zone])
        print(f"Shape des indices sans proposition {idx_not_done.shape}")
        #not centered 
        L[i+1:, idx_failure_zone, :] = np.repeat(np.expand_dims(candidat[idx_candidat], axis=0), (length_chain-i-1), axis=0) #Tensor l-i x accep_phi x d
        L[i+1, idx_not_done, :] = L[i, idx_not_done, :]
        #centered 
        U[i+1:, idx_failure_zone, :] = np.repeat(np.expand_dims(centered_candidat[idx_candidat], axis=0), (length_chain-i-1), axis=0) #Tensor l-i x accep_phi x d
        U[i+1, idx_not_done, :] = U[i, idx_not_done, :]

        #MAJ phi values
        phi_threshold[idx_failure_zone] = phi_candidate[failure_zone] 
        accep_sequance[idx_ratio] = accep_sequance[idx_ratio] +  failure_zone *1 #Acceptance for each Markov chain 
        idx = idx_not_done
        n = idx.shape[0]
        print(f"Indice deja acceptés : {idx_failure_zone}")
        if n == 0:
            break
    print(n, '\n', i)

    #N_chain of length 10 are completed 

    PHI = phi_threshold #MAJ phi value

    print(np.sum(np.sort(np.concatenate([idx_not_done, idx_failure_zone]))))
    

    #lag by keeping the last MCMC values for each chain 
    sample = L[i+1]
    print(sample)
    # print(sample)

    return PHI, sample, accep_sequance, run_count

#########################################################
##################### Evaluation of phi only if the proposal is accepted before 
def MCMC_optimized_evaluation(length_chain, sample_threshold, sample_threshold_centered, PHI, N,d, inf_mixture, rv, phi, sd, mean, gamma, run_count):
    L = np.zeros((length_chain, N, d)) #NON_centered chain MCMC
    U = np.zeros((length_chain, N, d)) #Centered chain MCMC 
    accep_sequance = np.zeros(N)
    #BOOTSTRAP :  threshold or sample ? 
    random_seed = np.random.choice(sample_threshold.shape[0], size = N) 
    L[0] = sample_threshold[random_seed, :]
    U[0] = sample_threshold_centered[random_seed, :]
    phi_threshold = PHI[random_seed] # N

    idx = np.arange(N)
    for i in range(length_chain-1):
        centered_candidat = np.array(inf_mixture.getSample(size = N))#N x d 
        candidat = sd  * centered_candidat + mean
        pdf_candidat = rv.logpdf(candidat)
        pdf_Li = rv.logpdf(L[i]) #(N,)
        ratio = np.squeeze(pdf_candidat.reshape(-1,1)  + \
                           np.array(inf_mixture.computeLogPDF(U[i])) - pdf_Li.reshape(-1,1) - np.array(inf_mixture.computeLogPDF(centered_candidat)))
        
        log_u = np.log(np.random.uniform(size = N))

        #track for the indices of the acceptation / rejection Markov process at each iteration
        idx_ratio = idx[log_u < ratio]
        phi_candidate = phi(candidat[idx_ratio]) 
        failure_zone = phi_candidate > gamma
        idx_failure_zone = idx_ratio[failure_zone] 

        run_count += phi_candidate.shape[0] #call to phi
        
        
        # MAJ samples values
        idx_out_failure_zone = np.delete(idx, [i for i,val in enumerate(idx) if val in idx_failure_zone])
        L[i+1:, idx_failure_zone, :] = candidat[idx_failure_zone]
        L[i+1, idx_out_failure_zone, :] = L[i, idx_out_failure_zone, :]
        #centered 
        U[i+1:, idx_failure_zone, :] = centered_candidat[idx_failure_zone]
        U[i+1, idx_out_failure_zone, :] = U[i, idx_out_failure_zone, :]

        #MAJ phi values
        phi_threshold[idx_failure_zone] = phi_candidate[failure_zone] 
        #Acceptance rate 
        accep_sequance[idx_failure_zone] = accep_sequance[idx_failure_zone] +  1 #Acceptance for each Markov chain to be in the failure zone 

    #MAJ phi value
    PHI = phi_threshold 

    

    #keeping the last MCMC values for each chain 
    sample = L[i+1]

    return PHI, sample, accep_sequance, run_count




# @profile
def ss_vae(sample, threshold, phi, level, latent_dim, K_peuso_inputs, N_prior, length_chain, **kwargs):
    quantile = list()
    sequence = list()
    acceptance = list()
    ratio_traj= list()
    

    N, d = sample.shape
    PHI = phi(sample)
    run_count = N
    quantile.append(np.quantile(PHI, 1-level)) #first threshold 

    rv = sp.multivariate_normal(mean= np.zeros(d)) #fX
    k = 0
    while quantile[k] <= threshold :
        gc.collect()  # Force garbage collection
        idx =  np.where(PHI > quantile[k])[0] 
         # RAISING AN ERROR
        try: 
            idx[0]
        except IndexError:
            print("Exception raised")
            break
        
        
        PHI = PHI[idx]

        print('---------------------------------------')
        print(f"Moyenne avant MCMC : {np.mean(PHI)}")

        sample_threshold = sample[idx, :]
        #reduction + centering 
        sd = np.std(sample_threshold, axis = 0) #d 
        mean = np.mean(sample_threshold, axis = 0)

        sample_threshold_centered = (sample_threshold - mean) / sd
        vae = set_vae(d, latent_dim, K_peuso_inputs, sample_threshold_centered)

        encoder, decoder = vae.get_encoder_decoder()

        z_encoded,_, _ = encoder(sample_threshold_centered)

        ps_mean, ps_logvar, z, isnan = vae.density_x(N_prior)
        if isnan :
            return tuple([None]*7) #stopping the excution (7 because there is 7 values return by this function)

        inf_mixture = vae.distrx #openturns Mixture (Finite mixture returned by the vae)
        gc.collect()
        
        #MCMC algorithm (here the optimized)
        PHI, sample, accep_sequence, run_count= MCMC_optimized_evaluation(length_chain, sample_threshold, sample_threshold_centered, 
                                                            PHI, N, d, inf_mixture, rv, phi, sd, mean, quantile[k], run_count)
        quantile.append(np.quantile(PHI, 1-level))
        acceptance.append(accep_sequence / length_chain)
        
        ##############
        print(f"moyenne de PHI après MCMC {np.mean(PHI)}")
        print(f"Proba superieur à 3.5 : {np.mean(PHI > threshold)}")
        print(f"Les différents taux d'acceptation au sein des chaînes {np.unique(accep_sequence)}")
        
        #Chains where there haven't been any accepted proposal
        D = sample[accep_sequence == 0, :]
        unique, index, counts = np.unique(D, return_counts= True, return_index=True ,axis=0)
        print(unique.shape)
        index_D = index[counts > 1]
        print(f"echantillons qui n'ont pas bouges {(accep_sequence == 0 *1 ).sum()}")
        print(f"echantillons qui se repetent {(counts[counts > 1]).shape} et occurence {counts[counts > 1].sum()}")

        print('--------------------------------------------------------')
        ##############
        z_mcmc, _, _ = encoder(sample)

        if kwargs['plot'] :
            if d < 21 :
                density_vae = sd * np.array(inf_mixture.getSample(5000)) + mean
                
                plt.figure(figsize = (15,7))
                if d < 6:
                    row = 1
                else : 
                    row = d//5
                for i in range(d):
                    plt.subplot(row,5,i+1)
                    plt.hist(density_vae[:,i], bins = 100, density=True)
                plt.savefig('figures/hist_d' + str(d) + '_' + str(k) + '.png')
                plt.show(block = False)
                plt.pause(3)
                plt.close()

            #Sampling in the latent space : coaperation of the Vamprior and variationnel posterior
            path = 'figures/nature_latent_d' + str(d) + '_' + str(k) + '.png'
            plot_prior(z, ps_mean.numpy(), ps_logvar.numpy(), path, z_encoded, z_mcmc, False)
            path = 'figures/latent_d' + str(d) + '_' + str(k) + '.png'
            plot_prior(z, ps_mean.numpy(), ps_logvar.numpy(), path, z_encoded, z_mcmc)

        if kwargs['memory']:
            #samples returned at each intermediate event 
            sequence.append(sample)
            # ratio_traj.append(ratio) # ratio before the indicator function 1_(phi(x))
        
            # if np.shape(index_D) != 0:
            #     heights = [(ratio[index_D] > 1 ).sum(), (ratio[index_D] <= 1 ).sum()]
            #     levels = ['ratio > 1', 'ratio <= 1']

            #     fig, ax = plt.figure()
            #     bar_container = ax.bar(levels, heights, color = ['blue', 'green'])
            #     ax.bar_label(bar_container, fontsize = 14)
            #     ax.set_ylim(0, np.max(heights)+5)
            #     ax.show()
            
        
        print(f"Evenement {k} termine") 
        #next event 
        k +=1


    acceptance = np.array(acceptance)
    mean_accep = np.round(acceptance.mean(axis = 1), 5)
    failure = level**k * np.mean(PHI > threshold)
    print(np.mean(PHI))

    return sequence, quantile, mean_accep, ratio_traj, failure, k, run_count