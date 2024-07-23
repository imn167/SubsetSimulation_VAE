import sys 

sys.path.append('Users/ibouafia/Desktop/Stage/VAE/Code/function')

from function.VAE import *
import time 
import matplotlib.pyplot as plt
from collections import deque
from memory_profiler import profile
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
### SS function with a independant VAE proposal kernel #####

import gc
import matplotlib
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Sequential", [ '#80ef80', 'green', "limegreen", 'yellow', 'orange'])

def plot_prior( z_variationnel, mu, sigma2,  path):
    K = mu.shape[0]
    #setting plot latent space if dim_latent == 2
    min_x = np.min(mu[:, 0])
    max_x = np.max(mu[:, 0])
    min_y = np.min(mu[:, 1])
    max_y = np.max(mu[:, 1])
    x1 = min_x - 0.3*(max_x - min_x)
    x2 = max_x + 0.3*(max_x - min_x)
    y1 = min_y - 0.3*(max_y - min_y)
    y2 = max_y + 0.3*(max_y - min_y)

    inf_x = np.min([x1, np.min(z_variationnel[:, 0])])
    sup_x = np.max([x2, np.max(z_variationnel[:, 0])])
    inf_y = np.min([y1, np.min(z_variationnel[:, 1])])
    sup_y = np.max([y2, np.max(z_variationnel[:, 1])])

    X, Y = np.meshgrid(np.linspace(inf_x,sup_x, 500), np.linspace(inf_y,sup_y, 500))
    pos = np.dstack((X,Y))
    rv =  [sp.multivariate_normal(mu, np.diag(sigma)) for mu, sigma in zip(mu, tf.exp(sigma2))]
    pdf = np.zeros((500, 500))
    for i in range(K):
        pdf = pdf + np.array( rv[i].pdf(pos))
    pdf = pdf/K
        # plt.contour(X, Y,np.array( rv[i].pdf(pos)), levels = [.1,.2,.5])
    plt.pcolormesh(X, Y, pdf, cmap = cmap)
    plt.scatter(z_variationnel[:, 0], z_variationnel[:, 1], s=6, c = 'red', alpha = .3)
    
    plt.title('Gaussians of the Vamprior Mixture')
    plt.savefig(path)
    plt.show(block = False)
    plt.pause(3)
    plt.close()


@profile 
def ss_vae(sample, threshold, phi, level, latent_dim, K_peuso_inputs, N_prior, **kwargs):
    quantile = deque()
    sequence = deque()
    acceptance = deque()
    ratio_traj= deque()
    

    N, d = sample.shape
    PHI = phi(sample)
    run_count = N
    quantile.append(np.quantile(PHI, 1-level)) #first threshold 

    rv = sp.multivariate_normal(mean= np.zeros(d)) #fX
    k = 0
    while quantile[k] <= threshold :
        gc.collect()  # Force garbage collection
        #keeping only the samples that are abose the threshold 
        idx =  np.where(PHI > quantile[k])[0] 
         # RAISING AN ERROR "vae approximation could be bad"
        try: 
            idx[0]
        except IndexError:
            print("Exception raised")
            break
        

        print(idx.shape)
        PHI = PHI[idx]
        sample_threshold = sample[idx, :]
        #reduction + centering 
        sd = np.std(sample_threshold, axis = 0) #d 
        mean = np.mean(sample_threshold, axis = 0)

        sample_threshold_centered = (sample_threshold - mean) / sd

        #Vae training on centered samples 
        encoder = Encoder(d, latent_dim, True)
        decoder = Decoder(d, latent_dim, True) 
        prior = VP(d, K_peuso_inputs)
        vae = VAE(encoder, decoder, prior, name_prior = 'vamprior')
        

        # # #init latent space
        ae = AutoEncoder(encoder, decoder)
        ae.initialized_ae(sample_threshold_centered, 1e-3, 100, 80)

        # #init of pseudo-inputs 
        prior.initialized_ps(sample_threshold_centered, 0.001)

        # #vae training 
        vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
        vae.fit(sample_threshold_centered, epochs = 100, batch_size =100, shuffle = True, verbose = 1)
        print(k)

        #Bootstrap in order to have N init seed for MCMC  
        #tensor to parallelize the process 
        length_chain = 5
        L = np.zeros((length_chain, N, d)) #NON_centered chain MCMC
        U = np.zeros((length_chain, N, d)) #Centered chain MCMC 

        accep_sequance = np.zeros(N)
        #BOOTSTRAP :  threshold or sample ? 
        random_seed = np.random.choice(sample_threshold.shape[0], size = N) 
        L[0] = sample_threshold[random_seed, :]
        U[0] = sample_threshold_centered[random_seed, :]
        phi_threshold = PHI[random_seed] # N

        ## Setting for VAE sampler 
        ps_mean, ps_logvar, z = vae.density_x(N_prior)
        inf_mixture = vae.distrx #openturns Mixture 

        #### Plot of the learned distribution by the VAE 
        # if kwargs['plot'] :
        #     density_vae = sd * np.array(inf_mixture.getSample(5000)) + mean
            
        #     plt.figure(figsize = (15,7))
        #     if d < 6:
        #         row = 1
        #     else : 
        #         row = d//5
        #     for i in range(d):
        #         plt.subplot(row,5,i+1)
        #         plt.hist(density_vae[:,i], bins = 100, density=True)
        #     plt.savefig('figures/hist_d' + str(d) + '_' + str(k) + '.png')
        #     plt.show(block = False)
        #     plt.pause(3)
        #     plt.close()

        #     #Sampling in the latent space : coaperation of the Vamprior and variationnel posterior
        #     path = 'figures/latent_d' + str(d) + '_' + str(k) + '.png'
        #     plot_prior(z, ps_mean.numpy(), ps_logvar.numpy(), path )
        
        #####
        for i in range(length_chain-1):
            centered_candidat = np.array(inf_mixture.getSample(size = N))#N x d 
            candidat = sd  * centered_candidat + mean
            pdf_candidat = rv.logpdf(candidat) #(N,) 
            phi_candidate = phi(candidat) # (N,)
            run_count += N
            pdf_Li = rv.logpdf(L[i]) #(N,)
            
            
            ratio = np.squeeze(pdf_candidat.reshape(-1,1)  + np.array(inf_mixture.computeLogPDF(U[i])) - pdf_Li.reshape(-1,1) - np.array(inf_mixture.computeLogPDF(centered_candidat)))
            # ratio = np.squeeze((pdf_candidat).reshape(-1,1) *np.array( inf_mixture.computePDF(U[i])) / (pdf_Li.reshape(-1,1) * np.array(inf_mixture.computePDF(centered_candidat))) )
            ratio = np.exp(ratio) * (phi_candidate > quantile[k])
            
            
            u = np.random.uniform(size = N)
            L[i+1] = candidat * np.expand_dims(u < ratio, axis = 1) + L[i] * np.expand_dims(u >= ratio, axis = 1) 
            U[i+1] = centered_candidat * np.expand_dims(u < ratio, axis = 1) + U[i] * np.expand_dims(u >= ratio, axis = 1) 
            accep_sequance = accep_sequance + (u < ratio) * 1 #Acceptance for each Markov chain 
            print((u < ratio).shape)
            phi_threshold[(u < ratio)] = phi_candidate[(u < ratio)] #MAJ phi values
        #N_chain of length 10 are completed 
        PHI = phi_threshold #MAJ phi values
        print(np.mean(PHI > threshold))
        #lag by keeping the last MCMC values for each chain 
        sample = L[i+1]
        del centered_candidat
        del phi_threshold
        del ratio
        del U
        del L
        #memory 
        quantile.append(np.quantile(PHI, 1-level))
        acceptance.append(accep_sequance / length_chain)
        if kwargs['memory']:
            sequence.append(sample)
            ratio_traj.append(ratio)
        gc.collect()  # Force garbage collection
        k += 1 
    
    #estimation 
    acceptance = np.array(acceptance)
    mean_accep = np.round(acceptance.mean(axis = 1), 5)
    plt.hist(PHI, density=True, bins = 'auto')
    failure = level**k * np.mean(PHI > threshold)
    

    return sequence, quantile, mean_accep, ratio_traj, failure, k, run_count




