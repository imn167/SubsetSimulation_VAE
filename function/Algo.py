import sys 

from function.VAE import *
import time 
import matplotlib.pyplot as plt
from collections import deque
# from memory_profiler import profile
from silence_tensorflow import silence_tensorflow
silence_tensorflow()


import gc
import matplotlib
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Sequential", [ '#80ef80', 'green', "limegreen", 'yellow', 'orange'])


def plot_prior( z_variationnel, mu, sigma2,  path, z_encoded):
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

    inf_x = np.min([x1, np.min(z_variationnel[:, 0]), np.min(z_encoded[:, 0])])
    sup_x = np.max([x2, np.max(z_variationnel[:, 0]), np.max(z_encoded[:, 0])])
    inf_y = np.min([y1, np.min(z_variationnel[:, 1]), np.min(z_encoded[:, 1])])
    sup_y = np.max([y2, np.max(z_variationnel[:, 1]), np.max(z_encoded[:, 1])])

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
    plt.scatter(z_encoded[:, 0], z_encoded[:, 1], s=6, c = 'blue', alpha = .3)
    
    plt.title('Gaussians of the Vamprior Mixture')
    plt.savefig(path)
    plt.show(block = False)
    plt.pause(3)
    plt.close()

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

# @profile
def MCMC(length_chain, sample_threshold, sample_threshold_centered, PHI, N,d, inf_mixture, rv, phi, sd, mean, gamma, run_count):
    L = np.zeros((length_chain, N, d)) #NON_centered chain MCMC
    U = np.zeros((length_chain, N, d)) #Centered chain MCMC 
    accep_sequance = np.zeros(N)
    #BOOTSTRAP :  threshold or sample ? 
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
        #keeping only the samples that are abose the threshold 
        idx =  np.where(PHI > quantile[k])[0] 
         # RAISING AN ERROR "vae approximation could be bad"
        try: 
            idx[0]
        except IndexError:
            print("Exception raised")
            break
        
        PHI = PHI[idx]
        print(f"Moyenne avant MCMC : {np.mean(PHI)}")
        sample_threshold = sample[idx, :]
        #reduction + centering 
        sd = np.std(sample_threshold, axis = 0) #d 
        mean = np.mean(sample_threshold, axis = 0)

        sample_threshold_centered = (sample_threshold - mean) / sd
        vae = set_vae(d, latent_dim, K_peuso_inputs, sample_threshold_centered)

        encoder, decoder = vae.get_encoder_decoder()

        z_encoded,_, _ = encoder(sample_threshold_centered)

        ps_mean, ps_logvar, z = vae.density_x(N_prior)
        inf_mixture = vae.distrx #openturns Mixture 
        gc.collect()

         ### Plot of the learned distribution by the VAE 
        if kwargs['plot'] :
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
            path = 'figures/latent_d' + str(d) + '_' + str(k) + '.png'
            plot_prior(z, ps_mean.numpy(), ps_logvar.numpy(), path, z_encoded)
        print(k)
        
        PHI, sample, accep_sequence, run_count = MCMC(length_chain, sample_threshold, sample_threshold_centered, PHI, N, d, inf_mixture, rv, phi, sd, mean, quantile[k], run_count)
        quantile.append(np.quantile(PHI, 1-level))
        acceptance.append(accep_sequence / length_chain)
        print(f"moyenne de PHI après MCMC {np.mean(PHI)}")
        print(f"Proba superieur à 3.5 : {np.mean(PHI > threshold)}")
        plt.hist(PHI, density = True)
        plt.show(block = False)
        plt.pause(3)
        plt.close()
        k +=1

    plt.figure(figsize = (15,7))
    if d < 6:
        row = 1
    else : 
        row = d//5
    for i in range(d):
        plt.subplot(row,5,i+1)
        plt.hist(sample[:,i], bins = 100, density=True)
    
    plt.show(block = False)
    plt.pause(3)
    plt.close()


    acceptance = np.array(acceptance)
    mean_accep = np.round(acceptance.mean(axis = 1), 5)
    failure = level**k * np.mean(PHI > threshold)
    print(np.mean(PHI))

    return sequence, quantile, mean_accep, ratio_traj, failure, k, run_count