import sys 

sys.path.append('..')

from function.Algo import *
from Exemple_Test.four_branch import four_branch
import matplotlib.pyplot as plt 



d = 10
Pf = 1.15e-6
s = 5

K = 35
N_prior = 300
length_chain = 20
p0 = .25
latent_dim = 2

batch = 12

file = 'txtfiles/SSVAE_' + str(d) + '.txt'
with open(file, 'a') as f :
    cost = deque()
    estimation = deque()
    for elt in range(1) : 
        # np.random.seed(123)#fixed seed to compare with other results 
        samples = np.random.normal(size = (10000, d))
        gc.collect()
        N, d = samples.shape
        if elt == 0:
            start = time.time()
            chain , quantile, acceptance_rate, ratio, Pf_SS, k, Ntot = ss_vae(samples, s, four_branch, p0, latent_dim, K, N_prior,length_chain, plot = True, memory = True)
            if Pf_SS is None :
                continue
            # print('----------------------------------------------------------------------', file= f)
            # print('----------------------------------------------------------------------', file= f)

            # print(f"Temps d'exécution de l'algorithme {time.time()-start}", file = f)
            ratio = np.array(ratio)
            chain = np.array(chain)
            
            # saving in a file 
            # np.savez('Resultats/Dim_' + str(d) , chain, acceptance_rate, ratio)
        else :
            _ , quantile, acceptance_rate, _ , Pf_SS, k, Ntot = ss_vae(samples, s, four_branch, p0, latent_dim, K, N_prior,length_chain, plot = False, memory = False)
            if Pf_SS is None :
                continue
                
        print('---------------------------')
        print(f'Pf_SS {Pf_SS}')
        print(f"Rate {acceptance_rate}")
        cost.append(Ntot)
        estimation.append(Pf_SS)
        print(f'Estimation {elt+1} terminée')
        print('-------------------------------')

    cost = np.array(cost)
    estimation = np.array(estimation)

    cov = estimation.std() / estimation.mean()
    Ntot = int(cost.mean())
    Nreq = int((1-Pf)/ (Pf*cov**2)) +1
        
    print('----------------------------------------------------------------------', file= f)
    print(f'SS_VAE avec vamprior k = {K} | {N_prior} de gaussiennes pour le mélange fini ', file= f)
    print('----------------------------------------------------------------------', file= f)
    print(f'dimension d = {d} | Pf_SS = {np.round(estimation.mean(), 8)} | cov(Pf_SS) = {np.round(cov, 8)} | Ntot = {Ntot} | l = {length_chain}', file=f)
    print('----------------------------------------------------------------------', file= f)
    print(f"Rate {acceptance_rate}", file = f)
    print('----------------------------------------------------------------------', file= f)
    print(f"Quantile {np.round(quantile, 5)}", file = f)


    path1 = 'Resultats/Estimation_d' + str(d) + '/memory'
    np.savez(path1, chain, ratio)







































# ## Avec de faible donnnées ?
# N =10000
# d = 20
# X = np.random.normal(size= (N, d ))

# quantile = np.quantile(four_branch(X), .8)

# idx = np.where(four_branch(X)> quantile)[0]
# sample = X[idx, :] # .2N x d
# print(sample.shape)


# prior = VP(d, 35)
# prior.initialized_ps(sample, 1e-3)

# encoder = Encoder(d, 2, True)
# decoder = Decoder(d, 2, True )

# ae = AutoEncoder(encoder, decoder)

# ae.initialized_ae(sample, 1e-3, 150, 32)

# plt.plot(ae.history.history['loss'])
# plt.show()
# _, _, z_var = encoder(sample)

# plt.scatter(z_var[:, 0], z_var[:, 1], s = 6)
# plt.show()

# from sklearn.decomposition import PCA

# pca = PCA(2)

# linear_decomp = pca.fit_transform(sample)

# plt.scatter(linear_decomp[:, 0], linear_decomp[:, 1], s = 6)
# plt.show()

# #creation d'un auto encoder avec réh=gularisation 

# sd_sample = np.std(sample, axis = 1)
# mean_sample = np.mean(sample, axis = 1)
# print(mean_sample.shape, sd_sample.shape)

# ss = (sample - np.expand_dims(mean_sample, axis=1))/np.expand_dims(sd_sample, axis=1)
# print(ss)

# ss = np.expand_dims(ss, axis=2)

# input_img = tf.keras.Input(shape = (d,1)) #input 
# encoded = layers.Conv1D(filters = 8, activation = 'relu', kernel_size = 3, padding="same", strides=1)(input_img)
# encoded = layers.MaxPooling1D(pool_size = 2)(encoded)
# encoded = layers.UpSampling1D(size = 2)
# encoded = layers.Conv1D(filters = 8, activation = 'relu', kernel_size = 2, padding="same", strides=1)(encoded)
# encoded = layers.MaxPooling1D(pool_size = 2)(encoded)
# encoded = layers.Flatten()(encoded)
# encoded = layers.Dense(2, activation = 'relu')(encoded)

# decoded = layers.Dense(2, activation = 'linear')(encoded)
# decoded = layers.Dense(32, activation = 'linear')(decoded)
# decoded = layers.Dense(d, activation = 'linear')(decoded) #[0,1] values 

# autoencoder = tf.keras.Model(input_img, decoded)
# autoencoder.compile(optimizer = 'adam', loss = 'mse') #cross entropy as metrics 
# autoencoder.fit(ss, ss, epochs = 150, batch_size = 100, shuffle = True)

# plt.plot(autoencoder.history.history['loss'])
# plt.show()

# encoder = tf.keras.Model(input_img, encoded)

# latent_sapce = encoder(ss)

# plt.scatter(latent_sapce[:, 0], latent_sapce[:, 1], s = 6)
# plt.show()