from function.Algo import *
from Exemple_Test.four_branch import four_branch
import matplotlib.pyplot as plt 
#On utilise le prior Vamprior pour apprendre la distribution X|g(X)> s 



d = 70
K = 35
N_prior = 300
length_chain = 25
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
                chain , quantile, acceptance_rate, ratio, Pf_SS, k, Ntot = ss_vae(samples, 3.5, four_branch, .25, 2, K, N_prior,length_chain, plot = True, memory = True)
                print(f"Temps d'exécution {time.time()-start}", file = f)
                ratio = np.array(ratio)
                
                # saving in a file 
                np.savez('Resultats/Dim_' + str(d) , chain, acceptance_rate, ratio)
            else :
                _ , quantile, acceptance_rate, _ , Pf_SS, k, Ntot = ss_vae(samples, 3.5, four_branch, .25, 2, K, N_prior,length_chain, plot = False, memory = False)
                

            cost.append(Ntot)
            estimation.append(Pf_SS)
            print('----------------------------------------', file= f)
            print(f'Une longueur de chaîne de {length_chain}')
            print(f"failure {np.round(Pf_SS, 5)}  with dimension {d}, pseudo_inputs {K}, noyau de {N_prior} et nombre d'appel {Ntot}", file = f)
            print(f"Quantile {np.round(quantile, 5)}", file = f)

            # acceptance_rate = np.array(acceptance_rate)

            # print(f"Le taux d'aceptation en moyenne pour chaque evenement {np.round(acceptance_rate.mean(axis = 1), 5)}")
            print(f"Rate {acceptance_rate}", file = f)


            print(f'Etape {elt} terminée')
    estimation = np.array(estimation)
    cost = np.array(cost)
    expectation = estimation.mean()
    sd = np.array(estimation).std()
    # print(f'Estimation Pf par VAESS : {expectation}', file = f)
    # print(f"Variance de l'estimateur VAESS : {sd}", file= f)
    # print(f"Le c.o.v {sd/expectation}",file=f)

# path1 = 'Resultats/Estimation_d' + str(d)
# path2 = 'Resultats/Cost_d' + str(d)
# np.save(path1, estimation)
# np.save(path2, cost)








































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