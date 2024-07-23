#################################################################################
#################### Test of the Vamprior ############
from function.VAE import *
import sys
import matplotlib.pyplot as plt 
from IPython.display import clear_output

dist1 = ot.TruncatedDistribution(ot.Normal(1), 2., ot.TruncatedDistribution.LOWER)
dist2 = ot.TruncatedDistribution(ot.Normal(1), -2., ot.TruncatedDistribution.UPPER)
dist = ot.Mixture([dist1,dist2])

#Testing on the Truncated Gaussians 
four_zone = np.load("./Exemple_Test/two_component_truncated50.npy")

N, d = four_zone.shape
#if we want to check the init
plot_history = True 

#Init vae 
latent = 2
K = 75
encoder = Encoder(d, latent, True)
decoder = Decoder(d, latent, True)

#init de l'espace latent 
ae = AutoEncoder(encoder, decoder)
ae.initialized_ae(four_zone, 1e-3)

#init des peseudo-inputs 
prior = VP(d, 35)
prior.initialized_ps(four_zone, 0.001)

if plot_history : 
    plt.plot(ae.history.history['loss'])
    plt.show()

    plt.plot(prior.history.history['loss'])
    plt.show()


#vae trainig 
vae = VAE(encoder, decoder, prior, name_prior = 'vamprior')
vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
vae.fit(four_zone, epochs = 150, batch_size = 100, shuffle = True, verbose = 0)



#### etude de l'espae latent #######
pseudo_inputs = prior(tf.eye(prior.K))
ps_mean, ps_logvar, _ = encoder(pseudo_inputs) #K x latent_dim 
aggregated_posterior = prior.mixture(ps_mean, ps_logvar) #prior 

_, _, z_variationnel = encoder(four_zone)

def plot_prior( mu, sigma2, x1,x2, y1, y2):
    K = mu.shape[0]
    X, Y = np.meshgrid(np.linspace(x1,x2, 100), np.linspace(y1,y2, 1000))
    pos = np.dstack((X,Y))
    rv =  [sp.multivariate_normal(mu, np.diag(sigma)) for mu, sigma in zip(mu, tf.exp(sigma2))]
    for i in range(K):
        plt.contour(X, Y,np.array( rv[i].pdf(pos)), levels = [.1,.2,.5])
    plt.scatter(z_variationnel[:, 0], z_variationnel[:, 1], s=6)
    
    plt.title('Gaussians of the Vamprior Mixture')
    # plt.savefig('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/figures_ss/Trunacted_Gaussian_Vamprior.png')
    plt.show()


##### MCMC M-H
Nc = 10000
z = np.array(aggregated_posterior.sample(Nc))
mean_x, log_var_x = decoder(z) #we get each mean and log variance of the several distribution then we sample from it

sample = np.random.normal(loc=mean_x.numpy(), scale= tf.exp(log_var_x.numpy()/2))

xx= np.linspace(-5,5, 1000)

plt.figure(figsize = (12,7))
for i in range(2,20):
  plt.subplot(5,5,i+1)
  plt.hist(sample[:,i], bins = 100, density=True);
  plt.plot(xx, sp.norm.pdf(xx))

plt.subplot(5,5,1)
plt.hist(sample[:,0], bins = 100, density=True);
plt.plot(xx, dist.computePDF(xx.reshape(-1,1)))

plt.subplot(5,5,2)
plt.hist(sample[:,1], bins = 100, density=True);
plt.plot(xx, dist.computePDF(xx.reshape(-1,1)))
plt.show()

### Analytical expression of the data distribution 
def truncated_density(x, rv, n_comp, distri):
  return rv.pdf(x[ n_comp : ]) * distri.computePDF(x[0].reshape(-1,1)) * distri.computePDF(x[1].reshape(-1,1))
rv = sp.multivariate_normal(np.zeros(d-2))
#print(truncated_density(sample, rv, 2, dist))


#### approximation of the infinite gaussian mixture 
idx = np.random.choice(np.arange(Nc), size = Nc)
ZN = z[idx, :]
x_mean, x_logvar = decoder(ZN)
gaussians = [ot.Normal(mu, sigma) for mu, sigma in zip(x_mean.numpy(), np.exp(x_logvar.numpy() * 0.5))]
inf_mixture = ot.Mixture(gaussians, np.ones(Nc)/Nc)
M = 1000 #taille chaine MCMC
chain = np.zeros((M+1 , d))

###### MCMC algorithm 
chain[0] = four_zone[6]
acceptance = 0
ratio_traj = list()
for i in range(M):
  
    r =  np.random.choice(np.arange(Nc))
    zr = ZN[r].reshape(1,-1)
    mu, logvar =  decoder(zr) # d 
    rv_gom = sp.multivariate_normal(mean = mu.numpy().reshape(-1), cov = np.exp(logvar.numpy()).reshape(-1))
    candidat = rv_gom.rvs() #d
  
    ratio = truncated_density(candidat, rv, 2, dist) * inf_mixture.computePDF(chain[i]) / (truncated_density(chain[i], rv, 2, dist) * inf_mixture.computePDF(candidat))
    ratio = ratio.reshape(-1)
    ratio_traj.append(ratio)
    u = np.random.uniform()
    if u < ratio:
        chain[i+1] = candidat
        acceptance += 1 
        gaussians.append(rv_gom)
    else :
        chain[i+1] = chain[i]
    if i%100 == 0:
        clear_output(wait=True)
        print("boucle %d terminée" %(i))


print(acceptance/M)

np.save('ratio-traj.npy', ratio_traj)
np.save('mcmcChain.npy', chain)



#### Bon apprentissage de la distribution donc un bon noyau de proposition 






