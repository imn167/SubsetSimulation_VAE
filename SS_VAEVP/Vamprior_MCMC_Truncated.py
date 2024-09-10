#################################################################################
#################### Test of the Vamprior ############

import openturns as ot 
from function.VAE import *
import sys
import matplotlib.pyplot as plt 
from IPython.display import clear_output



import matplotlib
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Sequential", [ '#80ef80', 'green', "limegreen", 'yellow', 'orange'])

dist1 = ot.TruncatedDistribution(ot.Normal(1), 2., ot.TruncatedDistribution.LOWER)
dist2 = ot.TruncatedDistribution(ot.Normal(1), -2., ot.TruncatedDistribution.UPPER)
dist = ot.Mixture([dist1,dist2])

#Testing on the Truncated Gaussians 
four_zone = np.load("./Exemple_Test/two_component_truncated50.npy")

N, d = four_zone.shape


#INITIALIZATION
latent = 2
K = 35
encoder = Encoder(d, latent, True)
decoder = Decoder(d, latent, True)

ae = AutoEncoder(encoder, decoder)
ae.initialized_ae(four_zone, 1e-3, 100, 80)

#init des peseudo-inputs 
prior = VP(d, K)
prior.initialized_ps(four_zone, 0.001)



####################################vae training 
vae = VAE(encoder, decoder, prior, name_prior = 'vamprior')
vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
vae.fit(four_zone, epochs = 150, batch_size = 100, shuffle = True, verbose = 1)

## PLOT LOSS TRAJ
plot_history = True 

if plot_history : 
    plt.plot(ae.history.history['loss'], label = 'AE_loss')
    plt.legend()
    plt.show()

    plt.plot(prior.history.history['loss'], label = 'Vp_Prior_Loss')
    plt.legend()
    plt.show()

    # plt.plot(vae.history.history['loss'], label = 'VAE_loss')
    plt.plot(vae.history.history['kl_loss'], label = 'KL_loss')
    plt.legend()
    plt.savefig('figures/Truncated_Klloss_d' + str(d)  + '.png')
    plt.show()
    plt.plot(vae.history.history['reconstruction_loss'], label = 'Reconstruction_loss')
    plt.legend()
    plt.savefig('figures/Truncated_Reconstloss_d' + str(d)  + '.png')
    plt.show()
    

#### LATENT SPACE #######
pseudo_inputs = prior(tf.eye(prior.K))
ps_mean, ps_logvar, _ = encoder(pseudo_inputs) 
aggregated_posterior = prior.mixture(ps_mean, ps_logvar) 

z_mean, _, z_variationnel = encoder(four_zone)

#Plot prior if latent_dim == 2
def plot_prior( z_variationnel, mu, sigma2,  path, z_encoded, scatter = True): #z_mcmc
    K = mu.shape[0]
    
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
        
    plt.pcolormesh(X, Y, pdf, cmap = cmap)
    if scatter :
        plt.scatter(z_variationnel[:, 0], z_variationnel[:, 1], s=4, c = 'red', alpha = .1)
        plt.scatter(z_encoded[:, 0], z_encoded[:, 1], s=4, c = 'blue', alpha = .1)
        # plt.scatter(z_mcmc[:, 0], z_mcmc[:, 1], s=6, c = 'purple', alpha = .1) 
    
    # plt.title('Gaussians of the Vamprior Mixture')
    plt.savefig(path)
    plt.show(block = False)
    plt.pause(3)
    plt.close()

path_latent = 'figures/Truncated_latent_d' + str(d)  + '.png'
plot_prior(z_variationnel, ps_mean.numpy(), ps_logvar.numpy(), path_latent, z_mean, False)


##### MCMC M-H
Nc = 1000
z = np.array(aggregated_posterior.sample(Nc))
mean_x, log_var_x = decoder(z) 

sample = np.random.normal(loc=mean_x.numpy(), scale= tf.exp(log_var_x.numpy()/2))

xx= np.linspace(-5,5, 1000)

path_hist = 'figures/Truncated_hist_d' + str(d)  + '.png'
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
plt.savefig(path_hist)
plt.show()

### Truncated Gaussians Density
def truncated_density(x, rv, n_comp, distri):
  return rv.pdf(x[ n_comp : ]) * distri.computePDF(x[0].reshape(-1,1)) * distri.computePDF(x[1].reshape(-1,1))
rv = sp.multivariate_normal(np.zeros(d-2))


#################################################################################################################################

z_mean, _, z_variationnel = encoder(four_zone)

Nc = 300
#### Finite Mixture 
idx = np.random.choice(np.arange(Nc), size = Nc)
ZN = z[idx, :]
x_mean, x_logvar = decoder(ZN)
gaussians = [ot.Normal(mu, sigma) for mu, sigma in zip(x_mean.numpy(), np.exp(x_logvar.numpy() * 0.5))]
inf_mixture = ot.Mixture(gaussians, np.ones(Nc)/Nc)
M = 10000 
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

print('-----------------------------------------------------')
print(acceptance/M)

z_mcmc, _, _ = encoder(tf.constant(chain))

path_latent = 'figures/Truncated_latent_MCMC_d' + str(d)  + '.png'
plot_prior(z_variationnel, ps_mean.numpy(), ps_logvar.numpy(), path_latent, z_mcmc, True)

z_mcmc, _, _ = encoder(tf.constant(chain[300:]))
path_latent = 'figures/Truncated_latent1_MCMC_d' + str(d)  + '.png'
plot_prior(z_variationnel, ps_mean.numpy(), ps_logvar.numpy(), path_latent, z_mcmc, True)

# np.save('ratio-traj.npy', ratio_traj)
# np.save('mcmcChain.npy', chain)

path_hist = 'figures/Truncated_hist_MCMC_d' + str(d)  + '.png'
plt.figure(figsize= (12,7))
for i in range(20):
  plt.subplot(5,5,i+1)
  plt.hist(chain[:, i], bins = 100, density=True);
plt.savefig(path_hist)
plt.show()

path_hist = 'figures/Truncated_hist_MCMC1_d' + str(d)  + '.png'
plt.figure(figsize= (12,7))
for i in range(20):
  plt.subplot(5,5,i+1)
  plt.hist(chain[300:, i], bins = 100, density=True); #300 burned 
plt.savefig(path_hist)
plt.show()








