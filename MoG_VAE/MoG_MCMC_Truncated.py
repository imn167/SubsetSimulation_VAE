import sys 
import time
sys.path.append('..')
# sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS')

# sys.path.append('/Users/ibouafia/Desktop/Stage/VAE/VAE_SS/figures_ss')

from function.EM import *
from function.VAE import *
import matplotlib
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Sequential", [ '#80ef80', 'green', "limegreen", 'yellow', 'orange'])

def plot_prior( z_variationnel, mu, sigma2,  path, z_encoded, scatter = True): #z_mcmc
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
    if scatter :
        plt.scatter(z_variationnel[:, 0], z_variationnel[:, 1], s=4, c = 'red', alpha = .1)
        # plt.scatter(z_encoded[:, 0], z_encoded[:, 1], s=4, c = 'blue', alpha = .1)
        # plt.scatter(z_mcmc[:, 0], z_mcmc[:, 1], s=6, c = 'purple', alpha = .1)
    
    # plt.title('Gaussians of the Vamprior Mixture')
    plt.savefig(path)
    plt.show(block = False)
    plt.pause(3)
    plt.close()


start = time.time()
#importing data 
two_mode = np.load('two_component_truncated50.npy') #10000 x 50
#print(two_mode)
N, d = two_mode.shape

print(N,d)
####### Establishing a Latent spae for the data ###########

encoder = Encoder(d,2, True)
decoder = Decoder(d,2, True)

ae = AutoEncoder(encoder, decoder)
ae.initialized_ae(two_mode, 1e-3, 100, 80)

z_mean, _, _ = encoder(two_mode)

plt.scatter(z_mean[:, 0], z_mean[:, 1])
plt.show()

K = 25
prior = MoGPrior(2,K) # 35 gaussienne diagonales dnas R2
w_t, mu_t, sigma2_t, n_iter = EM(z_mean, prior, 1000, 1e-3)
print(w_t, n_iter)



path_latent = 'figures/Truncated_AELatent_MoG_d' + str(d)  + '.png'
plot_prior(z_mean, mu_t, sigma2_t , path_latent, z_mean, False)


prior.means.assign(tf.constant(mu_t, dtype=tf.float32))
prior.logvars.assign(tf.math.log(tf.constant(sigma2_t, dtype=tf.float32)))
prior.w.assign(tf.constant(w_t.reshape(1,-1), dtype=tf.float32))


vae = VAE(encoder, decoder, prior, name_prior = 'MoG')
vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
vae.fit(two_mode, epochs = 150, batch_size = 100, shuffle = True) 


z_mean, _, z_variationnel = encoder(two_mode)

path_latent = 'figures/Truncated_latent_MoG_d' + str(d)  + '.png'
plot_prior(z_variationnel, prior.means.numpy(), prior.logvars.numpy() , path_latent, z_mean, False)
print(prior.w.numpy())

print(time.time() - start)

ColDist = [ot.Normal(np.array(mu), np.exp(0.5*np.array(sigma))) for mu, sigma in zip(prior.means, prior.logvars)]
weight = np.array(tf.nn.softmax(prior.w, axis =1)).reshape(-1)
myMixture = ot.Mixture(ColDist, weight)
z =myMixture.getSample(10000)
z= np.array(z)
mean_x, log_var_x = decoder(z) #we get each mean and log variance of the several distribution then we sample from it

sample = np.random.normal(loc=mean_x, scale= tf.exp(log_var_x/2))


xx= np.linspace(-5,5, 1000)
dist1 = ot.TruncatedDistribution(ot.Normal(1), 2., ot.TruncatedDistribution.LOWER)
dist2 = ot.TruncatedDistribution(ot.Normal(1), -2., ot.TruncatedDistribution.UPPER)
dist = ot.Mixture([dist1,dist2])


path_hist = 'figures/Truncated_hist_MoG_d' + str(d)  + '.png'
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

#### 2d plot, truncated area 
plt.hist2d(sample[:, 0], sample[:, 1], bins= (1000, 1000), cmap = plt.cm.jet)
plt.show()

def truncated_density(x, rv, n_comp, distri):
  return rv.pdf(x[n_comp : ]) * distri.computePDF(x[0]) * distri.computePDF(x[1])

######################################### MCMC ##############################

#approximation de la loi empirique du mélange 
Nc = 300
# MoG = prior.sampling() 
gaussians = [ot.Normal(mu, sigma) for mu, sigma in zip(prior.means.numpy(), np.exp(prior.logvars.numpy() * 0.5))]
w = tf.nn.softmax(prior.w)
MoG = ot.Mixture(gaussians, (w.numpy()).reshape(-1))
Z = np.array(MoG.getSample(Nc))
idx = np.random.choice(np.arange(Nc), size = Nc)

Z_N = Z[idx, :] 

mean_x, logvar_x  = decoder(Z_N) #lois conditionnelles 
gaussians = [ot.Normal(mu, sigma) for mu, sigma in zip(mean_x.numpy(), np.exp(logvar_x.numpy() * 0.5))]
inf_mixture = ot.Mixture(gaussians, np.ones(Nc)/Nc)
M = 1000 #taille chaine MCMC
chain = np.zeros((M+1 , d))

chain[0] = two_mode[6] #initialization 

rv = sp.multivariate_normal(mean = np.zeros(d-2)) 

acceptance = 0
for i in range(M):
  
  r =  np.random.choice(np.arange(Nc))
  zr = Z_N[r].reshape(1,-1)
  mu, logvar =  decoder(zr) 
  rv_gom = sp.multivariate_normal(mean = mu.numpy().reshape(-1), cov = np.exp(logvar.numpy()).reshape(-1))
  candidat = rv_gom.rvs() #d
  ratio_traj = list()
  ratio = (truncated_density(candidat, rv, 2, dist) * inf_mixture.computePDF(chain[i])) / (truncated_density(chain[i], rv, 2, dist) * inf_mixture.computePDF(candidat))
  ratio_traj.append(ratio)
  u = np.random.uniform()
  if u < ratio:
    chain[i+1] = candidat
    acceptance += 1 
    gaussians.append(rv_gom)
  else :
    chain[i+1] = chain[i]

print(acceptance/M)
plt.figure(figsize = (15, 7))
for i in range(10):
  plt.subplot(2,5, i+1)
  plt.plot(chain[:, i])
plt.show()

plt.hist(chain[:, 0], density=True, bins=100)
plt.show()
plt.hist(chain[:, 1], density = True, bins=100)
plt.show()
plt.hist(chain[:, 2], density = True, bins= 100)
plt.show()

#save chain & ratio
# np.save('GoM_MCMC_chain.npy', chain)
# np.save('GoM_ratio_MCMC.npy', np.array(ratio_traj))