import numpy as np 
import matplotlib.pyplot as plt 

chain = np.load('mcmcChain.npy')

N, d = chain.shape 
plt.figure(figsize= (15,7))
for i in range(20):
  plt.subplot(4,5, i+1)
  plt.plot(chain[:, i])
plt.show()

