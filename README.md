# SubsetSimulation_VAE
Enhancement of the Subset Simulation Algorithm by integrating a proposal kernel based on the Variational Autoencoder (VAE)

## Package Version (python version: 3.11.8)    
- tensorflow==2.15.0
- scipy==1.13.0
- openturns==1.22
- numpy==1.26.4

## Three Types of VAE
- Classical with a gaussian prior
- Mixture of Gaussians *MoG*
- Variational Mixture of Posteriors *Vamprior*

## Content  
The folder **function** contains  
- the vae's keras class in *VAE.py*  
- Implementation of the EM algorithm for a gaussian case in *EM.py*
- Implementation of the fixed probability Subset Simulation in *Vanilla_SS*  
- Implementation of the Modified Metropolis Algorithm in *MMA.py*
- Implementation of the M-H algorithms with vae in *Algo.py*

The folder **Exemple_Test** contains several examples, including the 4-branch problem.

The folder **MMA** contains script for MMA applied to the 4-branch example.

The folder **MoG_VAE** demonstrates the application of the Mixture of Gaussians (MoG) prior with truncated Gaussian examples.

The folder **SS_VAEVP** includes the application of the Vamprior to truncated Gaussian examples and the implementation of the SS_VAE algorithm on the 4-branch problem. 

-------------------------------------------------------------------------------------
Only the Vamprior was used in the Subset Simulation algorithm, as the MoG prior was not sufficiently effective.   

You can read the intership report. 

## Numerical Problems  
There is a memory leak when using TensorFlow models within loops (while, for, etc.).
Using `gc.collect()` or `del` does not resolve the issue.
Memory usage is tracked during the Subset Simulation algorithm with **memory_profiler** and the function profiler. 

  


