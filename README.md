# SubsetSimulation_VAE
Innovation for the Subset Simulation Algorithm by adding a proposal kernel based on the VAE

## Package Version  (version python : 3.11.8)    
- tensorflow==2.15.0
- scipy==1.13.0
- openturns==1.22
- numpy==1.26.4

## Three VAE  
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

The folder **Exemple_Test** contains the different exemple including the 4-branches.   

The folder **MMA** contains the scripts of MMA for the 4branch exemple.  

The folder **MoG_VAE** contains the application of the prior MoG with the exemple of the truncated gaussians.  

The folder **SS_VAEVP** contains the application of the prior Vamprior for the exemple of the truncated gaussians and the application of the algorithm **SS_VAE** on the 4-branches.   

-------------------------------------------------------------------------------------
Only Vamprior were used in the SS algorithm ! (MoG not effecient enough)   

You can read the intership report. 

## Numerical Problems  
Memory loss while using tensorflow Model with a loop (while, for ...) 
Nothing changes if we use gc.collect(), del.   
Tracking the memory during the Subset Simulation algorithm with **memory_profiler**, function profiler.   

  


