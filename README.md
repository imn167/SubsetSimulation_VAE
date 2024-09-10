# SubsetSimulation_VAE
Innovation for the Subset Simulation Algorithm by adding a proposal kernel based on the VAE

## Package Version  (version python : 3.11.8)    
- tensorflow==2.15.0
- scipy==1.13.0
- openturns==1.22
- numpy==1.26.4


## Problems encountered  (numerical)  
Memory loss while using tensorflow Model with a loop (while, for ...) 
Nothing changes if we use gc.collect(), del.   
Tracking the memory during the Subset Simulation algorithm with **memory_profiler**, function profiler.   

## Three VAE coded   
- Classical with a gaussian prior
- Mixture of Gaussians *MoG*
- Variational Mixture of Posteriors *Vamprior*

Only Vamprior were used in the SS algorithm ! (MoG not effecient enough)   

You can read the intership report.   


