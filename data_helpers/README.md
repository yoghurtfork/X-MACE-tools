# Data Helpers

This directory will contain reusable utilities that prepare molecular datasets for X-MACE workflows. 
Intent is for these utils to be easily imported and used in subsequent workflows 
They are features that are required for differnt data generation steps 


## Helper Overview

### Generation
These are helpers that help to generate the various xyz geometries used in training 

1) make_grid.py
    - Utils to help generate a 2D grid based on equilibrium geometry 
    - Input Equilibrium geometry and target angles / bond length
    - Output the extended xyz file of all geometries in the 2D grid 
2) augment.py 
    - Utils to help augment a particular geometry with NMS 
    - Can specify temperature-based or manually controlled perturbations 
    - Calculates normal modes and hessian, outputs augmented non equilibrium geometries 
    - Used as a helper to generate data for some of the data generation strategies 

### Filtering
1) sanity_filter.py
    - Deals with noisy dynamic data 
    - Input an extended xyz file of geometries 
    - Automatically filters out unphysical bad geometries by bond length / energy / angles (?)

* Possibly can add a utils file here
* Reference from Shnitsel Toolkit

### Test-Set Sampling
Helper utils to generate custom test set used 
1) near_CI.py
    - Input the extended xyz dataset 
    - Functions to sample from near CI intersections 
    - Creates a custom test set with even balance between near-CI geometries and near equilibrium ones

### AzoFlip Import
For future work with DANN
1) read_azoflip_data.py
    - Reads the DANN AzoFLip database messagepack files 

### Transfer-Learning Sampling
Key folder containing utils required to sample datapoints for Transfer Learning 
So far tested briefly but no real results yet since we stuck at base training 
1) descriptors.py 
    - Ouputs descriptors of each geometry 
    - Descriptors are used to sample from later on 
2) selectors.py 
    - Selects geometries based on selection strategy 
    - Furthest point / clustering etc 
3) run_TL_sampling.py
    - Overall orchestrator file 
* TBC maybe need more robustness considering we won't only sample from grid anymore
* Consider that we might need to sample from grid + augmented points / dynamic 
