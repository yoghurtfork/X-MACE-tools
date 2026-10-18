# X-MACE TOOLS  

Tools for the X-MACE project!

## Data helpers

This directory helps to prepare molecular datasets for X-MACE workflows

Contents:

- Scripts for generating data

- Scripts for augmenting data

- Scripts for reading azoflip MessagePack files

- Tools for filtering data

- Tools for sampling a test set

- Scripts for sampling transfer learning datasets

## Tutorials 

This directory contains a few notebooks documenting how to use the various classes built in the X-MACE-TL repository.
These classes are used to interface and train the X-MACE model without using the original CLI-Script. 
Intended for greater flexibility in terms of training whilst maintaining usability.

Contents:

- 00_model_inspection documents the various changes in raw architecture made to the original model

- 01_base_model_training documents basic usage of the various trainer classes for training of a base model 

- 02_transfer_learning_strategies documents a new class called strategies that can be used to modify the model for transfer learning, eg via freezing weights or via LORA injection

- 03_multihead_training documents an extension of strategies that involves transfer learning with multi head data, more complicated as it involves changes to the dataloaders as well. 

## X-MACE scripts

This directory contains scripts that can be used to train and test X-MACE models. They are built using the classes documented in the tutorials. Each script accepts inputs from an input file specified by user, and the scripts will execute that specific type of training with the datafiile paths and various parameters. 

Intended use is for the functions in `Data helpers` to be used to generate various xyz files for different kinds of datasets, to test out different kinds of data sampling methods. The file paths of these xyz files can then be specified into different input files in the scripts and used to train various base models. 

Contents:

- Scripts for training a base model 

- Scripts for transfer learning from a base model 

- Scripts for testing a model on a particular test set 

- Sample data, input and output folders 


## SHARC scripts

SHARC itself runs from a separate program, but this directory helps to prepare and launch ensembles from configuration files

Contents:

- Scripts for preparing initial condition files

- Scripts for running SHARC MACE

- Scripts for running SHARC Molcas

- Tools for analysing SHARC trajectories/ensembles, including calculating 'MD match'

## Analysis tools

This directory contains tools to analyse data and model prediction results

- Tools for visualising dataset geometries and energies

- Scripts for running model predictions

- Tools for visualising model predictions and residuals

```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⠶⢦⡀⠀⠀⢀⣀⡀⠀⠀
⠀⠀⠀⣀⣤⣄⡀⠀⠀⠀⠀⠀⠀⢠⠖⠒⢆⠀⠀⠀⡏⠀⠀⠀⠙⣇⡞⠋⠙⢿⣷
⠀⠀⢀⡟⠁⠉⠛⣧⠀⠀⠀⠀⣠⠇⠀⠀⠈⢇⠀⢸⠁⠀⠀⠀⠀⡟⠀⠀⠀⠀⣿⠃
⠀⠀⣾⠃⠀⠀⠀⠈⠁⠒⠊⠉⠁⠀⠀⠀⠀⠘⢢⡈⡆⠀⠀⠀⣰⠃⠀⠀⠀⠀⡟⠀
⠀⣀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡴⠋⠙⠲⡄⠀⠁⠀⠀⠀⢀⣞⠁⠀
⣰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⡼⠁⠀⠀⠀⢧⠀⠀⠀⠀⠐⠋⠀⢱⡀   thanks
⡇⠀⠀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠃⠀⣏⠀⠸⣄⡤⠇⠀⠀⢀⡞⠀⠀⠀⢠⠇   hope it helps
⡇⠀⠀⠀⠉⠀⠀⢠⠀⣠⣄⣰⠀⠀⠀⠀⠀⠈⢧⡀⠀⠀⠀⠀⠀⢸⡀⠀⠀⡰⠃⠀
⠙⣄⠀⠀⠀⠀⠀⠈⠛⠁⠀⠀⠀⠀⠀⣀⣀⣴⠿⠃⠀⠀⠀⠀⠀⢀⡛⠒⠚⠁⠀⠀
⠀⠈⠳⠤⣀⣀⣀⣀⣀⣠⡤⠤⠔⠒⠈⠁⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡖⠦⠤⣄⣀⡤⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢀⡞⠁⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⢠⡛⣆⠀⠀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠸⣇⣀⣴⢀⣧⠀⠀⠀⠀⠀⠀⠀⢀⡏⠣⣈⠙⠉⢁⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠈⠉⠈⡇⢰⠒⠒⠒⢺⣠⠏⠀⠀⠈⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠙⠚⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```