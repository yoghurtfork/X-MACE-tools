# XMACE SCRIPTS 

## Overview
This X-MACE scripts contains the main training scripts used to train and test the autoencoder model
Main segments 
1) Data -> Contains the raw XYZ files required to train the models
2) Outputs -> Every run will be saved into a new run folder 
3) train-scripts -> Contains scripts required to train / transfer a model 
4) test-scripts -> Contains scripts to test the model on test set / calculate stable lifetime

## Key Outputs structure 
Outputs 
    - run 001 
        - models 
            - best_model.pt 
            - checkpoints
                - epoch05.pt
                - epoch10.pt 
                - epcoh15.pt...
        - preds
            - dynamic_test_set.xyz 
            - static_test_set.xyz
        - input
            - original input file containing training details 
            - copied in while training base model 

## Rough Workflow
train scripts will run a python file that inputs an input_json containing training paramters 
model hyperparamters are stored in inputs/default.json 
By default they will be used, but can be parsed as a new field in the input_json
train_model will run training based on input xyz files 
Output model and checkpoint models as a series of .pt files 
It will create a new run_00x folder, and place best_model.pt / checkpoints inside 
It will also copy the input json inside that run folder, so all training details are labelled 

test_scripts will also run from an input json
Input json will specify a path for the models 
script will first load in the model 
It will make the required predictions on various test sets 
It will save all predictions under the folder preds 
All preds output will be an extended xyz file, with reference and Preds present 

test_lifetime.py will be run in POST after predictions are done 
IE it will read in the extended XYZ file and calculate lifetime based on that 
Best called in notebook as a function 

        