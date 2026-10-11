# XMACE SCRIPTS 

## Overview
This X-MACE scripts contains the main training scripts used to train and test the autoencoder model
Main segments 
1) Data -> Contains the raw XYZ files required to train the models
2) Outputs -> Every run will be saved into a new run folder 
3) train-scripts -> Contains scripts required to train / transfer a model 
4) test-scripts -> Contains scripts to test the model on test set / calculate stable lifetime

## Base model training

With the local X-MACE package installed, run from the tools repository root:

```bash
cd xmace_scripts/train_scripts
python base_model_trainer.py input/run_000.json
```

The script first reads all the default settings from `train_scripts/input/defaults.json` then recursively
applies the experiment JSON specified in `input/run_000`overrides. 

The first run_000 uses lightweight presets that can be run on a laptop just to test the scripts functionality. Results are nonsense. 

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

## TBC TL / Testing scripts 
