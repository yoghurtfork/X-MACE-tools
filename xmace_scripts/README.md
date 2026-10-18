# XMACE SCRIPTS 

## Overview
X-MACE Scripts contains scripts that can be used to train, fine tune, and test X-MACE models. 
In general, each script requires inputs that specifies the datasets used and model parameters. 
This is broken down into a defaults.json which contains default parameters that can be used for most runs. For every run, the run-specific paramters and datafolders will also be specified in a separate json. This json will override the defaults and take precedence everytime. This structure keeps most of the defaults tidy but also maintains flexibility in each runs customisability. 
The full list of customisable parameters are in the default json files. 

## Base model training

With the local X-MACE package installed, run from the tools repository root:

```bash
cd xmace_scripts
python base_model_trainer.py input/base/run_000.json
```

The script first reads all the default settings from `train_scripts/input/defaults.json` then recursively
applies the experiment JSON specified in `input/run_000`overrides. 

The first run_000 uses lightweight presets that can be run on a laptop just to test the scripts functionality. Results are nonsense.
In this example, the training dataset used is the first 7400 dynamic data that has been filtered with the shinitsel toolkit. 

## Testing 

After a model has been trained, either a base model or with transfer learning, the `test.py` script can be used to test the model on specified test sets.
The directory containing the outputs of the model should be specified as `run_dir`.
The output directories of the dynamic and static test sets should also be specified. 

In this script, the best/final model, as well as the individual checkpoint models can be tested, so that predictions from models trained at various epochs can be visualised. 

In the sample outputs, the TL models are still untested due to lack of suitable NEVPT test data, but they can be specified in the exact same way as the base model file. 

## Transfer learning 
TBC

## Key Outputs structure 

TBC