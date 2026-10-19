# XMACE SCRIPTS 

## Overview
X-MACE Scripts contains scripts that can be used to train, fine tune, and test X-MACE models. 
In general, each script requires inputs that specifies the datasets used and model parameters. 
This is broken down into a `defaults.json` which contains default parameters that can be used for most runs. For every run, the run-specific paramters and datafolders will also be specified in a separate json. This json will override the defaults and take precedence everytime. This structure keeps most of the defaults tidy but also maintains flexibility in each runs customisability. 
Most of the customisable parameters are in the default json files. 

## Base model training

To train a base model,  run from `xmace_scripts`:

```bash
python base_model_trainer.py input/base/run_000.json
```

The script first reads all the default settings from `input/base/defaults.json` then recursively
applies the experiment JSON overrides specified in `input/base/run_000.json`.

The first run_000 uses lightweight presets that can be run on a laptop just to test the scripts functionality. Results are nonsense. 
In this example, the training dataset used is the first 7400 dynamic data that has been filtered with the shinitsel toolkit. 

## Testing 

To test a model, run from `xmace_scripts`:

```bash
python test.py input/test/run_000_test_base.json
```

After a model has been trained, either a base model or with transfer learning, the `test.py` script can be used to test the model on specified test sets. 
The script accepts a dictionary of named test datasets, each containing an XYZ file path. The supplied example uses static and dynamic test sets.

The directory containing the outputs of the model should be specified as `run_dir`.
The script can test the best/final model, as well as the individual checkpoint models. This can be specified under the input file, designed so that predictions from models trained at various epochs can be visualised. 

Relative paths are resolved from the directory where the command is run.

For multihead models, each dataset can also specify a `head`, such as `"LF"` or `"HF"`. The reference labels should match the fidelity of the selected head. Running the same test again will overwrite its previous predictions and metrics.

In the sample outputs, the TL models are still untested due to lack of suitable NEVPT test data, but they can be specified in the exact same way as the base model file. 

## Transfer learning 

To transfer learn from an existing model, run from `xmace_scripts`:

```bash
python transfer_learning_trainer.py input/transfer/run_000_freeze.json
```

The script first reads the default settings from `input/transfer/defaults.json`, then applies the overrides from the experiment input file.

The input file specifies the saved model under `base_model_path`, the training data, and the strategies to apply. The base model should be a full saved model, such as `best_model.pt`, rather than a checkpoint containing only weights.

For single-head training, `train_path` can be one file path. For multihead training, it is a dictionary of head labels and file paths, such as `"LF"` and `"HF"`. Validation can be supplied through `valid_path`, or created by splitting each training dataset using `valid_fraction`.

Strategies are applied in the order specified in the input file. The current examples are:

- `run_000_freeze.json`: freezes the shared graph layers and trains the remaining layers on NEVPT2 data.
- `run_000_multihead.json`: creates separate LF and HF heads, then applies LoRA to their decoders.
- `run_000_multihead_correction.json`: uses the LF head as a base and adds a correction head for HF predictions.

To run the multihead examples:

```bash
python transfer_learning_trainer.py input/transfer/run_000_multihead.json
python transfer_learning_trainer.py input/transfer/run_000_multihead_correction.json
```

The correction example requires HF frames to contain `REF_raw_energy_differences`, which stores HF minus LF energies for the same geometry and state ordering. 

## Key Outputs structure 

Base training and transfer learning use the same output structure. The input file specifies the run directory under `output_dir`.

```text
outputs/
└── run_name/
    ├── models/
    │   ├── best_model.pt
    │   ├── metadata.pt
    │   └── checkpoints/
    │       └── checkpoint_epoch_000010.pt
    ├── preds/  <--- Supplied by test scripts
    │   ├── best/
    │   │   ├── static/
    │   │   │   ├── metrics.json
    │   │   │   └── predictions.xyz
    │   │   └── dynamic/
    │   │       ├── metrics.json
    │   │       └── predictions.xyz
    │   └── checkpoint_epoch_000010/
    │       └── static/
    │           ├── metrics.json
    │           └── predictions.xyz
    ├── input/ 
    │   ├── input.json
    │   ├── resolved_config.json
    │   └── split.json
    └── history.json
```

- `models` contains the full saved model, dataset metadata, and periodic checkpoints. If `restore_best` is false, the returned model is saved as `final_model.pt` instead. Checkpoints contain model weights rather than full model objects.
- `preds` contains the testing outputs, grouped by model and dataset name. `metrics.json` records the errors, while `predictions.xyz` contains the original geometries and reference labels alongside the predictions. This folder remains empty until testing is run.
- `input` records the original input and the resolved settings after applying defaults. `split.json` is saved only when validation is created using `valid_fraction`. This is saved for reproducibility and record keeping. 
- `history.json` records the training and validation losses, metrics, learning rates, and epoch information.

For transfer learning, `resolved_config.json` also records the base model path and the strategies used.
