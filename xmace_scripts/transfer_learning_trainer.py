"""Train a saved X-MACE model using ordered transfer-learning strategies.

Run from xmace_scripts: python transfer_learning_trainer.py input/transfer/run_000.json
Paths are relative to the working directory. Supply compatible labelled datasets
and a full model; this starts a new optimisation run, not an exact resumption.
Expansion strategies expect a single-head base model and should precede freezing
or LoRA. Correction expansion currently requires three states and latent size 16;
its head order defines cumulative routes and successive energy-difference targets.
"""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from ase.data import chemical_symbols
from ase.io import read

from mace.data.atom_data_loader import AtomDataLoaderBuilder
from mace.modules.loss import InvariantsWeightedEnergyForcesNacsDipoleLoss
from mace.training.strategies import (
    FreezeStrategy, LoRAStrategy, MultiHeadCorrectionStrategy, MultiHeadStrategy, NaiveStrategy,
)
from mace.training.trainer import Trainer

if __package__:
    from .utils import merge_config, split_atoms
else:
    from utils import merge_config, split_atoms


def main():
    # Read the various input files and merge with defaults.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Transfer-learning experiment JSON")
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    defaults_path = Path(__file__).resolve().parent / "input" / "transfer" / "defaults.json"
    config = merge_config(json.loads(defaults_path.read_text()), json.loads(input_path.read_text()))

    if config["trainer"]["device"] == "auto":
        config["trainer"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve absolute file paths, update config to use absolute paths also
    base_model_path = str(Path(config["base_model_path"]).expanduser().resolve())
    output_dir = str(Path(config["output_dir"]).expanduser().resolve())
    config.update(base_model_path=base_model_path, output_dir=output_dir)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load in the base model, and get its dtype, cutoff, and compute_nacs flag for later use
    base_model = torch.load(base_model_path, map_location="cpu", weights_only=False)
    model_dtype = next(base_model.parameters()).dtype
    model_compute_nacs = base_model.compute_nacs
    model_cutoff = float(base_model.r_max)
    torch.set_default_dtype(model_dtype)

    # For TL, we can expect mulltiple heads sometimes, so we normalise train paths to dictionaries 
    # Each dictionary will be a dict of head_name: path_to_data
    # If only one path is provided, it is also normalised to a dictionary with a single key "default"
    train_paths = config["train_path"]
    if isinstance(train_paths, str):
        train_paths = {"default": train_paths}

    # Resolve all the relative train paths to absolute paths, then update config to use absolute paths also
    train_paths = {head: str(Path(path).expanduser().resolve()) for head, path in train_paths.items()}
    config["train_path"] = train_paths

    # Read in the training data for each head from the xyz file paths
    train_atoms = {head: read(path, index=":") for head, path in train_paths.items()}

    # Check for validation path first, if not provided then build validation set based on validation fraction
    valid_paths = config.get("valid_path")
    split_record = None
    if valid_paths is not None:
        # Normalise again, accept dictionary or str validation paths 
        if isinstance(valid_paths, str):
            valid_paths = {"default": valid_paths}
        # Resolve full path, and update config to save
        valid_paths = {head: str(Path(path).expanduser().resolve()) for head, path in valid_paths.items()}
        config["valid_path"] = valid_paths
        valid_atoms = {head: read(path, index=":") for head, path in valid_paths.items()}
    else:
        config["valid_path"] = None
        valid_atoms = {}
        # Save the validation split records
        split_record = {"seed": seed, "valid_fraction": config["valid_fraction"], "heads": {}}
        for head, atoms in train_atoms.items():
            train_subset, valid_subset, train_indices, valid_indices = split_atoms(
                atoms, config["valid_fraction"], seed
            )
            # Update the train and validation atoms dict, as well as the split record for each head
            train_atoms[head] = train_subset
            valid_atoms[head] = valid_subset
            split_record["heads"][head] = {
                "source_path": str(train_paths[head]), "num_frames": len(atoms),
                "train_indices": train_indices, "valid_indices": valid_indices,
            }

    # For single head models, if E0s are not specified, we use the base models ones
    # For multi head models, E0s are different, so if unspecified, we will parse in None
    # ie the E0s will be calculated from the dataset instead 
    data_settings = deepcopy(config["data"])
    # Checks if multihead strategy is being done 
    expands_heads = any(
        settings["type"] in ("multihead", "multihead_correction")
        for settings in config["strategies"]
    )
    # If no E0s specified and single head training, then get E0s from base model
    if data_settings["E0s"] is None and not expands_heads:
        atomic_numbers = base_model.atomic_numbers.tolist()
        atomic_energies = (
            base_model.atomic_energies_fn.atomic_energies[0]
            .detach().cpu().tolist()
        )
        data_settings["E0s"] = {
            chemical_symbols[z]: value
            for z, value in zip(atomic_numbers, atomic_energies)
        }

    builder = AtomDataLoaderBuilder(cutoff=model_cutoff, **data_settings)
    train_loader = builder.load(
        train_atoms, batch_size=config["batch_size"], shuffle=True,
        seed=seed, balance_heads=config["balance_heads"],
    )
    valid_loader = builder.load(
        valid_atoms, batch_size=config["batch_size"], shuffle=False, balance_heads=False,
    )
    metadata = builder.get_metadata()
    for head in train_atoms:
        print(f"{head}: {len(train_atoms[head])} training geometries")
    for head in valid_atoms:
        print(f"{head}: {len(valid_atoms[head])} validation geometries")

    # Apply the strategies on the base model
    strategy_classes = {
        "naive": NaiveStrategy, "freeze": FreezeStrategy, "lora": LoRAStrategy,
        "multihead": MultiHeadStrategy, "multihead_correction": MultiHeadCorrectionStrategy,
    }
    model = base_model

    # Loops through all the strategies in the input file and applies them all in sequence 
    for settings in config["strategies"]:
        arguments = dict(settings)
        strategy_type = arguments.pop("type")
        if strategy_type in ("multihead", "multihead_correction"):
            arguments["metadata"] = metadata
        model = strategy_classes[strategy_type](**arguments).apply(model)

    trainer = Trainer(**config["trainer"])
    loss_fn = InvariantsWeightedEnergyForcesNacsDipoleLoss(**config["loss"]).to(trainer.device)

    # Record the starting model, resolved settings, and target-training metadata.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    models_dir = output_dir / "models"
    inputs_dir = output_dir / "input"
    for directory in (models_dir, inputs_dir, output_dir / "preds"):
        directory.mkdir()
    (inputs_dir / "input.json").write_text(input_path.read_text())
    (inputs_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
    torch.save(metadata, models_dir / "metadata.pt")
    if split_record is not None:
        (inputs_dir / "split.json").write_text(json.dumps(split_record, indent=2))

    model, history = trainer.train_model(
        model, train_loader, valid_loader, loss_fn,
        checkpoint_epoch=config["checkpoint_every"],
        checkpoint_models_dir=models_dir / "checkpoints", compute_nacs=model_compute_nacs,
    )
    model_filename = "best_model.pt" if trainer.restore_best else "final_model.pt"
    torch.save(model, models_dir / model_filename)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Model saved to: {models_dir / model_filename}")


if __name__ == "__main__":
    main()
