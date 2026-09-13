"""
Sample Script to Train one base model 
Reads in input JSON and defaults 
Outputs trained model and saves it

* STILL WORK IN PROGRESS
Main Training loop is in place but the args are not fixed yet 
More work for smoother integration with args etc
"""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from mace.data.atom_data_loader import AtomDataLoaderBuilder
from mace.modules.loss import InvariantsWeightedEnergyForcesNacsDipoleLoss
from mace.training.model_factory import initialise_autoencoder
from mace.training.trainer import Trainer


def merge_config(defaults, overrides):
    """
    Merge nested experiment overrides without changing the other defaults.
    Every experiment will use default parameters unless specifed otherwise
    """
    config = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = merge_config(config[key], value)
        else:
            config[key] = deepcopy(value)
    return config


def main():
    # Read the experiment settings and shared defaults.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Experiment JSON file")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    defaults_path = Path(__file__).resolve().parent / "input" / "defaults.json"
    overrides = json.loads(input_path.read_text())
    defaults = json.loads(defaults_path.read_text())
    config = merge_config(defaults, overrides)

    # Resolve experiment paths relative to the input JSON.
    train_path = (input_path.parent / Path(config["train_path"]).expanduser()).resolve()
    valid_path = (input_path.parent / Path(config["valid_path"]).expanduser()).resolve()
    output_dir = (input_path.parent / Path(config["output_dir"]).expanduser()).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if config["model"].get("load_base") is not None:
        raise ValueError("Base training requires model.load_base to be null.")
    if not config["trainer"].get("restore_best", True):
        raise ValueError("Base training requires trainer.restore_best to be true.")
    if not config["model"]["compute_nacs"]:
        raise ValueError("The current invariant loss requires compute_nacs=true.")

    # Seed and get default dtype
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(getattr(torch, config["dtype"]))

    # Reuse the training metadata, including E0s, for validation.
    builder = AtomDataLoaderBuilder(**config["data"])
    train_atoms = read(str(train_path), index=":")
    valid_atoms = read(str(valid_path), index=":")
    train_loader = builder.load(
        train_atoms, batch_size=config["batch_size"], shuffle=True, seed=seed
    )
    valid_loader = builder.load(valid_atoms, batch_size=config["batch_size"], shuffle=False)
    metadata = builder.get_metadata()

    # Initialise a fresh model, loss, and trainer.
    model = initialise_autoencoder(metadata, **config["model"])
    trainer = Trainer(**config["trainer"])
    loss_fn = InvariantsWeightedEnergyForcesNacsDipoleLoss(**config["loss"])
    loss_fn = loss_fn.to(trainer.device)

    # Record inputs before training starts; never overwrite an existing run.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "input.json").write_text(input_path.read_text())
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
    torch.save(metadata, output_dir / "metadata.pt")

    # The trainer handles optimisation, validation, EMA, and checkpoints.
    model, history = trainer.train_model(
        model,
        train_loader,
        valid_loader,
        loss_fn,
        checkpoint_epoch=config["checkpoint_every"],
        checkpoint_models_dir=output_dir / "checkpoints",
        compute_nacs=config["model"]["compute_nacs"],
    )

    # The trainer restores the best validation weights before returning.
    if history["best_epoch"] == 0:
        raise RuntimeError("Training produced no valid best model.")
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
