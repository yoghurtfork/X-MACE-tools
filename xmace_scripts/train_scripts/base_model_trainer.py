"""
Sample Script to Train one base model 
Reads in input JSON and defaults 
Outputs trained model and saves it

In general, the default.json contains all the default parameters for training, 
each dictionary key contains the arguments for each of the moudles in the training pipeline.
To run individual experiments or runs, modify the input.json to override the defaults 
The overwritten updated parameters will then be parsed into the training pipeline 
Each input.json must additionally also contain the paths to the training and validation data,
as well as the output directory to save the trained model and checkpoints. 
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from utils import merge_config

from mace.data.atom_data_loader import AtomDataLoaderBuilder
from mace.modules.loss import InvariantsWeightedEnergyForcesNacsDipoleLoss
from mace.training.model_factory import FOUNDATION_MODELS, initialise_autoencoder
from mace.training.trainer import Trainer


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

    # Resolve automatic device selection before constructing Trainer or saving settings.
    if config["trainer"]["device"] == "auto":
        config["trainer"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve user paths relative to the directory where the command is run.
    train_path = Path(config["train_path"]).expanduser().resolve()
    valid_path = Path(config["valid_path"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    config.update(train_path=str(train_path), valid_path=str(valid_path), output_dir=str(output_dir))

    # For foundation model, normalise and resolve if a path is provided 
    load_base = config["model"].get("load_base")
    if load_base is not None and load_base not in FOUNDATION_MODELS:
        config["model"]["load_base"] = str(Path(load_base).expanduser().resolve())

    # Check that the output directory does not exist, or is empty
    # Safeguard to avoid overwriting existing runs
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
    
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
    output_dir.mkdir(parents=True, exist_ok=True)
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

    # Save the best or final model to the output directory
    model_filename = "best_model.pt" if trainer.restore_best else "final_model.pt"
    torch.save(model, output_dir / model_filename)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Model saved to: {output_dir / model_filename}")


if __name__ == "__main__":
    main()
