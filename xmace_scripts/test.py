"""Test selected models against named datasets and save predictions and MAEs.

Run from xmace_scripts: python test.py input/test/ethene.json outputs/base_000
User paths are relative to the working directory. Selected tests always rerun
and replace their outputs. Do not run concurrent tests against the same destinations.
Errors stop execution and display Python's traceback.
Models in a run must share architecture, dtype, elements, cutoff, and head mapping.
"""

import argparse
import json
from pathlib import Path
from unittest import loader

import torch
from ase.data import chemical_symbols
from ase.io import read

from mace.data.atom_data_loader import AtomDataLoaderBuilder
from mace.testing import Tester

if __package__:
    from .utils import load_models, merge_config, save_results
else:
    from utils import load_models, merge_config, save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Test settings JSON")
    parser.add_argument("run_dir", type=Path, help="Run directory containing models/ and preds/")
    args = parser.parse_args()

    # Read the experiment settings and shared defaults.
    input_path = args.input.expanduser().resolve()
    defaults_path = Path(__file__).resolve().parent / "input" / "test" / "defaults.json"
    overrides = json.loads(input_path.read_text())
    defaults = json.loads(defaults_path.read_text())
    config = merge_config(defaults, overrides)

    # Resolve automatic device selection before constructing Tester.
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve relative paths to absolute paths
    run_dir = args.run_dir.expanduser().resolve()
    for dataset in config["datasets"].values():
        dataset["path"] = Path(dataset["path"]).expanduser().resolve()

    # Load in models
    models = load_models(run_dir, config["models"])

    # Use one reference model to get metadata information
    reference_model = next(iter(models.values()))
    atomic_numbers = reference_model.atomic_numbers.tolist()
    cutoff = float(reference_model.r_max)
    atomic_energies = reference_model.atomic_energies_fn.atomic_energies.detach().cpu().numpy()
    # Follow model's dtype
    torch.set_default_dtype(next(reference_model.parameters()).dtype)
    
    # Load in training metadata from run directory to get head mapping later
    training_metadata = torch.load(run_dir / "models" / "metadata.pt", 
                                   map_location="cpu", 
                                   weights_only=False)

    """
    For testing, one head is being tested at a time. This head is chosen based on the user 
    specified input, so every dataset will have one head tied to it. If unspecified, for single
    head training, then it defaults to 0. If not, we get the head index from the training
    metadata. This head is then used to construct the E0s.
    Later we specify the head for each graph in the dataloader to ensure that the correct
    head is used for testing. 
    Slightly suboptimal routine but its mainly due to the fact that our model choses the path
    for which head to take based on the dataset, which is done so that one model can train on
    different pieces of data simultaneously. For testing, we are only testing one head at a time
    so we can just specify it and manually insert it into the dataloader.
    """
    # Loop through all the datasets provided
    for dataset_name, dataset in config["datasets"].items():
        atoms_list = read(dataset["path"], index=":")

        # If no head is specified, default to 0 
        # Head selection is specific to each dataset
        head = (
            0 if dataset.get("head") is None
            else training_metadata.head_to_index[dataset["head"]]
        )

        # Construct E0s from the reference model and the head used
        e0s = {chemical_symbols[z]: float(value)
               for z, value in zip(atomic_numbers, atomic_energies[head])}

        # Construct the builder and dataloader 
        builder = AtomDataLoaderBuilder(
            cutoff=cutoff, 
            E0s=e0s, 
            energy_difference_references=None, 
            **config["labels"]
        )
        loader = builder.load(
            atoms_list, 
            batch_size=config["batch_size"], 
            shuffle=False, 
            balance_heads=False
        )
        # Replace the builder's local index with the trained model index.
        for graph in loader.dataset:
            graph.head = torch.tensor(head, dtype=torch.long)

        # Light validation check against the reference model to ensure dataset is compatible 
        # with the model
        test_metadata = builder.get_metadata()
        if test_metadata.atomic_numbers != atomic_numbers:
            raise ValueError("Test elements must match the model's element ordering.")
        if test_metadata.n_energies != reference_model.n_energies:
            raise ValueError("Test state count must match the model.")

        # Loop through all the models and run the various tests 
        for model_name, model in models.items():
            print(f"RUN {model_name} / {dataset_name}", flush=True)
            tester = Tester(device=config["device"])
            with torch.enable_grad():
                tester.run_test(model, loader, compute_nacs=config["compute_nacs"])
            output_dir = run_dir / "preds" / model_name / dataset_name
            save_results(output_dir, atoms_list, tester, config["compute_nacs"])
            model.cpu()
            print(f"DONE {model_name} / {dataset_name}", flush=True)


if __name__ == "__main__":
    main()
