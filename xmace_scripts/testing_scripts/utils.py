"""Settings, model paths, and exports for the testing entrypoint."""

import json
from copy import deepcopy
from itertools import combinations

import torch
from ase.io import write


def merge_config(defaults, overrides):
    """Recursively merge experiment overrides into a copy of the defaults."""
    config = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = merge_config(config[key], value)
        else:
            config[key] = deepcopy(value)
    return config


def load_models(run_dir, selection):
    """
    Based on the selection requested to test, load the relevant models 
    The final or best models are saved as full models (from base scripts)
    But the checkpoints are saved as weights only 
    First we load in the base/final models, then load in any requested checkpoints
    using either the final or best model as a template
    
    """
    models_dir = run_dir / "models"
    models = {}

    # Load in best / final models 
    for name in ("best", "final"):
        if selection[name]:
            path = models_dir / f"{name}_model.pt"
            models[name] = torch.load(path, map_location="cpu", weights_only=False).eval()

    # For every checkpoint, load in model
    for epoch in sorted(set(selection["checkpoints"])):
        # First find the template model, either best or final, whichever is available
        template = models_dir / "best_model.pt"
        if not template.is_file():
            template = models_dir / "final_model.pt"

        # Load in the template model
        model = torch.load(template, map_location="cpu", weights_only=False)
        # Load in the checkpoint weights 
        checkpoint_path = models_dir / "checkpoints" / f"checkpoint_epoch_{epoch:06d}.pt"
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Copy over checkpoint weights and set model to eval mode 
        model.load_state_dict(weights, strict=True)
        models[checkpoint_path.stem] = model.eval()

    if not models:
        raise ValueError("No models or checkpoints selected.")

    return models


def collect_metrics(tester, num_geometries, compute_nacs):
    metrics = {
        "num_geometries": num_geometries,
        "state_indices": list(range(tester.ref_energies.shape[-1])),
        "energy": {"unit": "eV", "mae": tester.get_energy_mae(),
                   "mae_by_state": tester.get_energy_mae_by_state().tolist()},
        "forces": {"unit": "eV/angstrom", "mae": tester.get_force_mae(),
                   "mae_by_state": tester.get_force_mae_by_state().tolist()},
    }
    if compute_nacs:
        metrics["state_pairs"] = list(combinations(metrics["state_indices"], 2))
        metrics["raw_nacs"] = {
            "unit": "1/angstrom", "phase_mae": tester.get_raw_nac_phase_mae(),
            "phase_mae_by_pair": tester.get_raw_nac_phase_mae_by_pair().tolist(),
        }
        metrics["smooth_nacs"] = {
            "unit": "eV/angstrom", "phase_mae": tester.get_smooth_nac_phase_mae(),
            "phase_mae_by_pair": tester.get_smooth_nac_phase_mae_by_pair().tolist(),
        }
    return metrics


def save_results(output_dir, atoms_list, tester, compute_nacs):
    metrics = collect_metrics(tester, len(atoms_list), compute_nacs)
    metrics_text = json.dumps(metrics, indent=2, allow_nan=False)
    predictions = []
    offset = 0
    for index, atoms in enumerate(atoms_list):
        end = offset + len(atoms)
        output = atoms.copy()
        output.info = deepcopy(atoms.info)
        # Replace only our prediction fields, including stale NACs from an earlier export.
        for key in ("PRED_energy", "PRED_forces", "PRED_nacs", "PRED_smooth_nacs"):
            output.info.pop(key, None)
        output.info["PRED_energy"] = tester.pred_energies[index:index + 1]
        output.info["PRED_forces"] = tester.pred_forces[offset:end]
        if compute_nacs:
            output.info["PRED_nacs"] = tester.pred_nacs[offset:end]
            output.info["PRED_smooth_nacs"] = tester.pred_smooth_nacs[offset:end]
        predictions.append(output)
        offset = end

    output_dir.mkdir(parents=True, exist_ok=True)
    xyz_temp = output_dir / "predictions.xyz.tmp"
    metrics_temp = output_dir / "metrics.json.tmp"
    try:
        write(xyz_temp, predictions, format="extxyz", write_results=False)
        metrics_temp.write_text(metrics_text)
        # Invalidate an incomplete old pair before publishing the new one.
        (output_dir / "metrics.json").unlink(missing_ok=True)
        xyz_temp.replace(output_dir / "predictions.xyz")
        metrics_temp.replace(output_dir / "metrics.json")
    finally:
        xyz_temp.unlink(missing_ok=True)
        metrics_temp.unlink(missing_ok=True)
