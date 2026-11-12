"""Shared helpers for training and testing scripts."""

import json
import math
from copy import deepcopy
from difflib import SequenceMatcher
from itertools import combinations

import numpy as np
import torch
from ase.io import write


def merge_config(defaults, overrides, run_only_keys=()):
    """
    In general, each training run will contain information from defaults.json and its individual
    run's input.json. This function merges the inputs from both jsons into one dictionary,
    with the input.json values overriding the default ones. 
    Every key in the run JSON must exist in defaults.json, except the top-level run_only_keys.
    """
    unknown = _find_unknown_keys(defaults, overrides, set(run_only_keys))
    if unknown:
        valid_paths = _key_paths(defaults) + sorted(run_only_keys)
        lines = [f"  '{path}'{_suggest_key(path, valid_paths)}" for path in unknown]
        raise ValueError(
            "Unknown key(s) in run config (not present in defaults.json):\n" + "\n".join(lines)
        )
    return _merge_dicts(defaults, overrides)


def _merge_dicts(defaults, overrides):
    config = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = _merge_dicts(config[key], value)
        else:
            config[key] = deepcopy(value)
    return config


def _find_unknown_keys(defaults, overrides, allowed=frozenset(), prefix=""):
    # Only descend where the default is a dict; null/list defaults (e.g. E0s) take free-form values.
    unknown = []
    for key, value in overrides.items():
        path = prefix + key
        if key in allowed:
            continue
        if key not in defaults:
            unknown.append(path)
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            unknown += _find_unknown_keys(defaults[key], value, prefix=path + ".")
    return unknown


def _key_paths(defaults, prefix=""):
    paths = []
    for key, value in defaults.items():
        paths.append(prefix + key)
        if isinstance(value, dict):
            paths += _key_paths(value, prefix + key + ".")
    return paths


def _suggest_key(path, valid_paths, cutoff=0.6):
    name = path.rsplit(".", 1)[-1]
    score, best = max(
        (SequenceMatcher(None, name, valid.rsplit(".", 1)[-1]).ratio(), valid)
        for valid in valid_paths
    )
    return f" -> did you mean '{best}'?" if score >= cutoff else ""


def split_atoms(atoms, valid_fraction, seed):
    """
    Split xyz files into training and validation sets based on a specified validation fraction
    First light validations on the input
    Then uses rng to randomly split the indices of the xyz file
    Return the indices also so they can be recorded and saved
    """
    if isinstance(valid_fraction, bool) or not isinstance(valid_fraction, (int, float)):
        raise TypeError("valid_fraction must be a number between 0 and 1.")
    if not math.isfinite(valid_fraction) or not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1.")

    num_frames = len(atoms)
    if num_frames < 2:
        raise ValueError("Automatic splitting requires at least two geometries.")

    num_valid = math.ceil(num_frames * valid_fraction)
    if num_valid >= num_frames:
        raise ValueError("valid_fraction leaves no geometries for training.")

    indices = np.random.default_rng(seed).permutation(num_frames)
    valid_indices = indices[:num_valid].tolist()
    train_indices = indices[num_valid:].tolist()

    train_atoms = [atoms[i] for i in train_indices]
    valid_atoms = [atoms[i] for i in valid_indices]

    return train_atoms, valid_atoms, train_indices, valid_indices


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
